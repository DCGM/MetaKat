import json
import logging
import types
from unittest import mock

import pytest
from doc_worker.doc_worker_wrapper import WorkerResponse

from metakat.worker.docapi import metakat_worker as worker_module
from metakat.worker.docapi.job_summary import render_log, render_mail, render_text, summarize_job
from metakat.worker.docapi.metakat_worker import MetakatWorker

WORKER = "metakat 1234567, docapi 89abcde, host"


def _text(value):
    return [{"text": value, "confidence": 0.9, "lang": None, "id": f"v-{value}"}]


def _page(index, parent, page_type="normalPage", number=True):
    return {"type": "page", "id": f"p{index}", "batch_index": index, "parent_id": parent, "pageIndex": 1,
            "pageType": {"text": page_type}, "pageNumber": {"text": str(index)} if number else None}


def _report(stage="biblio", core_engine="biblio_core_engine_vlm_periodical", attempts=1, status="ok",
            cost=None, provider="Azure", zdr=True, images=()):
    return {
        "stage": stage, "core_engine": core_engine, "api_url": "https://openrouter.ai/api/v1",
        "model": "openai/gpt-6-luna", "reasoning_effort": "medium", "status": status, "duration_s": 10.0,
        "cost": cost if cost is not None else {"amount": 0.001, "currency": "USD", "source": "provider"},
        "zdr": zdr, "images": list(images),
        "attempts": [{"status": "ok", "provider": provider}] * attempts,
    }


def _image(sent, original, text=None):
    return {"page_id": "p", "sent": list(sent), "original": list(original), "smallest_text_px": text}


def _periodical():
    """A title of two volumes: two issues, the second with a supplement, then one issue."""
    elements = [
        {"type": "title", "id": "t", "title": _text("Title"), "language": _text("cze")},
        {"type": "volume", "id": "v1", "parent_id": "t", "dateIssued": _text("1900")},
        {"type": "volume", "id": "v2", "parent_id": "t", "dateIssued": _text("1901")},
        {"type": "issue", "id": "i1", "parent_id": "v1", "partNumber": _text("1")},
        {"type": "issue", "id": "i2", "parent_id": "v1", "partNumber": _text("2")},
        {"type": "supplement", "id": "s1", "parent_id": "i2", "title": _text("Supplement")},
        {"type": "issue", "id": "i3", "parent_id": "v2", "partNumber": _text("1"), "groups": [{"id": "g"}]},
    ]
    pages = [("i1", "titlePage"), ("i1", "normalPage"), ("i2", "titlePage"), ("i2", "blank"),
             ("s1", "normalPage"), ("i3", "titlePage"), ("i3", "normalPage"), ("i3", "map")]
    elements += [_page(index, parent, page_type, number=index % 2 == 0)
                 for index, (parent, page_type) in enumerate(pages)]
    reports = [
        _report(attempts=2, images=[_image((1000, 1500), (2000, 3000), 40.0)]),
        _report(provider="Novita", zdr=False, images=[_image((1100, 1700), (1100, 1700), 30.0)]),
        _report(status="failed", cost={"amount": None, "currency": None, "source": None}),
    ]
    return {"elements": elements, "vlm_processing_reports": reports}


def _job(pages=8, chapter=False):
    definition = {"biblio": {}, **({"chapter": {}} if chapter else {})}
    return types.SimpleNamespace(id="3f2a9c1e-7d4b-4c1a-9e2f-0a6b5c8d9e10", engine_name="periodical",
                                 engine_version="v1.7.0", images=[None] * pages, engine_definition=definition,
                                 alto_required=True, page_required=False)


def _job_dir(tmp_path, data=None, mods=None, alto=8, meta=None):
    if data is not None:
        (tmp_path / "result").mkdir()
        (tmp_path / "result" / "metakat.json").write_text(json.dumps(data))
        if mods is not None:
            (tmp_path / "result" / "mods").mkdir()
            for element_id in mods:
                (tmp_path / "result" / "mods" / f"{element_id}.xml").write_text("<mods/>")
    (tmp_path / "alto").mkdir()
    for index in range(alto):
        (tmp_path / "alto" / f"{index}.xml").write_text("<alto/>")
    if meta is not None:
        (tmp_path / "meta.json").write_text(json.dumps(meta))
    return tmp_path


def _all_ids(data):
    return [element["id"] for element in data["elements"]]


def _summary(tmp_path, data, response=None, chapter=False, pages=8, mods="all", metadata=None):
    job_dir = _job_dir(tmp_path, data, mods=_all_ids(data) if mods == "all" else mods)
    return summarize_job(_job(pages, chapter), response or WorkerResponse.ok(), 125.0, job_dir=job_dir,
                         has_chapter_stage=chapter, worker=WORKER, metadata=metadata)


def test_the_header_names_the_job_its_engine_and_worker(tmp_path):
    subject, body = render_mail(_summary(tmp_path, _periodical()))

    assert subject == "[MetaKat] OK job 3f2a9c1e, 8 pages, 4 documents"
    assert body.startswith(
        "job       3f2a9c1e-7d4b-4c1a-9e2f-0a6b5c8d9e10\n"
        "engine    periodical v1.7.0\n"
        "worker    metakat 1234567, docapi 89abcde, host\n"
        "status    OK, 2m 05s\n\n")


def test_the_input_lists_the_images_alto_metadata_and_override_settings_without_secret_values(tmp_path):
    metadata = {"metakat_json": {}, "proarc_json": None, "engine_config_override": {
        "biblio": {"core": {"vlm": {"api_key": "sk-secret", "reasoning_effort": "high",
                                    "image_preprocessing": {"max_side": 2048}}}}}}
    body = render_text(_summary(tmp_path, _periodical(), metadata=metadata))

    assert ("input\n"
            " images   8\n"
            " alto     8\n"
            " meta     metakat_json\n"
            " override biblio.core.vlm.api_key\n"
            "          biblio.core.vlm.reasoning_effort = high\n"
            "          biblio.core.vlm.image_preprocessing.max_side = 2048\n\n") in body
    assert "sk-secret" not in body


def test_settings_below_a_secret_key_show_their_path_only(tmp_path):
    metadata = {"engine_config_override": {"credentials": {"token": {"value": "tok-secret"}}, "vlm": {"model": "m"}}}
    body = render_text(_summary(tmp_path, _periodical(), metadata=metadata))

    assert " override credentials.token.value\n          vlm.model = m\n\n" in body
    assert "tok-secret" not in body


def test_the_output_compares_json_and_mods_and_counts_pages_types_and_fields(tmp_path):
    data = _periodical()
    exported = [element_id for element_id in _all_ids(data) if element_id != "s1"]
    body = render_text(_summary(tmp_path, data, mods=exported))

    assert ("output\n"
            " json     1 title, 2 volumes, 3 issues,\n"
            "          1 supplement, 8 pages\n"
            " mods     1 title, 2 volumes, 3 issues,\n"
            "          0 supplements, 8 pages\n"
            " numbered 4 pages\n"
            " types    3 titlePage, 1 blank, 1 map\n"
            " fields   2 title, 2 volume, 3 issue,\n"
            "          1 supplement\n\n") in body


def test_a_job_without_mods_says_so(tmp_path):
    assert " mods     none\n" in render_text(_summary(tmp_path, _periodical(), mods=None))


def test_the_documents_are_a_tree_of_their_ancestors(tmp_path):
    body = render_text(_summary(tmp_path, _periodical()))

    assert ("documents\n"
            " title 1\n"
            "  volume 1\n"
            "   1 issue        p. 1-2\n"
            "   2 issue        p. 3-4\n"
            "    3 supplement  p. 5\n"
            "  volume 2\n"
            "   4 issue        p. 6-8\n\n") in body


def test_a_chapter_engine_counts_nested_chapters_and_wraps_long_page_ranges(tmp_path):
    data = {"elements": [
        {"type": "volume", "id": "v"},
        {"type": "chapter", "id": "c1", "parent_id": "v", "title": _text("One")},
        {"type": "chapter", "id": "c2", "parent_id": "c1", "title": _text("Two")},
        {"type": "volume", "id": "w"},
        *[_page(index, parent) for index, parent in enumerate("vvwvvwvvwv")],
        {"type": "page", "id": "loose", "batch_index": 10, "parent_id": None},
    ]}
    body = render_text(_summary(tmp_path, data, chapter=True, pages=11))

    assert ("documents\n"
            " 1 volume  p. 1-2, 4-5, 7-8, 10\n"
            "                       ch. 2\n"
            " 2 volume  p. 3, 6, 9  ch. 0\n"
            " unassigned pages 1\n") in body


def test_the_vlm_total_and_each_engine_have_calls_images_and_time(tmp_path):
    body = render_text(_summary(tmp_path, _periodical()))

    assert ("vlm\n"
            " calls    3, 1 failed, 1 retry\n"
            " images   2, max 1100x1700, min text 20.0 px\n"
            " time     30.0 s, 0.0020 USD, partly unknown\n\n"
            "biblio/biblio_core_engine_vlm_periodical\n"
            " api      https://openrouter.ai/api/v1\n"
            " model    openai/gpt-6-luna medium\n"
            " provider 3 Azure, 1 Novita, 2 of 3 ZDR\n"
            " calls    3, 1 failed, 1 retry\n") in body


@pytest.mark.parametrize("cost, expected", [
    ({"amount": 0.002, "currency": "USD", "source": "config"}, "0.0020 USD (estimated)"),
    ({"amount": 0.0, "currency": "USD", "source": "local"}, "local"),
    ({"amount": None, "currency": None, "source": None}, "cost unknown"),
])
def test_estimated_local_and_unknown_costs_are_marked(tmp_path, cost, expected):
    data = _periodical()
    data["vlm_processing_reports"] = [_report(cost=cost)]
    assert f" time     10.0 s, {expected}\n" in render_text(_summary(tmp_path, data))


def test_a_failed_job_carries_its_error_and_traceback_but_not_in_the_user_text(tmp_path):
    try:
        raise ValueError("broken")
    except ValueError as error:
        response = WorkerResponse.fail("Failed to download job data from server", exception=error)
    summary = summarize_job(_job(), response, 3.0, job_dir=_job_dir(tmp_path, alto=3), has_chapter_stage=False,
                            worker=WORKER)

    subject, body = render_mail(summary)
    assert subject == "[MetaKat] FAILED job 3f2a9c1e"
    assert " alto     3\n" in body and "output" not in body and "vlm" not in body
    assert "error\nFailed to download job data from server\nValueError: broken\n\nTraceback (most recent" in body
    user_text = render_text(summary, with_error=False)
    assert "status    FAILED, 0m 03s\n" in user_text
    assert "error" not in user_text and "broken" not in user_text
    assert "error: Failed to download job data from server | ValueError: broken" in render_log(summary)
    assert "Traceback" not in render_log(summary)


def test_the_log_has_a_line_for_the_job_each_document_and_each_engine(tmp_path):
    lines = render_log(_summary(tmp_path, _periodical())).splitlines()

    assert lines[0].startswith("Job summary: job 3f2a9c1e-7d4b-4c1a-9e2f-0a6b5c8d9e10 OK | engine periodical v1.7.0"
                               " | worker metakat 1234567, docapi 89abcde, host | duration 2m 05s"
                               " | input images 8, alto 8, meta none | json 1 title")
    assert lines[1:5] == [
        "  document 1: issue < volume 1 < title 1, pages 1-2",
        "  document 2: issue < volume 1 < title 1, pages 3-4",
        "  document 3: supplement < issue 2 < volume 1 < title 1, pages 5",
        "  document 4: issue < volume 2 < title 1, pages 6-8",
    ]
    assert lines[5].startswith("  vlm biblio/biblio_core_engine_vlm_periodical: api https://openrouter.ai/api/v1")
    assert len(lines) == 6


@pytest.fixture
def worker():
    worker = object.__new__(MetakatWorker)
    worker.mail_sender = mock.Mock()
    with mock.patch.object(worker_module, "worker_description", return_value=WORKER):
        yield worker


def test_the_worker_logs_the_summary_into_the_job_log_and_mails_it(tmp_path, worker, caplog):
    _job_dir(tmp_path, _periodical(), mods=_all_ids(_periodical()),
             meta={"metakat_json": {}, "engine_config_override": {"biblio": {"core": {"vlm": {"model": "m"}}}}})
    job_log = logging.FileHandler(tmp_path / "worker.log")

    with caplog.at_level(logging.INFO):
        worker.on_job_finished(_job(), WorkerResponse.ok(), str(tmp_path), 10.0, job_log)
    job_log.close()

    assert "Job summary: job 3f2a9c1e" in caplog.text
    assert "Job summary: job 3f2a9c1e" in (tmp_path / "worker.log").read_text()
    assert job_log not in logging.getLogger().handlers
    subject, body = worker.mail_sender.send.call_args.args[:2]
    assert subject == "[MetaKat] OK job 3f2a9c1e, 8 pages, 4 documents"
    assert " meta     metakat_json\n override biblio.core.vlm.model = m\n" in body


def test_without_a_mail_sender_the_worker_only_logs(tmp_path, worker, caplog):
    worker.mail_sender = None

    with caplog.at_level(logging.INFO):
        worker.on_job_finished(_job(), WorkerResponse.fail("Failed to upload results to server"), None, 1.0, None)

    assert "Job summary: job 3f2a9c1e-7d4b-4c1a-9e2f-0a6b5c8d9e10 FAILED" in caplog.text


def test_the_api_gets_the_same_summary_in_both_logs_of_a_processed_job(tmp_path, worker):
    _job_dir(tmp_path, _periodical(), mods=_all_ids(_periodical()))

    log, log_user = worker.job_api_logs(_job(), WorkerResponse.ok(), str(tmp_path), 10.0)

    assert log == log_user
    assert log.startswith("job       3f2a9c1e-7d4b-4c1a-9e2f-0a6b5c8d9e10\n")
    assert "    3 supplement  p. 5\n" in log


def test_the_user_log_of_a_failed_job_leaves_out_its_error(tmp_path, worker):
    try:
        raise ValueError("broken")
    except ValueError as error:
        response = WorkerResponse.fail("MetakatWorker processing failed", exception=error)

    log, log_user = worker.job_api_logs(_job(), response, str(tmp_path), 10.0)

    assert "status    FAILED, 0m 10s\n" in log and "status    FAILED, 0m 10s\n" in log_user
    assert "error\nMetakatWorker processing failed\nValueError: broken" in log
    assert "Traceback (most recent call last):" in log
    assert "error" not in log_user and "Traceback" not in log_user and "broken" not in log_user
