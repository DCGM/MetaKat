import json
import logging
import types
from unittest import mock

from doc_worker.doc_worker_wrapper import WorkerResponse

from metakat.worker.docapi.job_summary import render_log, render_mail, summarize_job
from metakat.worker.docapi.metakat_worker import MetakatWorker


def _text(value):
    return [{"text": value, "confidence": 0.9, "lang": None, "id": f"v-{value}"}]


def _page(index, parent):
    return {"type": "page", "id": f"p{index}", "batch_index": index, "parent_id": parent, "pageIndex": 1}


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
    """A periodical title of two volumes, the first with two issues and the second with one."""
    elements = [
        {"type": "title", "id": "t", "title": _text("Title"), "language": _text("cze")},
        {"type": "volume", "id": "v1", "parent_id": "t", "dateIssued": _text("1900")},
        {"type": "volume", "id": "v2", "parent_id": "t", "dateIssued": _text("1901")},
        {"type": "issue", "id": "i1", "parent_id": "v1", "partNumber": _text("1")},
        {"type": "issue", "id": "i2", "parent_id": "v1", "partNumber": _text("2")},
        {"type": "issue", "id": "i3", "parent_id": "v2", "partNumber": _text("1"), "groups": [{"id": "g"}]},
    ]
    parents = ["i1", "i1", "i2", "i2", "i3", "i3"]
    elements += [_page(index, parent) for index, parent in enumerate(parents)]
    reports = [
        _report(attempts=2, images=[_image((1000, 1500), (2000, 3000), 40.0)]),
        _report(provider="Novita", zdr=False, images=[_image((1100, 1700), (1100, 1700), 30.0)]),
        _report(status="failed", cost={"amount": None, "currency": None, "source": None}),
    ]
    return {"elements": elements, "vlm_processing_reports": reports}


def _job(pages=6, chapter=False):
    definition = {"biblio": {}, **({"chapter": {}} if chapter else {})}
    return types.SimpleNamespace(id="3f2a9c1e-7d4b-4c1a-9e2f-0a6b5c8d9e10", engine_name="periodical",
                                 engine_version="v1.7.0", images=[None] * pages, engine_definition=definition)


def _summary(tmp_path, data, response=None, chapter=False, pages=6):
    path = tmp_path / "metakat.json"
    path.write_text(json.dumps(data))
    return summarize_job(_job(pages, chapter), response or WorkerResponse.ok(), 125.0, path, chapter)


def test_a_periodical_lists_its_issues_under_numbered_volumes(tmp_path):
    subject, body = render_mail(_summary(tmp_path, _periodical()))

    assert subject == "[MetaKat] OK job 3f2a9c1e, 6 pages, 3 documents"
    assert "documents 1 title, 2 volumes, 3 issues\n" in body
    assert (" 1 issue < volume 1 < title 1  p. 1-2\n"
            " 2 issue < volume 1 < title 1  p. 3-4\n"
            " 3 issue < volume 2 < title 1  p. 5-6\n") in body
    assert "ch." not in body
    assert "duration  2m 05s\n" in body
    assert "fields    7 (title 2, volume 2, issue 3)\n" in body


def test_the_vlm_calls_of_an_engine_are_summed_into_one_block(tmp_path):
    body = render_mail(_summary(tmp_path, _periodical()))[1]

    assert "vlm       3 calls, 1 failed, 1 retry\n          2 images, 30.0 s, 0.0020 USD, partly unknown\n" in body
    assert "biblio/biblio_core_engine_vlm_periodical\n" in body
    assert " provider Azure 3, Novita 1, ZDR 2 of 3\n" in body
    assert " calls    3, 1 failed, 1 retry\n" in body
    # the biggest image sent, and the smallest text in pixels as sent: 40 px of 3000 sent at 1500 is 20
    assert " images   2, max 1100x1700, min text 20.0 px\n" in body


def test_a_chapter_engine_counts_nested_chapters_under_their_document(tmp_path):
    data = {"elements": [
        {"type": "volume", "id": "v"},
        {"type": "chapter", "id": "c1", "parent_id": "v", "title": _text("One")},
        {"type": "chapter", "id": "c2", "parent_id": "c1", "title": _text("Two")},
        {"type": "volume", "id": "w"},
        _page(0, "v"), _page(1, "v"), _page(2, "w"), _page(3, "v"),
    ]}
    body = render_mail(_summary(tmp_path, data, chapter=True, pages=4))[1]

    assert "documents 2 volumes, 2 chapters\n" in body
    assert " 1 volume  p. 1-2, 4  ch. 2\n 2 volume  p. 3       ch. 0\n" in body


def test_estimated_and_local_costs_are_marked(tmp_path):
    data = _periodical()
    data["vlm_processing_reports"] = [_report(cost={"amount": 0.002, "currency": "USD", "source": "config"})]
    assert " time     10.0 s, 0.0020 USD (estimated)\n" in render_mail(_summary(tmp_path, data))[1]
    data["vlm_processing_reports"] = [_report(cost={"amount": 0.0, "currency": "USD", "source": "local"})]
    assert " time     10.0 s, local\n" in render_mail(_summary(tmp_path, data))[1]


def test_a_failed_job_carries_its_error_and_traceback(tmp_path):
    try:
        raise ValueError("broken")
    except ValueError as error:
        response = WorkerResponse.fail("Failed to download job data from server", exception=error)

    summary = summarize_job(_job(), response, 3.0, None, False)
    subject, body = render_mail(summary)

    assert subject == "[MetaKat] FAILED job 3f2a9c1e"
    assert "status    FAILED\n" in body
    assert "documents" not in body and "vlm" not in body
    assert "error\nFailed to download job data from server\nValueError: broken\n\nTraceback (most recent call last):" in body
    assert "error: Failed to download job data from server | ValueError: broken" in render_log(summary)
    assert "Traceback" not in render_log(summary)


def test_the_log_has_a_line_for_the_job_each_document_and_each_engine(tmp_path):
    lines = render_log(_summary(tmp_path, _periodical())).splitlines()

    assert lines[0].startswith("Job summary: job 3f2a9c1e-7d4b-4c1a-9e2f-0a6b5c8d9e10 OK | engine periodical v1.7.0")
    assert lines[1] == "  document 1: issue < volume 1 < title 1, pages 1-2"
    assert lines[4].startswith("  vlm biblio/biblio_core_engine_vlm_periodical: api https://openrouter.ai/api/v1")
    assert len(lines) == 5


def test_the_worker_logs_the_summary_into_the_job_log_and_mails_it(tmp_path, caplog):
    result = tmp_path / "result"
    result.mkdir()
    (result / "metakat.json").write_text(json.dumps(_periodical()))
    worker = object.__new__(MetakatWorker)
    worker.mail_sender = mock.Mock()
    job_log = logging.FileHandler(tmp_path / "worker.log")

    with caplog.at_level(logging.INFO):
        worker.on_job_finished(_job(), WorkerResponse.ok(), str(tmp_path), 10.0, job_log)
    job_log.close()

    assert "Job summary: job 3f2a9c1e" in caplog.text
    assert "Job summary: job 3f2a9c1e" in (tmp_path / "worker.log").read_text()
    assert job_log not in logging.getLogger().handlers
    subject, body = worker.mail_sender.send.call_args.args[:2]
    assert subject == "[MetaKat] OK job 3f2a9c1e, 6 pages, 3 documents"
    assert "issue < volume 2 < title 1" in body


def test_without_a_mail_sender_the_worker_only_logs(tmp_path, caplog):
    worker = object.__new__(MetakatWorker)
    worker.mail_sender = None

    with caplog.at_level(logging.INFO):
        worker.on_job_finished(_job(), WorkerResponse.fail("Failed to upload results to server"), None, 1.0, None)

    assert "Job summary: job 3f2a9c1e-7d4b-4c1a-9e2f-0a6b5c8d9e10 FAILED" in caplog.text
