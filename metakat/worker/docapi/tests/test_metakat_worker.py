import json
import logging
import types
from pathlib import Path
from unittest import mock

import pytest

from metakat.worker.docapi import metakat_worker as worker_module
from metakat.worker.docapi.config import Config, LoggingConfig
from metakat.worker.docapi.metakat_worker import MetakatWorker
from doc_worker.doc_worker_wrapper import WorkerResponseStatus


def _job(engine_definition):
    return types.SimpleNamespace(
        images=[types.SimpleNamespace(name="page.jpg", order=0)],
        engine_definition=engine_definition,
        id="job-1",
        engine_name="monograph",
        engine_version="v1.5.0",
    )


# Config requires a key; these tests never contact DocAPI.
KEY = {"WORKER_KEY": "metakat.testkid.testsecret"}


@pytest.fixture(autouse=True)
def worker_key(monkeypatch):
    monkeypatch.setenv("WORKER_KEY", KEY["WORKER_KEY"])


@pytest.fixture
def worker():
    worker = object.__new__(MetakatWorker)
    return worker


@pytest.fixture
def workspace(tmp_path):
    """The images/alto/result/engines layout process_job expects."""
    images = tmp_path / "images"
    altos = tmp_path / "alto"
    result = tmp_path / "result"
    engines = tmp_path / "engines"
    for directory in (images, altos, result, engines):
        directory.mkdir()
    (images / "page.jpg").touch()
    (altos / "page.xml").write_text("<alto/>", encoding="utf-8")
    return images, altos, result, engines


def test_store_metakat_pdf_defaults_false_and_reads_environment():
    with mock.patch.dict("os.environ", KEY, clear=True):
        assert not Config().STORE_METAKAT_PDF
    with mock.patch.dict("os.environ", {**KEY, "STORE_METAKAT_PDF": "true"}, clear=True):
        assert Config().STORE_METAKAT_PDF


def test_metakat_settings_reach_the_worker(tmp_path):
    environment = {**KEY, "BASE_DIR": str(tmp_path), "ALLOWED_IMAGE_EXTENSIONS": ".png, .tif",
                   "STORE_METAKAT_PDF": "true", "STORE_MODS_OVERVIEW": "true"}
    with mock.patch.dict("os.environ", environment, clear=True):
        worker = MetakatWorker.from_config(Config())
    assert worker.allowed_image_extensions == {".png", ".tif"}
    assert worker.store_metakat_pdf and worker.store_mods_overview


def test_mailing_is_off_without_a_server(tmp_path):
    with mock.patch.dict("os.environ", {**KEY, "BASE_DIR": str(tmp_path)}, clear=True):
        worker = MetakatWorker.from_config(Config())
    assert worker.worker_mail_sender is None


def test_half_configured_mail_stops_the_worker():
    with mock.patch.dict("os.environ", {**KEY, "MAIL_SENDER_MAIL": "metakat@example.com"}, clear=True):
        with pytest.raises(ValueError, match="MAIL_SERVER is required"):
            Config()


def test_mailing_reads_its_settings_from_the_environment(tmp_path):
    environment = {
        **KEY,
        "BASE_DIR": str(tmp_path),
        "MAIL_SERVER": "smtp.example.com",
        "MAIL_PORT": "587",
        "MAIL_TIMEOUT_SECONDS": "12.5",
        "MAIL_SECURITY": "starttls",
        "MAIL_SENDER_MAIL": "metakat@example.com",
        "MAIL_USERNAME": "login",
        "MAIL_PASSWORD": "secret",
        "MAIL_RECEIVER_MAILS": "admin@example.com, user@example.com",
    }
    with mock.patch.dict("os.environ", environment, clear=True):
        config = Config()
        mail_sender = MetakatWorker.from_config(config).worker_mail_sender.sender

    assert config.MAIL_RECEIVER_MAILS == ["admin@example.com", "user@example.com"]
    assert config.MAIL_SENDER_NAME == "MetaKat worker"
    assert mail_sender.ca_file is None
    assert (mail_sender.send_receiver_limit, mail_sender.send_server_limit) == (10, 100)
    with mock.patch("smtplib.SMTP") as smtp_class:
        assert mail_sender.send("Job done", "body", "admin@example.com").status.value == "sent"
    smtp_class.assert_called_once_with("smtp.example.com", 587, timeout=12.5)
    smtp = smtp_class.return_value
    smtp.starttls.assert_called_once()
    smtp.login.assert_called_once_with("login", "secret")
    assert smtp.sendmail.call_args.args[1] == ["admin@example.com"]


def test_mail_limits_api_timeouts_and_outage_mail_read_the_environment(tmp_path):
    environment = {
        **KEY,
        "BASE_DIR": str(tmp_path),
        "MAIL_SERVER": "smtp.example.com",
        "MAIL_SENDER_MAIL": "metakat@example.com",
        "MAIL_RECEIVER_MAILS": "admin@example.com",
        "MAIL_SEND_RECEIVER_LIMIT": "0",
        "MAIL_SEND_RECEIVER_WINDOW_SECONDS": "60",
        "MAIL_SEND_SERVER_LIMIT": "5",
        "MAIL_SEND_SERVER_WINDOW_SECONDS": "120",
        "API_CONNECT_TIMEOUT_SECONDS": "3",
        "API_READ_TIMEOUT_SECONDS": "30",
        "OUTAGE_GRACE_SECONDS": "0",
        "OUTAGE_MAIL_INTERVAL_SECONDS": "600",
    }
    with mock.patch.dict("os.environ", environment, clear=True):
        config = Config()
        mail_sender = MetakatWorker.from_config(config).worker_mail_sender.sender
    assert (mail_sender.send_receiver_limit, mail_sender.send_receiver_window_seconds) == (0, 60)
    assert (mail_sender.send_server_limit, mail_sender.send_server_window_seconds) == (5, 120)
    assert (config.API_CONNECT_TIMEOUT_SECONDS, config.API_READ_TIMEOUT_SECONDS) == (3, 30)
    assert (config.OUTAGE_GRACE_SECONDS, config.OUTAGE_MAIL_INTERVAL_SECONDS) == (0, 600)


def test_api_timeouts_and_outage_mail_defaults():
    with mock.patch.dict("os.environ", KEY, clear=True):
        config = Config()
    assert (config.API_CONNECT_TIMEOUT_SECONDS, config.API_READ_TIMEOUT_SECONDS) == (10, 300)
    assert (config.OUTAGE_GRACE_SECONDS, config.OUTAGE_MAIL_INTERVAL_SECONDS) == (60, 3600)


def test_metakat_defaults_for_docapi_settings():
    with mock.patch.dict("os.environ", KEY, clear=True):
        config = Config()
    assert config.API_URL == "https://metakat.smart.lib.cas.cz"
    assert config.BASE_DIR == "./metakat_worker_data"
    assert config.MAIL_SENDER_NAME is None


@pytest.mark.parametrize("key,value", [
    ("API_CONNECT_TIMEOUT_SECONDS", "0"), ("API_READ_TIMEOUT_SECONDS", "inf"),
    ("OUTAGE_GRACE_SECONDS", "-1"), ("OUTAGE_MAIL_INTERVAL_SECONDS", "0"),
    ("MAIL_SEND_RECEIVER_LIMIT", "-1"), ("MAIL_SEND_SERVER_WINDOW_SECONDS", "0"),
    ("STORE_METAKAT_PDF", "maybe"),
])
def test_timeouts_limits_and_outage_settings_are_validated(key, value):
    with mock.patch.dict("os.environ", {**KEY, key: value}, clear=True):
        with pytest.raises(ValueError, match=key):
            Config()


def test_process_job_prepares_final_config_and_embedded_metadata(
    tmp_path,
    worker,
    workspace,
):
    images, altos, result, engines = workspace
    metadata_path = tmp_path / "metadata.json"
    metadata_path.write_text(
        json.dumps(
            {
                "metakat_json": {"batch_id": "00000000-0000-0000-0000-000000000001"},
                "proarc_json": {"elements": []},
                "engine_config_override": {
                    "chapter": {
                        "core": {
                            "alignment": {
                                "minimum_title_substring_similarity": 0.5,
                            }
                        }
                    }
                },
            }
        ),
        encoding="utf-8",
    )
    definition = {
        "chapter": {
            "core": {
                "name": "chapter_core_engine_pipeline",
                "page_analysis": {
                    "name": "chapter_page_analysis_engine_yolo_alto",
                    "model_path": "chapter/analysis.pt",
                },
                "extraction": {
                    "name": "chapter_extraction_engine_yolo_alto",
                    "model_path": "chapter/extraction.pt",
                },
                "alignment": {
                    "name": "chapter_alignment_engine_fuzzy",
                    "minimum_title_substring_similarity": 0.7,
                },
            },
            "bind": {"name": "chapter_bind_engine_base"},
        }
    }

    with mock.patch.object(worker_module, "process_batch") as process:
        response = worker.process_job(
            _job(definition),
            str(images),
            str(result),
            alto_dir=str(altos),
            meta_file=str(metadata_path),
            engine_dir=str(engines),
        )

    assert response.status == WorkerResponseStatus.OK
    arguments = process.call_args.kwargs
    assert arguments["engine_config"]["chapter"]["core"]["page_analysis"][
        "model_path"
    ] == str(engines / "chapter/analysis.pt")
    assert (
        arguments["engine_config"]["chapter"]["core"]["alignment"][
            "minimum_title_substring_similarity"
        ]
        == 0.5
    )
    assert (
        arguments["metakat_data"]["batch_id"]
        == "00000000-0000-0000-0000-000000000001"
    )
    assert arguments["proarc_data"] == {"elements": []}


def test_no_meta_file_passes_none_metadata(tmp_path, worker, workspace):
    images, altos, result, engines = workspace

    with mock.patch.object(worker_module, "process_batch") as process:
        response = worker.process_job(
            _job({}),
            str(images),
            str(result),
            alto_dir=str(altos),
            engine_dir=str(engines),
        )

    assert response.status == WorkerResponseStatus.OK
    assert process.call_args.kwargs["engine_config"] == {}
    assert process.call_args.kwargs["metakat_data"] is None
    assert process.call_args.kwargs["proarc_data"] is None
    # The job's engine is recorded in the MetaKat JSON the pipeline writes.
    assert process.call_args.kwargs["engine_name"] == "monograph"
    assert process.call_args.kwargs["engine_version"] == "v1.5.0"
    # MODS records are part of the result, next to the MetaKat JSON; the
    # overview is only for inspection and off by default.
    result_dir = Path(process.call_args.kwargs["output_metakat_json"]).parent
    assert Path(process.call_args.kwargs["output_mods_dir"]) == result_dir / "mods"
    assert process.call_args.kwargs["output_mods_overview"] is None


def test_path_escaping_engine_directory_fails_the_job(tmp_path, worker, workspace):
    images, altos, result, engines = workspace
    definition = {
        "page_number": {
            "core": {
                "name": "page_number_core_engine_yolo",
                "model_path": "../outside.pt",
            },
            "bind": {"name": "page_number_bind_engine_base"},
        }
    }

    with mock.patch.object(worker_module, "process_batch") as process:
        response = worker.process_job(
            _job(definition),
            str(images),
            str(result),
            alto_dir=str(altos),
            engine_dir=str(engines),
        )

    assert response.status != WorkerResponseStatus.OK
    assert isinstance(response.exception, ValueError)
    process.assert_not_called()


def test_metadata_envelope_rejects_unknown_keys(tmp_path, worker):
    # An unknown key is only an error once the document has identified itself
    # as an envelope by carrying at least one envelope key; a document with no
    # envelope key at all is a plain ProArc JSON, covered below.
    path = tmp_path / "metadata.json"
    path.write_text('{"proarc_json": {}, "legacy": {}}', encoding="utf-8")

    with pytest.raises(ValueError, match="Unknown"):
        worker._load_metadata_envelope(str(path))


_PLAIN_PROARC_JSON = {
    "type": "monograph",
    "objects": [
        {
            "pid": "uuid:00000000-0000-0000-0000-0000000000ff",
            "model": "volume",
            "metadata": "<mods/>",
        }
    ],
}


def test_plain_proarc_json_meta_file_is_read_as_proarc_json(tmp_path, worker):
    # Regression test: jobs that predate the envelope send packageInfo.json
    # itself as the meta file. Its "type"/"objects" keys used to be rejected
    # as unknown envelope keys, failing the job.
    path = tmp_path / "packageInfo.json"
    path.write_text(json.dumps(_PLAIN_PROARC_JSON), encoding="utf-8")

    metadata = worker._load_metadata_envelope(str(path))

    assert metadata["proarc_json"] == _PLAIN_PROARC_JSON
    assert metadata["metakat_json"] is None
    assert metadata["engine_config_override"] is None


def test_plain_proarc_json_reaches_the_pipeline_as_proarc_data(
    tmp_path,
    worker,
    workspace,
):
    images, altos, result, engines = workspace
    path = tmp_path / "packageInfo.json"
    path.write_text(json.dumps(_PLAIN_PROARC_JSON), encoding="utf-8")

    with mock.patch.object(worker_module, "process_batch") as process:
        response = worker.process_job(
            _job({}),
            str(images),
            str(result),
            alto_dir=str(altos),
            meta_file=str(path),
            engine_dir=str(engines),
        )

    assert response.status == WorkerResponseStatus.OK
    arguments = process.call_args.kwargs
    assert arguments["proarc_data"] == _PLAIN_PROARC_JSON
    assert arguments["metakat_data"] is None
    assert arguments["engine_config"] == {}


def test_empty_metadata_object_stays_an_empty_envelope(tmp_path, worker):
    # {} carries no envelope key, but it cannot be a ProArc JSON either - that
    # always has its own required keys - so it stays an empty envelope rather
    # than becoming an unusable proarc_json.
    path = tmp_path / "metadata.json"
    path.write_text("{}", encoding="utf-8")

    assert worker._load_metadata_envelope(str(path)) == {
        "metakat_json": None,
        "proarc_json": None,
        "engine_config_override": None,
    }


def test_partial_envelope_leaves_the_absent_keys_none(tmp_path, worker):
    path = tmp_path / "metadata.json"
    path.write_text(
        json.dumps({"proarc_json": _PLAIN_PROARC_JSON}),
        encoding="utf-8",
    )

    metadata = worker._load_metadata_envelope(str(path))

    assert metadata["proarc_json"] == _PLAIN_PROARC_JSON
    assert metadata["metakat_json"] is None
    assert metadata["engine_config_override"] is None


def test_explicitly_null_envelope_values_stay_none(tmp_path, worker):
    path = tmp_path / "metadata.json"
    path.write_text(
        json.dumps(
            {
                "metakat_json": None,
                "proarc_json": None,
                "engine_config_override": None,
            }
        ),
        encoding="utf-8",
    )

    assert worker._load_metadata_envelope(str(path)) == {
        "metakat_json": None,
        "proarc_json": None,
        "engine_config_override": None,
    }


def test_process_job_stores_pdf_beside_result_zip_when_enabled(
    tmp_path,
    worker,
    workspace,
):
    images, altos, result, engines = workspace

    with (
        mock.patch.object(worker_module, "process_batch") as process,
        mock.patch.object(worker, "store_metakat_pdf", True),
    ):
        response = worker.process_job(
            _job({}),
            str(images),
            str(result),
            alto_dir=str(altos),
            engine_dir=str(engines),
        )

    assert response.status == WorkerResponseStatus.OK
    assert process.call_args.kwargs["output_metakat_pdf"] == str(
        tmp_path / "result.pdf"
    )


def test_process_job_stores_mods_overview_beside_result_zip_when_enabled(
    tmp_path,
    worker,
    workspace,
):
    # Like the PDF, the overview is for inspection: kept beside result.zip,
    # never inside the uploaded result.
    images, altos, result, engines = workspace

    with (
        mock.patch.object(worker_module, "process_batch") as process,
        mock.patch.object(worker, "store_mods_overview", True),
    ):
        response = worker.process_job(
            _job({}),
            str(images),
            str(result),
            alto_dir=str(altos),
            engine_dir=str(engines),
        )

    assert response.status == WorkerResponseStatus.OK
    assert process.call_args.kwargs["output_mods_overview"] == str(
        tmp_path / "metakat.mods.txt"
    )


def _run_with_override(tmp_path, worker, workspace, definition, override):
    images, altos, result, engines = workspace
    metadata_path = tmp_path / "metadata.json"
    metadata_path.write_text(json.dumps({"engine_config_override": override}), encoding="utf-8")
    with mock.patch.object(worker_module, "process_batch") as process:
        response = worker.process_job(
            _job(definition),
            str(images),
            str(result),
            alto_dir=str(altos),
            engine_dir=str(engines),
            meta_file=str(metadata_path),
        )
    return response, process


def test_an_override_setting_up_a_local_model_fails_the_job(tmp_path, worker, workspace):
    definition = {"biblio": {"core": {"name": "biblio_core_engine_yolo_monograph"},
                             "bind": {"name": "biblio_bind_engine_monograph"}}}
    override = {"biblio": {"core": {"name": "biblio_core_engine_vlm_monograph", "local": {"model_dir": "model"}}}}

    response, process = _run_with_override(tmp_path, worker, workspace, definition, override)

    assert response.status != WorkerResponseStatus.OK
    assert "biblio.core.local" in str(response.exception)
    process.assert_not_called()


def test_an_override_may_replace_one_api_model_with_another(tmp_path, worker, workspace):
    def api_core(model):
        return {"name": "biblio_core_engine_vlm_monograph",
                "vlm": {"api_url": "https://openrouter.ai/api/v1", "model": model,
                        "api_key_env": "OPENROUTER_API_KEY"}}

    definition = {"biblio": {"core": api_core("qwen/qwen3.8-27b:free"),
                             "bind": {"name": "biblio_bind_engine_monograph"}}}
    override = {"biblio": {"core": {"vlm": {"model": "google/gemma-4-31b-it:free"}}}}

    response, process = _run_with_override(tmp_path, worker, workspace, definition, override)

    assert response.status == WorkerResponseStatus.OK
    assert process.call_args.kwargs["engine_config"]["biblio"]["core"] == api_core("google/gemma-4-31b-it:free")


@pytest.fixture
def pipeline_worker(tmp_path, monkeypatch):
    """Real wrapper pipeline, local files and a mocked DocAPI transport/inference."""
    worker = MetakatWorker("http://localhost", mock.Mock(), base_dir=str(tmp_path))
    worker.adapter = mock.Mock()
    worker.adapter.patch_job_finish.return_value = types.SimpleNamespace(is_success=True)
    worker.adapter.patch_job_error.return_value = types.SimpleNamespace(is_success=True)
    worker.adapter.patch_job_fail.return_value = types.SimpleNamespace(is_success=True)
    job = _job({})
    job.created_date = job.started_date = None
    job.previous_attempts = 0
    job.alto_required = True
    job.page_required = job.meta_json_required = False
    job.engine_id = job.engine_files_updated = None
    worker.adapter.post_job_lease.return_value = types.SimpleNamespace(
        is_success=True, data=types.SimpleNamespace(id=job.id, lease_expire_at=None))
    worker.adapter.get_job.return_value = types.SimpleNamespace(is_success=True, data=job)
    worker._check_and_download_engine_files = mock.Mock(return_value=worker_module.WorkerResponse.ok())
    worker._upload_results = mock.Mock(return_value=worker_module.WorkerResponse.ok())
    worker.get_engine_data_path = mock.Mock(return_value=str(tmp_path / "engine"))

    def download():
        directory = Path(worker.get_job_data_path())
        (directory / "images").mkdir()
        (directory / "alto").mkdir()
        (directory / "images" / "page.jpg").touch()
        (directory / "alto" / "page.xml").write_text("<alto/>")
        return worker_module.WorkerResponse.ok()

    worker._download_job_data = mock.Mock(side_effect=download)
    monkeypatch.setattr(worker_module, "process_batch", mock.Mock())
    worker.worker_mail_sender = mock.Mock()
    return worker


def test_finished_job_sends_final_summaries_to_docapi(pipeline_worker, caplog):
    caplog.set_level(logging.INFO)
    worker = pipeline_worker
    assert worker.run_job_pipeline().id == "job-1"
    reports = worker.adapter.patch_job_finish.call_args.kwargs
    assert "status    OK" in reports["final_log"]
    assert reports["final_log"] == reports["final_log_user"]
    assert "Job summary: job job-1 OK" in (Path(worker.jobs_dir) / "unknown.0.job-1" / "worker.log").read_text()
    worker.adapter.patch_job_progress_update.assert_not_called()
    worker.worker_mail_sender.send.assert_not_called()


def test_processing_exception_is_retryable_and_technical_only(pipeline_worker, monkeypatch):
    worker = pipeline_worker
    monkeypatch.setattr(worker_module, "process_batch", mock.Mock(side_effect=RuntimeError("private diagnostic")))
    assert worker.run_job_pipeline() is None
    reports = worker.adapter.patch_job_error.call_args.kwargs
    assert "status    ERROR" in reports["final_log"]
    assert "RuntimeError: private diagnostic" in reports["final_log"]
    assert "private diagnostic" not in reports["final_log_user"]
    assert "Please try again" in reports["final_log_user"]
    worker.adapter.patch_job_fail.assert_not_called()
    worker.worker_mail_sender.send.assert_not_called()


def test_invalid_input_is_a_permanent_failure(pipeline_worker, monkeypatch):
    worker = pipeline_worker
    monkeypatch.setattr(worker, "allowed_image_extensions", frozenset({".png"}))
    assert worker.run_job_pipeline() is None
    report = worker.adapter.patch_job_fail.call_args.kwargs
    assert "unsupported extension" in report["final_log"]
    assert "status    FAILED" in report["final_log_user"]
    worker.adapter.patch_job_error.assert_not_called()


@pytest.mark.parametrize("raises", [False, True])
def test_unreportable_job_failure_triggers_worker_failure_mail(pipeline_worker, raises):
    worker = pipeline_worker
    worker._download_job_data.side_effect = OSError("download unavailable")
    if raises:
        worker.adapter.patch_job_error.side_effect = ConnectionError("DocAPI unavailable")
    else:
        worker.adapter.patch_job_error.return_value = types.SimpleNamespace(is_success=False, status=500, code=None, response=None)
    assert worker.run_job_pipeline() is None
    worker.worker_mail_sender.send.assert_called_once()
    assert "Could not report" in worker.worker_mail_sender.send.call_args.args[1]


def test_failed_finish_request_alerts_without_replacing_success(pipeline_worker):
    worker = pipeline_worker
    worker.adapter.patch_job_finish.return_value = types.SimpleNamespace(is_success=False, status=500, code=None, response=None)
    assert worker.run_job_pipeline() is None
    worker.adapter.patch_job_finish.assert_called_once()
    worker.adapter.patch_job_error.assert_not_called()
    worker.adapter.patch_job_fail.assert_not_called()
    worker.worker_mail_sender.send.assert_called_once()


def test_acquisition_failure_cannot_touch_the_previous_job(pipeline_worker):
    worker = pipeline_worker
    worker.current_job = types.SimpleNamespace(id="previous-job")
    worker.adapter.post_job_lease.return_value = types.SimpleNamespace(is_success=False, data=None, status=500, code=None, response=None)
    assert worker.run_job_pipeline() is None
    assert worker.current_job is None
    worker.adapter.patch_job_error.assert_not_called()
    worker.adapter.patch_job_fail.assert_not_called()
    worker.worker_mail_sender.send.assert_called_once()


def test_failure_to_fetch_leased_job_reports_the_correct_lease(pipeline_worker):
    worker = pipeline_worker
    worker.adapter.get_job.return_value = types.SimpleNamespace(is_success=False, status=500, code=None, response=None)
    assert worker.run_job_pipeline() is None
    assert worker.adapter.patch_job_error.call_args.kwargs["job_id"] == "job-1"


def test_failure_to_summarize_does_not_prevent_reporting(pipeline_worker):
    worker = pipeline_worker
    worker._summarize = mock.Mock(side_effect=ValueError("summary failure"))
    worker_module.process_batch.side_effect = RuntimeError("private diagnostic")
    assert worker.run_job_pipeline() is None
    report = worker.adapter.patch_job_error.call_args.kwargs
    assert "private diagnostic" in report["final_log"]
    assert "private diagnostic" not in report["final_log_user"]


@pytest.mark.parametrize("failure", [KeyboardInterrupt, RuntimeError])
def test_worker_loop_reports_only_unexpected_termination(pipeline_worker, failure):
    worker = pipeline_worker
    worker.run_job_pipeline = mock.Mock(side_effect=failure)
    with mock.patch.object(worker, "alive_monitor") as alive_monitor:
        if failure is KeyboardInterrupt:
            worker.start()
            worker.worker_mail_sender.send.assert_not_called()
        else:
            with pytest.raises(RuntimeError):
                worker.start()
            worker.worker_mail_sender.send.assert_called_once()
        alive_monitor.start.assert_called_once()
        alive_monitor.close.assert_called_once()


def test_worker_formatter_does_not_mask():
    # Names such as page_key only look secret; real secrets are redacted where they are logged.
    from metakat.worker.docapi.config import MetakatWorkerFormatter
    message = "unknown page key: uuid:1234, page_key: p1, contains_keyword=True"
    record = logging.LogRecord("worker", logging.ERROR, __file__, 1, message, (), None)
    assert message in MetakatWorkerFormatter().format(record)


def test_alive_mail_defaults_and_source_preferences(monkeypatch):
    for key in ("ALIVE_MAIL_STARTUP", "ALIVE_MAIL_INTERVAL_SECONDS",
                "PREFER_IMAGE_URL", "PREFER_ALTO_URL", "PREFER_PAGE_URL"):
        monkeypatch.delenv(key, raising=False)
    config = Config()
    assert config.ALIVE_MAIL_STARTUP is True
    assert config.ALIVE_MAIL_INTERVAL_SECONDS == 86400
    assert not config.PREFER_IMAGE_URL and not config.PREFER_ALTO_URL and not config.PREFER_PAGE_URL
    monkeypatch.setenv("ALIVE_MAIL_STARTUP", "false")
    monkeypatch.setenv("ALIVE_MAIL_INTERVAL_SECONDS", "0")
    monkeypatch.setenv("PREFER_IMAGE_URL", "true")
    assert Config().ALIVE_MAIL_STARTUP is False
    assert Config().ALIVE_MAIL_INTERVAL_SECONDS == 0
    assert Config().PREFER_IMAGE_URL is True


@pytest.mark.parametrize("value", ["-1", "nan", "inf"])
@pytest.mark.parametrize("key", ["ALIVE_MAIL_INTERVAL_SECONDS"])
def test_alive_mail_intervals_must_be_finite_nonnegative(monkeypatch, key, value):
    monkeypatch.setenv(key, value)
    with pytest.raises(ValueError, match=key):
        Config()


def test_download_failure_uses_wrapper_messages_without_metakat_summary(pipeline_worker):
    worker = pipeline_worker
    worker._download_job_data.side_effect = OSError("private connection detail")
    worker._summarize = mock.Mock()
    assert worker.run_job_pipeline() is None
    report = worker.adapter.patch_job_error.call_args.kwargs
    assert report["final_log"].startswith("Failed to download job data from server")
    assert report["final_log_user"] == "Failed to download job data from server"
    worker._summarize.assert_not_called()


def test_history_is_off_unless_set(monkeypatch):
    monkeypatch.delenv('JOB_HISTORY_PATH', raising=False)
    assert Config().JOB_HISTORY_PATH is None


def test_logging_defaults_follow_the_base_directory():
    with mock.patch.dict("os.environ", {}, clear=True):
        logging_config = LoggingConfig("/data/worker")
    assert logging_config.LOGGING_DIR == "/data/worker/logs"
    assert (logging_config.LOGGING_CONSOLE_LEVEL, logging_config.LOGGING_FILE_LEVEL) == ("INFO", "INFO")
    assert logging_config.LOGGING_CONFIG["handlers"]["file_log"]["filename"] == "/data/worker/logs/worker.log"


def test_logging_overrides_win_over_the_environment():
    environment = {"LOGGING_DIR": "/var/log/worker", "LOGGING_FILE_LEVEL": "DEBUG"}
    with mock.patch.dict("os.environ", environment, clear=True):
        logging_config = LoggingConfig("/data/worker", {"LOGGING_DIR": "/tmp/development-worker-logs"})
    assert logging_config.LOGGING_DIR == "/tmp/development-worker-logs"
    assert logging_config.LOGGING_FILE_LEVEL == "DEBUG"


def test_logging_levels_are_validated():
    with mock.patch.dict("os.environ", {"LOGGING_FILE_LEVEL": "LOUD"}, clear=True):
        with pytest.raises(ValueError, match="LOGGING_FILE_LEVEL"):
            LoggingConfig("/data/worker")


def test_logging_settings_are_not_worker_settings():
    # Config holds only what the worker is built from, so from_config passes every
    # setting it adds to MetakatWorker without a hand-written subclass_args().
    assert not any(name.startswith("LOGGING") for name in Config.setting_names())
    assert "subclass_args" not in vars(MetakatWorker)


def test_history_path_can_be_overridden_or_disabled(monkeypatch):
    monkeypatch.setenv('JOB_HISTORY_PATH', '/tmp/worker-history.jsonl')
    assert Config().JOB_HISTORY_PATH == '/tmp/worker-history.jsonl'
    monkeypatch.setenv('JOB_HISTORY_PATH', '')
    assert Config().JOB_HISTORY_PATH is None


@pytest.mark.parametrize("value", ["0", "-1", "nan", "inf", "-inf"])
def test_mail_timeout_must_be_finite_positive(monkeypatch, value):
    monkeypatch.setenv("MAIL_TIMEOUT_SECONDS", value)
    with pytest.raises(ValueError, match="MAIL_TIMEOUT_SECONDS"):
        Config()


def test_worker_mail_identity_defaults_to_wrapper_defaults(monkeypatch, tmp_path):
    for key in ("WORKER_IDENTITY", "WORKER_HOSTNAME", "MAIL_SUBJECT"):
        monkeypatch.delenv(key, raising=False)
    monkeypatch.setenv("BASE_DIR", str(tmp_path))
    config = Config()
    assert (config.WORKER_IDENTITY, config.WORKER_HOSTNAME) == (None, None)
    assert config.MAIL_SUBJECT == "[{identity}] {status} ({hostname})"
    # Without an identity, the worker takes its key's label from DocAPI when it starts.
    assert MetakatWorker.from_config(config).identity == "MetakatWorker"


def test_worker_mail_identity_names_the_deployment_in_mail(monkeypatch, tmp_path):
    for key, value in {"BASE_DIR": str(tmp_path), "WORKER_IDENTITY": "MetaKat worker prod",
                       "WORKER_HOSTNAME": "gpu-host", "MAIL_SUBJECT": "{identity}: {status} on {hostname}",
                       "MAIL_SERVER": "smtp.example.com", "MAIL_SENDER_MAIL": "metakat@example.com",
                       "MAIL_RECEIVER_MAILS": "admin@example.com"}.items():
        monkeypatch.setenv(key, value)
    worker = MetakatWorker.from_config(Config())
    worker.worker_mail_sender.sender = mock.Mock()
    worker.worker_mail_sender.send("ALIVE", "report")
    assert worker.worker_mail_sender.sender.send.call_args.args[0] == "MetaKat worker prod: ALIVE on gpu-host"


def test_mail_timeout_defaults_to_sixty_seconds(monkeypatch):
    monkeypatch.delenv("MAIL_TIMEOUT_SECONDS", raising=False)
    assert Config().MAIL_TIMEOUT_SECONDS == 60
