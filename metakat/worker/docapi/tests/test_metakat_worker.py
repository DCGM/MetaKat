import json
import logging
import types
from unittest import mock

import pytest

from metakat.worker.docapi import metakat_worker as worker_module
from metakat.worker.docapi.config import Config
from metakat.worker.docapi.metakat_worker import MetakatWorker


def _job(engine_definition):
    return types.SimpleNamespace(
        images=[types.SimpleNamespace(name="page.jpg", order=0)],
        engine_definition=engine_definition,
    )


@pytest.fixture
def worker():
    return object.__new__(MetakatWorker)


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
    with mock.patch.dict("os.environ", {}, clear=True):
        assert not Config().STORE_METAKAT_PDF
    with mock.patch.dict(
        "os.environ",
        {"STORE_METAKAT_PDF": "true"},
        clear=True,
    ):
        assert Config().STORE_METAKAT_PDF


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
    handler = logging.FileHandler(tmp_path / "job.log")

    with mock.patch.object(worker_module, "process_batch") as process:
        response = worker.process_job(
            _job(definition),
            handler,
            str(images),
            str(result),
            alto_dir=str(altos),
            meta_file=str(metadata_path),
            engine_dir=str(engines),
        )

    assert response.success
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
    handler = logging.FileHandler(tmp_path / "job.log")

    with mock.patch.object(worker_module, "process_batch") as process:
        response = worker.process_job(
            _job({}),
            handler,
            str(images),
            str(result),
            alto_dir=str(altos),
            engine_dir=str(engines),
        )

    assert response.success
    assert process.call_args.kwargs["engine_config"] == {}
    assert process.call_args.kwargs["metakat_data"] is None
    assert process.call_args.kwargs["proarc_data"] is None


def test_path_escaping_engine_directory_fails_the_job(tmp_path, worker, workspace):
    images, altos, result, engines = workspace
    handler = logging.FileHandler(tmp_path / "job.log")
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
            handler,
            str(images),
            str(result),
            alto_dir=str(altos),
            engine_dir=str(engines),
        )

    assert not response.success
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
    handler = logging.FileHandler(tmp_path / "job.log")

    with mock.patch.object(worker_module, "process_batch") as process:
        response = worker.process_job(
            _job({}),
            handler,
            str(images),
            str(result),
            alto_dir=str(altos),
            meta_file=str(path),
            engine_dir=str(engines),
        )

    assert response.success
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
    handler = logging.FileHandler(tmp_path / "job.log")

    with (
        mock.patch.object(worker_module, "process_batch") as process,
        mock.patch.object(worker_module.config, "STORE_METAKAT_PDF", True),
    ):
        response = worker.process_job(
            _job({}),
            handler,
            str(images),
            str(result),
            alto_dir=str(altos),
            engine_dir=str(engines),
        )

    assert response.success
    assert process.call_args.kwargs["output_metakat_pdf"] == str(
        tmp_path / "result.pdf"
    )
