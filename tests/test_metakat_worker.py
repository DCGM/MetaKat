import json
import logging
import tempfile
import types
import unittest
from pathlib import Path
from unittest import mock

from metakat.worker.docapi import metakat_worker as worker_module
from metakat.worker.docapi.config import Config
from metakat.worker.docapi.metakat_worker import MetakatWorker


def _job(engine_definition):
    return types.SimpleNamespace(
        images=[types.SimpleNamespace(name="page.jpg", order=0)],
        engine_definition=engine_definition,
    )


def _workspace(root: Path):
    images = root / "images"
    altos = root / "alto"
    result = root / "result"
    engines = root / "engines"
    for directory in (images, altos, result, engines):
        directory.mkdir()
    (images / "page.jpg").touch()
    (altos / "page.xml").write_text("<alto/>", encoding="utf-8")
    return images, altos, result, engines


class MetakatWorkerEngineTest(unittest.TestCase):
    def setUp(self):
        self.worker = object.__new__(MetakatWorker)

    def test_store_metakat_pdf_defaults_false_and_reads_environment(self):
        with mock.patch.dict("os.environ", {}, clear=True):
            self.assertFalse(Config().STORE_METAKAT_PDF)
        with mock.patch.dict(
            "os.environ",
            {"STORE_METAKAT_PDF": "true"},
            clear=True,
        ):
            self.assertTrue(Config().STORE_METAKAT_PDF)

    def test_process_job_prepares_final_config_and_embedded_metadata(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            images, altos, result, engines = _workspace(root)
            metadata_path = root / "metadata.json"
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
            handler = logging.FileHandler(root / "job.log")

            with mock.patch.object(worker_module, "process_batch") as process:
                response = self.worker.process_job(
                    _job(definition),
                    handler,
                    str(images),
                    str(result),
                    alto_dir=str(altos),
                    meta_file=str(metadata_path),
                    engine_dir=str(engines),
                )

        self.assertTrue(response.success)
        arguments = process.call_args.kwargs
        self.assertEqual(
            arguments["engine_config"]["chapter"]["core"]["page_analysis"]["model_path"],
            str(engines / "chapter/analysis.pt"),
        )
        self.assertEqual(
            arguments["engine_config"]["chapter"]["core"]["alignment"]
            ["minimum_title_substring_similarity"],
            0.5,
        )
        self.assertEqual(arguments["metakat_data"]["batch_id"], "00000000-0000-0000-0000-000000000001")
        self.assertEqual(arguments["proarc_data"], {"elements": []})

    def test_no_meta_file_passes_none_metadata(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            images, altos, result, engines = _workspace(root)
            handler = logging.FileHandler(root / "job.log")
            with mock.patch.object(worker_module, "process_batch") as process:
                response = self.worker.process_job(
                    _job({}),
                    handler,
                    str(images),
                    str(result),
                    alto_dir=str(altos),
                    engine_dir=str(engines),
                )

        self.assertTrue(response.success)
        self.assertEqual(process.call_args.kwargs["engine_config"], {})
        self.assertIsNone(process.call_args.kwargs["metakat_data"])
        self.assertIsNone(process.call_args.kwargs["proarc_data"])

    def test_path_escaping_engine_directory_fails_the_job(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            images, altos, result, engines = _workspace(root)
            handler = logging.FileHandler(root / "job.log")
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
                response = self.worker.process_job(
                    _job(definition),
                    handler,
                    str(images),
                    str(result),
                    alto_dir=str(altos),
                    engine_dir=str(engines),
                )

        self.assertFalse(response.success)
        self.assertIsInstance(response.exception, ValueError)
        process.assert_not_called()

    def test_metadata_envelope_rejects_unknown_keys(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            path = Path(temporary_directory) / "metadata.json"
            path.write_text('{"legacy": {}}', encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "Unknown"):
                self.worker._load_metadata_envelope(str(path))

    def test_process_job_stores_pdf_beside_result_zip_when_enabled(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            images, altos, result, engines = _workspace(root)
            handler = logging.FileHandler(root / "job.log")
            with (
                mock.patch.object(worker_module, "process_batch") as process,
                mock.patch.object(worker_module.config, "STORE_METAKAT_PDF", True),
            ):
                response = self.worker.process_job(
                    _job({}),
                    handler,
                    str(images),
                    str(result),
                    alto_dir=str(altos),
                    engine_dir=str(engines),
                )

        self.assertTrue(response.success)
        self.assertEqual(
            process.call_args.kwargs["output_metakat_pdf"],
            str(root / "result.pdf"),
        )


if __name__ == "__main__":
    unittest.main()
