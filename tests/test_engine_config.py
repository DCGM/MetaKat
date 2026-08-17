import json
import tempfile
import unittest
from pathlib import Path

from metakat.engine_config import (
    apply_assignments,
    load_config_file,
    merge_configs,
    prepare_engine_config,
    resolve_config_paths,
)
from metakat.logging_utils import REDACTED_LOG_VALUE, redacted_for_logging


class EngineConfigTest(unittest.TestCase):
    def test_logging_wrapper_recursively_redacts_potential_secrets(self):
        config = {
            "apiKey": "api-secret",
            "chapter": {
                "core": {
                    "model_path": "model.pt",
                    "credentials": {
                        "username": "operator",
                        "password": "password-secret",
                    },
                    "headers": [
                        {"Authorization": "Bearer header-secret"},
                    ],
                    "minimum_tokens": 10,
                    "tokenizer_path": "tokenizer.json",
                }
            },
        }

        logged = json.loads(str(redacted_for_logging(config)))

        self.assertEqual(logged["apiKey"], REDACTED_LOG_VALUE)
        credentials = logged["chapter"]["core"]["credentials"]
        self.assertEqual(credentials, REDACTED_LOG_VALUE)
        self.assertEqual(
            logged["chapter"]["core"]["headers"][0]["Authorization"],
            REDACTED_LOG_VALUE,
        )
        self.assertEqual(
            logged["chapter"]["core"]["model_path"],
            "model.pt",
        )
        self.assertEqual(logged["chapter"]["core"]["minimum_tokens"], 10)
        self.assertEqual(
            logged["chapter"]["core"]["tokenizer_path"],
            "tokenizer.json",
        )
        self.assertEqual(config["apiKey"], "api-secret")

    def test_process_batch_logs_redacted_final_pipeline_config_first(self):
        from metakat.process_batch import process_batch

        with tempfile.TemporaryDirectory() as temporary_directory:
            with self.assertLogs("metakat.process_batch", level="INFO") as logs:
                process_batch(
                    batch_dir=temporary_directory,
                    engine_config={
                        "api_key": "must-not-appear",
                        "diagnostic": {"model_path": "/engines/model.pt"},
                    },
                )

        output = "\n".join(logs.output)
        self.assertIn(
            "Starting MetaKat processing with engine pipeline configuration",
            output,
        )
        self.assertIn(REDACTED_LOG_VALUE, output)
        self.assertNotIn("must-not-appear", output)
        self.assertIn("/engines/model.pt", output)
        self.assertLess(
            output.index("Starting MetaKat processing"),
            output.index("MetakatIO has been successfully validated"),
        )

    def test_loads_json_and_yaml_to_the_same_mapping(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            expected = {"chapter": {"core": {"threshold": 0.7}}}
            json_path = root / "pipeline.json"
            yaml_path = root / "pipeline.yaml"
            json_path.write_text(json.dumps(expected), encoding="utf-8")
            yaml_path.write_text(
                "chapter:\n  core:\n    threshold: 0.7\n",
                encoding="utf-8",
            )

            self.assertEqual(load_config_file(json_path), expected)
            self.assertEqual(load_config_file(yaml_path), expected)

    def test_rejects_non_object_and_non_json_yaml_values(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            for name, content, error in (
                ("list.yaml", "- one\n", "must be an object"),
                ("date.yaml", "created: 2026-08-14\n", "JSON-compatible"),
                ("config.txt", "{}", "must use"),
            ):
                path = root / name
                path.write_text(content, encoding="utf-8")
                with self.subTest(name=name):
                    with self.assertRaisesRegex(ValueError, error):
                        load_config_file(path)

    def test_deep_merge_replaces_only_leaves_and_lists(self):
        base = {
            "chapter": {
                "core": {"threshold": 0.7, "labels": ["a"]},
                "bind": {"name": "base"},
            }
        }
        override = {
            "chapter": {
                "core": {"threshold": 0.5, "labels": ["b"], "new": True}
            }
        }

        merged = merge_configs(base, override)

        self.assertEqual(
            merged,
            {
                "chapter": {
                    "core": {
                        "threshold": 0.5,
                        "labels": ["b"],
                        "new": True,
                    },
                    "bind": {"name": "base"},
                }
            },
        )
        self.assertEqual(base["chapter"]["core"]["threshold"], 0.7)

    def test_deep_merge_rejects_mapping_leaf_conflicts(self):
        for base, override in (({"a": {}}, {"a": 1}), ({"a": 1}, {"a": {}})):
            with self.subTest(base=base, override=override):
                with self.assertRaisesRegex(ValueError, "mapping and non-mapping"):
                    merge_configs(base, override)

    def test_assignments_are_ordered_and_parse_json_values(self):
        result = apply_assignments(
            {"chapter": {"core": {"value": 1}}},
            [
                ("chapter:core:value", "0.5"),
                ("chapter:core:value", "2"),
                ("chapter:core:enabled", "true"),
                ("chapter:core:labels", '["a", "b"]'),
                ("chapter:core:model_path", "models/model.pt"),
            ],
        )

        self.assertEqual(result["chapter"]["core"]["value"], 2)
        self.assertIs(result["chapter"]["core"]["enabled"], True)
        self.assertEqual(result["chapter"]["core"]["labels"], ["a", "b"])
        self.assertEqual(
            result["chapter"]["core"]["model_path"],
            "models/model.pt",
        )

    def test_path_resolution_handles_scalar_and_list_fields(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            result = resolve_config_paths(
                {
                    "model_path": "model.pt",
                    "cache_dir": str(root / "absolute"),
                    "source_paths": ["one.json", "two.json"],
                    "unrelated": "relative.txt",
                },
                root,
            )

        self.assertEqual(result["model_path"], str(root / "model.pt"))
        self.assertEqual(result["cache_dir"], str(root / "absolute"))
        self.assertEqual(
            result["source_paths"],
            [str(root / "one.json"), str(root / "two.json")],
        )
        self.assertEqual(result["unrelated"], "relative.txt")

    def test_worker_containment_rejects_relative_absolute_and_symlink_escape(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            engine_dir = root / "engines"
            engine_dir.mkdir()
            outside = root / "outside"
            outside.mkdir()
            (engine_dir / "link").symlink_to(outside, target_is_directory=True)
            for value in ("../outside/model.pt", str(outside / "model.pt"), "link/model.pt"):
                with self.subTest(value=value):
                    with self.assertRaisesRegex(ValueError, "escapes"):
                        resolve_config_paths(
                            {"model_path": value},
                            engine_dir,
                            require_within_base=True,
                        )

    def test_preparation_resolves_only_after_merge_and_assignments(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            result = prepare_engine_config(
                {"core": {"model_path": "base.pt", "threshold": 0.9}},
                override={"core": {"model_path": "override.pt"}},
                assignments=(("core:threshold", "0.4"),),
                base_dir=root,
            )

        self.assertEqual(result["core"]["model_path"], str(root / "override.pt"))
        self.assertEqual(result["core"]["threshold"], 0.4)


if __name__ == "__main__":
    unittest.main()
