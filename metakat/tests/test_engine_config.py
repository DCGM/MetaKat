import json
import logging
from pathlib import Path

import pytest

from metakat.engine_config import (
    apply_assignments,
    load_config_file,
    merge_configs,
    prepare_engine_config,
    resolve_config_paths,
)
from metakat.logging_utils import REDACTED_LOG_VALUE, redacted_for_logging


def test_logging_wrapper_recursively_redacts_potential_secrets():
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

    assert logged["apiKey"] == REDACTED_LOG_VALUE
    credentials = logged["chapter"]["core"]["credentials"]
    assert credentials == REDACTED_LOG_VALUE
    assert logged["chapter"]["core"]["headers"][0]["Authorization"] == REDACTED_LOG_VALUE
    assert logged["chapter"]["core"]["model_path"] == "model.pt"
    assert logged["chapter"]["core"]["minimum_tokens"] == 10
    assert logged["chapter"]["core"]["tokenizer_path"] == "tokenizer.json"
    assert config["apiKey"] == "api-secret"


def test_process_batch_logs_redacted_final_pipeline_config_first(tmp_path, caplog):
    from metakat.process_batch import process_batch

    with caplog.at_level(logging.INFO, logger="metakat.process_batch"):
        process_batch(
            batch_dir=str(tmp_path),
            engine_config={
                "api_key": "must-not-appear",
                "diagnostic": {"model_path": "/engines/model.pt"},
            },
        )

    records = [
        record
        for record in caplog.records
        if record.name.startswith("metakat.process_batch")
    ]
    assert records
    output = "\n".join(record.getMessage() for record in records)
    assert "Starting MetaKat processing with engine pipeline configuration" in output
    assert REDACTED_LOG_VALUE in output
    assert "must-not-appear" not in output
    assert "/engines/model.pt" in output
    assert output.index("Starting MetaKat processing") < output.index(
        "MetakatIO has been successfully validated"
    )


def test_loads_json_and_yaml_to_the_same_mapping(tmp_path):
    expected = {"chapter": {"core": {"threshold": 0.7}}}
    json_path = tmp_path / "pipeline.json"
    yaml_path = tmp_path / "pipeline.yaml"
    json_path.write_text(json.dumps(expected), encoding="utf-8")
    yaml_path.write_text(
        "chapter:\n  core:\n    threshold: 0.7\n",
        encoding="utf-8",
    )

    assert load_config_file(json_path) == expected
    assert load_config_file(yaml_path) == expected


@pytest.mark.parametrize(
    "name,content,error",
    (
        ("list.yaml", "- one\n", "must be an object"),
        ("date.yaml", "created: 2026-08-14\n", "JSON-compatible"),
        ("config.txt", "{}", "must use"),
    ),
)
def test_rejects_non_object_and_non_json_yaml_values(tmp_path, name, content, error):
    path = tmp_path / name
    path.write_text(content, encoding="utf-8")

    with pytest.raises(ValueError, match=error):
        load_config_file(path)


def test_deep_merge_replaces_only_leaves_and_lists():
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

    assert merged == {
        "chapter": {
            "core": {
                "threshold": 0.5,
                "labels": ["b"],
                "new": True,
            },
            "bind": {"name": "base"},
        }
    }
    assert base["chapter"]["core"]["threshold"] == 0.7


@pytest.mark.parametrize(
    "base,override",
    (({"a": {}}, {"a": 1}), ({"a": 1}, {"a": {}})),
)
def test_deep_merge_rejects_mapping_leaf_conflicts(base, override):
    with pytest.raises(ValueError, match="mapping and non-mapping"):
        merge_configs(base, override)


def test_assignments_are_ordered_and_parse_json_values():
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

    assert result["chapter"]["core"]["value"] == 2
    assert result["chapter"]["core"]["enabled"] is True
    assert result["chapter"]["core"]["labels"] == ["a", "b"]
    assert result["chapter"]["core"]["model_path"] == "models/model.pt"


def test_path_resolution_handles_scalar_and_list_fields(tmp_path):
    result = resolve_config_paths(
        {
            "model_path": "model.pt",
            "cache_dir": str(tmp_path / "absolute"),
            "source_paths": ["one.json", "two.json"],
            "unrelated": "relative.txt",
        },
        tmp_path,
    )

    assert result["model_path"] == str(tmp_path / "model.pt")
    assert result["cache_dir"] == str(tmp_path / "absolute")
    assert result["source_paths"] == [
        str(tmp_path / "one.json"),
        str(tmp_path / "two.json"),
    ]
    assert result["unrelated"] == "relative.txt"


# The absolute case needs the temporary directory, so the escape route is named
# here and the offending value is built inside the test.
@pytest.mark.parametrize("escape", ("relative", "absolute", "symlink"))
def test_worker_containment_rejects_relative_absolute_and_symlink_escape(
    tmp_path,
    escape,
):
    engine_dir = tmp_path / "engines"
    engine_dir.mkdir()
    outside = tmp_path / "outside"
    outside.mkdir()
    (engine_dir / "link").symlink_to(outside, target_is_directory=True)
    value = {
        "relative": "../outside/model.pt",
        "absolute": str(outside / "model.pt"),
        "symlink": "link/model.pt",
    }[escape]

    with pytest.raises(ValueError, match="escapes"):
        resolve_config_paths(
            {"model_path": value},
            engine_dir,
            require_within_base=True,
        )


def test_preparation_resolves_only_after_merge_and_assignments(tmp_path):
    result = prepare_engine_config(
        {"core": {"model_path": "base.pt", "threshold": 0.9}},
        override={"core": {"model_path": "override.pt"}},
        assignments=(("core:threshold", "0.4"),),
        base_dir=tmp_path,
    )

    assert result["core"]["model_path"] == str(tmp_path / "override.pt")
    assert result["core"]["threshold"] == 0.4
