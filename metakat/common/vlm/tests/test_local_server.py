import json
import socket
import urllib.request

import pytest

from metakat.common.vlm.local_server import (
    LocalModelConfig,
    VLLMSettings,
    local_model_locations,
    serve,
)

def _port_open(port: int) -> bool:
    with socket.socket() as probe:
        return probe.connect_ex(("127.0.0.1", port)) == 0


def test_the_server_runs_only_inside_the_block(fake_vllm):
    model = LocalModelConfig.from_config({"model_dir": "/models/qwen-biblio", "vllm_args": ["--max-model-len", "8192"]})

    with serve(model) as api_url:
        port = int(api_url.rsplit(":", 1)[1].split("/")[0])
        assert api_url == f"http://127.0.0.1:{port}/v1"
        with urllib.request.urlopen(f"http://127.0.0.1:{port}/health") as response:
            assert response.status == 200
    assert not _port_open(port)

    args = json.loads(fake_vllm.read_text())
    assert args[:2] == ["serve", "/models/qwen-biblio"]
    assert args[args.index("--served-model-name") + 1] == "qwen-biblio"
    assert args[args.index("--host") + 1] == "127.0.0.1"
    assert args[args.index("--gpu-memory-utilization") + 1] == "0.85"
    assert args[-2:] == ["--max-model-len", "8192"]


def test_the_server_stops_when_the_block_fails(fake_vllm):
    with pytest.raises(KeyError):
        with serve(LocalModelConfig.from_config({"model_dir": "/models/m"})) as api_url:
            port = int(api_url.rsplit(":", 1)[1].split("/")[0])
            raise KeyError("boom")
    assert not _port_open(port)


def test_a_server_that_dies_while_starting_reports_its_log(fake_vllm, monkeypatch):
    monkeypatch.setenv("FAKE_VLLM_FAIL", "1")

    with pytest.raises(RuntimeError, match="(?s)exited with code 3.*CUDA out of memory"):
        with serve(LocalModelConfig.from_config({"model_dir": "/models/m"})):
            pass


def test_a_missing_executable_is_reported(monkeypatch):
    monkeypatch.setenv("VLLM_EXECUTABLE", "/nonexistent/vllm")

    with pytest.raises(RuntimeError, match="Cannot start vLLM"):
        with serve(LocalModelConfig.from_config({"model_dir": "/models/m"})):
            pass


def test_machine_settings_come_from_the_environment():
    settings = VLLMSettings.from_environment({
        "VLLM_EXECUTABLE": "/opt/vllm/bin/vllm",
        "VLLM_GPU_MEMORY_UTILIZATION": "0.6",
        "VLLM_STARTUP_TIMEOUT": "120",
        "VLLM_CUDA_VISIBLE_DEVICES": "1",
    })
    assert settings == VLLMSettings("/opt/vllm/bin/vllm", 0.6, 120.0, "1")
    assert VLLMSettings.from_environment({}) == VLLMSettings()
    with pytest.raises(ValueError, match="GPU_MEMORY_UTILIZATION"):
        VLLMSettings.from_environment({"VLLM_GPU_MEMORY_UTILIZATION": "1.5"})


@pytest.mark.parametrize("config, message", [
    ({}, "needs model_dir"),
    ({"model_dir": "m", "vllm_args": ["--port", "1"]}, "must not set --port"),
    ({"model_dir": "m", "vllm_args": ["--gpu-memory-utilization=0.9"]}, "must not set --gpu-memory-utilization"),
    ({"model_dir": "m", "vllm_args": "--max-model-len 8192"}, "list of strings"),
    ({"model_dir": "m", "command": ["rm"]}, "unknown keys: command"),
])
def test_invalid_local_model_configuration_is_rejected(config, message):
    with pytest.raises(ValueError, match=message):
        LocalModelConfig.from_config(config)


def test_local_models_are_found_anywhere_in_a_configuration():
    override = {
        "biblio": {"core": {"name": "biblio_core_engine_vlm", "local": {"model_dir": "m"}}},
        "chapter": {"core": {"stages": [{"local": {}}]}},
        "page_type": {"core": {"vlm": {"model": "x"}}},
    }
    assert local_model_locations(override) == ["biblio.core.local", "chapter.core.stages[0].local"]
    assert local_model_locations(None) == []
