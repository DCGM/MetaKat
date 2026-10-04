"""A vision-language model served locally by vLLM, only while it is needed.

A local model is part of an engine, as a YOLO model is: its files come with
the engine, and `model_dir` with the `vllm_args` that belong to the model
(context length, image limit, chat template, ...) are in the engine
configuration. How vLLM runs on this machine is the machine's business and is
read from the environment, so the worker and a standalone process_batch run
it the same way:

| Variable | Default | Meaning |
|---|---|---|
| `VLLM_EXECUTABLE` | `vllm` | The `vllm` command, typically from vLLM's own environment, which pins its own torch. |
| `VLLM_GPU_MEMORY_UTILIZATION` | `0.85` | The fraction of the GPU's memory vLLM reserves. |
| `VLLM_STARTUP_TIMEOUT` | `900` | Seconds to wait until the server answers. |
| `VLLM_CUDA_VISIBLE_DEVICES` | - | The GPU(s) to run on; unset leaves CUDA_VISIBLE_DEVICES as it is. |

`serve()` starts `vllm serve` as a process group of its own, on a free port
of 127.0.0.1, waits until it answers /health and yields the OpenAI-compatible
base URL. On leaving - normally or by an error - the whole group is
terminated, so all of its GPU memory returns to the system.
"""
from __future__ import annotations

import contextlib
import logging
import os
import signal
import socket
import subprocess
import tempfile
import time
import urllib.error
import urllib.request
from collections.abc import Iterator, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

# Set by the server itself from the machine's settings and the free port.
_RESERVED_ARGS = ("--host", "--port", "--gpu-memory-utilization", "--served-model-name", "--api-key")
_LOCAL_KEYS = {"model_dir", "served_model_name", "vllm_args"}
_STOP_TIMEOUT_SECONDS = 60.0
_POLL_SECONDS = 1.0
_LOG_TAIL_LINES = 40


@dataclass(frozen=True)
class LocalModelConfig:
    """The engine's part: which model, and the vLLM arguments it needs."""

    model_dir: str
    served_model_name: str
    vllm_args: tuple[str, ...] = ()

    @classmethod
    def from_config(cls, config: Mapping[str, Any], location: str = "Local VLM config") -> "LocalModelConfig":
        if not isinstance(config, Mapping):
            raise ValueError(f"{location} must be an object")
        unknown = sorted(set(config) - _LOCAL_KEYS)
        if unknown:
            raise ValueError(f"{location} has unknown keys: {', '.join(unknown)}")
        model_dir = config.get("model_dir")
        if not isinstance(model_dir, str) or not model_dir:
            raise ValueError(f"{location} needs model_dir")
        args = config.get("vllm_args") or ()
        if isinstance(args, (str, bytes)) or not all(isinstance(arg, str) for arg in args):
            raise ValueError(f"{location} vllm_args must be a list of strings")
        reserved = sorted({arg.split("=", 1)[0] for arg in args} & set(_RESERVED_ARGS))
        if reserved:
            raise ValueError(f"{location} vllm_args must not set {', '.join(reserved)}; the server sets it")
        return cls(
            model_dir=model_dir,
            served_model_name=config.get("served_model_name") or Path(model_dir).name,
            vllm_args=tuple(args),
        )


@dataclass(frozen=True)
class VLLMSettings:
    """The machine's part, from the environment."""

    executable: str = "vllm"
    gpu_memory_utilization: float = 0.85
    startup_timeout: float = 900.0
    cuda_visible_devices: str | None = None

    @classmethod
    def from_environment(cls, environ: Mapping[str, str] = os.environ) -> "VLLMSettings":
        settings = cls(
            executable=environ.get("VLLM_EXECUTABLE") or cls.executable,
            gpu_memory_utilization=float(environ.get("VLLM_GPU_MEMORY_UTILIZATION") or cls.gpu_memory_utilization),
            startup_timeout=float(environ.get("VLLM_STARTUP_TIMEOUT") or cls.startup_timeout),
            cuda_visible_devices=environ.get("VLLM_CUDA_VISIBLE_DEVICES") or None,
        )
        if not 0 < settings.gpu_memory_utilization <= 1:
            raise ValueError("VLLM_GPU_MEMORY_UTILIZATION must be in (0, 1]")
        return settings


@contextlib.contextmanager
def serve(model: LocalModelConfig, settings: VLLMSettings | None = None) -> Iterator[str]:
    """Run vLLM for the duration of the block; yields its /v1 base URL."""
    settings = settings or VLLMSettings.from_environment()
    port = _free_port()
    command = [
        settings.executable, "serve", model.model_dir,
        "--served-model-name", model.served_model_name,
        "--host", "127.0.0.1",
        "--port", str(port),
        "--gpu-memory-utilization", str(settings.gpu_memory_utilization),
        *model.vllm_args,
    ]
    environment = dict(os.environ)
    executable_dir = os.path.dirname(settings.executable)
    if executable_dir:
        # As activating vLLM's environment would: vLLM runs tools installed
        # beside it, such as ninja when it compiles the model.
        environment["PATH"] = os.pathsep.join(filter(None, (executable_dir, environment.get("PATH"))))
    if settings.cuda_visible_devices is not None:
        environment["CUDA_VISIBLE_DEVICES"] = settings.cuda_visible_devices

    with tempfile.TemporaryDirectory(prefix="metakat_vllm_") as log_dir:
        log_path = Path(log_dir) / "vllm.log"
        logger.info("Starting vLLM for %s on port %d: %s", model.served_model_name, port, " ".join(command))
        started = time.monotonic()
        with log_path.open("wb") as log:
            try:
                process = subprocess.Popen(command, stdout=log, stderr=subprocess.STDOUT, env=environment,
                                           start_new_session=True)
            except OSError as error:
                raise RuntimeError(f"Cannot start vLLM ({settings.executable}): {error}") from error
        try:
            _wait_until_healthy(process, port, settings.startup_timeout, log_path)
            logger.info("vLLM serving %s after %.0f s", model.served_model_name, time.monotonic() - started)
            yield f"http://127.0.0.1:{port}/v1"
        finally:
            _stop(process)
            logger.info("vLLM for %s stopped", model.served_model_name)


def _wait_until_healthy(process: subprocess.Popen, port: int, timeout: float, log_path: Path) -> None:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if process.poll() is not None:
            raise RuntimeError(
                f"vLLM exited with code {process.returncode} before it was ready:\n{_tail(log_path)}"
            )
        try:
            with urllib.request.urlopen(f"http://127.0.0.1:{port}/health", timeout=5) as response:
                if response.status == 200:
                    return
        except (urllib.error.URLError, ConnectionError, TimeoutError):
            pass
        time.sleep(_POLL_SECONDS)
    raise RuntimeError(f"vLLM did not become ready within {timeout:.0f} s:\n{_tail(log_path)}")


def _stop(process: subprocess.Popen) -> None:
    # The whole group: vLLM runs its engine in child processes, and they hold
    # the GPU memory. A child left behind by a server that already exited is
    # killed too.
    if process.poll() is None:
        try:
            os.killpg(process.pid, signal.SIGTERM)
            process.wait(timeout=_STOP_TIMEOUT_SECONDS)
        except ProcessLookupError:
            pass
        except subprocess.TimeoutExpired:
            logger.warning("vLLM did not stop within %.0f s; killing it", _STOP_TIMEOUT_SECONDS)
    with contextlib.suppress(ProcessLookupError, PermissionError):
        os.killpg(process.pid, signal.SIGKILL)
    with contextlib.suppress(subprocess.TimeoutExpired):
        process.wait(timeout=10)


def local_model_locations(config: Any, location: str = "") -> list[str]:
    """Where a configuration sets up a local model: every `local` key, by path.

    A job override must not: a local model is part of the engine, and its
    settings start a process on the worker.
    """
    found = []
    if isinstance(config, Mapping):
        for key, value in config.items():
            path = f"{location}.{key}" if location else str(key)
            if key == "local":
                found.append(path)
            else:
                found.extend(local_model_locations(value, path))
    elif isinstance(config, list):
        for index, value in enumerate(config):
            found.extend(local_model_locations(value, f"{location}[{index}]"))
    return found


def _free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as probe:
        probe.bind(("127.0.0.1", 0))
        return probe.getsockname()[1]


def _tail(log_path: Path) -> str:
    try:
        lines = log_path.read_text(encoding="utf-8", errors="replace").splitlines()
    except OSError:
        return "(no log)"
    return "\n".join(lines[-_LOG_TAIL_LINES:])
