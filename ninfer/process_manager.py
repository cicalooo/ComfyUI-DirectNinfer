"""Lifecycle management for a short-lived NInfer server on Windows or Linux.

NInfer owns CUDA allocations in a native process, so clearing ComfyUI's
Python objects is not enough to release the model.  This module treats child
process exit as the release boundary and keeps all process handling in one
place so failures and interruptions use the same teardown path.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field, replace
import ipaddress
import logging
import os
from pathlib import Path
import re
import shlex
import shutil
import socket
import signal
import subprocess
import threading
import time
from typing import Any, Iterable, Mapping, Sequence

from .client import (
    ChatRequest,
    ChatResponse,
    NInferClient,
    NInferCancelledError,
    NInferClientError,
    NInferHTTPError,
    NInferTimeoutError,
    is_connection_refused_error,
    is_connection_reset_error,
)
from .vram import VramSnapshot, VramWaitResult, snapshot_vram, wait_for_vram_reclaim


LOGGER = logging.getLogger(__name__)


class NInferProcessError(RuntimeError):
    """Base class for process lifecycle failures."""


class NInferConfigurationError(NInferProcessError, ValueError):
    """The executable, artifact, or server options are invalid."""


class NInferStartupError(NInferProcessError):
    """The server could not become healthy."""


class NInferShutdownError(NInferProcessError):
    """The server or one of its tracked descendants did not exit."""


@dataclass(frozen=True)
class LifecycleTimeouts:
    """Independent deadlines for startup, request, process exit, and VRAM."""

    startup_s: float = 120.0
    request_s: float = 180.0
    graceful_shutdown_s: float = 10.0
    force_kill_s: float = 10.0
    vram_reclaim_s: float = 20.0

    def validate(self) -> None:
        for name, value in (
            ("startup_s", self.startup_s),
            ("request_s", self.request_s),
            ("graceful_shutdown_s", self.graceful_shutdown_s),
            ("force_kill_s", self.force_kill_s),
            ("vram_reclaim_s", self.vram_reclaim_s),
        ):
            if value <= 0:
                raise ValueError(f"{name} must be greater than zero")


@dataclass(frozen=True)
class ServerConfig:
    """Validated inputs used to construct one ``ninfer-serve`` command."""

    executable: str = "ninfer-serve.exe" if os.name == "nt" else "ninfer-serve"
    model_artifact: str = ""
    host: str = "127.0.0.1"
    # 0 asks the process manager to choose an available loopback port.
    port: int = 0
    model_id: str = "qwen3.8-27b"
    device: int = 0
    max_context: int = 4096
    kv_capacity: int | str = 4096
    max_concurrency: int = 1
    max_pending_requests: int = 1
    prefill_chunk: int = 512
    kv_dtype: str = "int8"
    spec_backend: str | None = "mtp"
    draft_tokens: int | None = 3
    lm_head_draft: bool = True
    vision: bool = False
    api_key: str | None = None
    default_max_tokens: int = 256
    no_cuda_graph: bool = False
    no_prefix_reuse: bool = False
    no_thinking: bool = False
    preserve_thinking: bool = False
    verify_model_id: bool = True
    extra_flags: Sequence[str] = field(default_factory=tuple)
    validate_flags: bool = True
    flag_probe_timeout_s: float = 5.0

    def __post_init__(self) -> None:
        object.__setattr__(self, "extra_flags", tuple(self.extra_flags))


@dataclass
class ServerHandle:
    """The running process and state needed to tear it down safely."""

    process: subprocess.Popen[str]
    config: ServerConfig
    executable: Path
    artifact: Path
    base_url: str
    started_at: float
    baseline_vram: VramSnapshot | None = None
    stderr_tail: deque[str] = field(default_factory=lambda: deque(maxlen=80))
    descendant_pids: set[int] = field(default_factory=set)
    stderr_thread: threading.Thread | None = None
    advertised_model_id: str | None = None

    @property
    def pid(self) -> int:
        return int(self.process.pid)

    def diagnostic_tail(self) -> str:
        return "".join(self.stderr_tail).strip()


@dataclass(frozen=True)
class ReclaimReport:
    """Evidence collected after the server teardown path."""

    graceful_requested: bool
    forced: bool
    process_exited: bool
    descendants_gone: bool | None
    vram_reclaimed: bool | None
    vram_before: VramSnapshot | None
    vram_after: VramSnapshot | None
    stderr_tail: str = ""
    notes: tuple[str, ...] = ()


_FLAG_RE = re.compile(r"--[A-Za-z0-9][A-Za-z0-9-]*")

_TYPED_FLAG_NAMES = {
    "--host",
    "--port",
    "--model-id",
    "--device",
    "--max-context",
    "--kv-capacity",
    "--max-concurrency",
    "--max-pending-requests",
    "--prefill-chunk",
    "--kv-dtype",
    "--api-key",
    "--default-max-tokens",
    "--spec",
    "--draft-tokens",
    "--lm-head-draft",
    "--vision",
    "--no-cuda-graph",
    "--no-prefix-reuse",
    "--no-thinking",
    "--preserve-thinking",
}


def parse_launch_flags(value: str | Sequence[str] | None) -> tuple[str, ...]:
    """Parse extra flags using the host shell's quoting conventions."""

    if value is None:
        return ()
    if isinstance(value, str):
        if not value.strip():
            return ()
        try:
            tokens = shlex.split(value, posix=os.name != "nt")
        except ValueError as exc:
            raise NInferConfigurationError(f"invalid server launch flags: {exc}") from exc
        cleaned: list[str] = []
        for token in tokens:
            if len(token) >= 2 and token[0] == token[-1] and token[0] in {"'", '"'}:
                token = token[1:-1]
            cleaned.append(token)
        return tuple(cleaned)
    return tuple(str(token) for token in value)


def _resolve_executable(executable: str) -> Path:
    if not executable or not executable.strip():
        raise NInferConfigurationError("NInfer executable path is empty")
    direct = Path(executable).expanduser()
    if direct.is_file():
        return direct.resolve()
    found = shutil.which(executable)
    if found:
        return Path(found).resolve()
    raise NInferConfigurationError(
        f"NInfer executable was not found: {executable!r}. "
        "Set the full path to ninfer-serve (Linux) or ninfer-serve.exe (Windows)."
    )


def _resolve_artifact(artifact: str) -> Path:
    if not artifact or not artifact.strip():
        raise NInferConfigurationError("NInfer model artifact path is empty")
    path = Path(artifact).expanduser()
    if not path.is_file():
        raise NInferConfigurationError(f"NInfer model artifact was not found: {artifact!r}")
    return path.resolve()


def _flag_name(token: str) -> str | None:
    if not token.startswith("--"):
        return None
    return token.split("=", 1)[0]


def _extra_flag_names(extra_flags: Sequence[str]) -> set[str]:
    names: set[str] = set()
    for token in extra_flags:
        name = _flag_name(token)
        if name:
            names.add(name)
    return names


def build_command(config: ServerConfig, *, executable: str | Path | None = None) -> list[str]:
    """Construct the argument vector passed to NInfer, never through a shell."""

    command = [
        str(executable if executable is not None else config.executable),
        str(config.model_artifact),
        "--host",
        config.host,
        "--port",
        str(config.port),
        "--model-id",
        config.model_id,
        "--device",
        str(config.device),
        "--max-context",
        str(config.max_context),
        "--kv-capacity",
        str(config.kv_capacity),
        "--max-concurrency",
        str(config.max_concurrency),
        "--max-pending-requests",
        str(config.max_pending_requests),
        "--prefill-chunk",
        str(config.prefill_chunk),
        "--kv-dtype",
        config.kv_dtype,
    ]
    if config.api_key:
        command.extend(["--api-key", config.api_key])
    if config.default_max_tokens > 0:
        command.extend(["--default-max-tokens", str(config.default_max_tokens)])
    if config.spec_backend:
        command.extend(["--spec", config.spec_backend])
        if config.draft_tokens is not None:
            command.extend(["--draft-tokens", str(config.draft_tokens)])
        if config.lm_head_draft:
            command.append("--lm-head-draft")
    if config.vision:
        command.append("--vision")
    if config.no_cuda_graph:
        command.append("--no-cuda-graph")
    if config.no_prefix_reuse:
        command.append("--no-prefix-reuse")
    if config.no_thinking:
        command.append("--no-thinking")
    if config.preserve_thinking:
        command.append("--preserve-thinking")
    command.extend(config.extra_flags)
    return command


def _generated_flag_names(config: ServerConfig) -> set[str]:
    command = build_command(config)
    names: set[str] = set()
    for index, token in enumerate(command):
        if index < 2:
            continue
        name = _flag_name(token)
        if name:
            names.add(name)
    return names


def probe_supported_flags(
    executable: str | Path, *, timeout_s: float = 5.0
) -> set[str]:
    """Read the installed binary's help text and return its long options."""

    try:
        result = subprocess.run(
            [str(executable), "--help"],
            stdin=subprocess.DEVNULL,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            encoding="utf-8",
            errors="replace",
            timeout=timeout_s,
            check=False,
        )
    except subprocess.TimeoutExpired as exc:
        raise NInferConfigurationError(
            f"timed out probing NInfer flags after {timeout_s:.1f}s"
        ) from exc
    except OSError as exc:
        raise NInferConfigurationError(f"could not run NInfer --help: {exc}") from exc
    text = f"{result.stdout}\n{result.stderr}"
    flags = set(_FLAG_RE.findall(text))
    if not flags:
        raise NInferConfigurationError(
            "NInfer --help returned no long options; cannot verify the configured "
            "server flags. Run the executable manually and check the installed build."
        )
    return flags


def validate_server_config(config: ServerConfig) -> tuple[Path, Path]:
    """Validate paths/ranges and reject flags absent from the installed binary."""

    executable = _resolve_executable(config.executable)
    artifact = _resolve_artifact(config.model_artifact)
    if not config.host.strip():
        raise NInferConfigurationError("host must not be empty")
    host = config.host.strip().lower()
    if host != "localhost":
        try:
            if not ipaddress.ip_address(host).is_loopback:
                raise NInferConfigurationError(
                    "host must be a loopback address; NInfer is launched without remote exposure"
                )
        except ValueError as exc:
            raise NInferConfigurationError(
                "host must be localhost or a loopback IP address"
            ) from exc
    if not 0 <= config.port <= 65535:
        raise NInferConfigurationError("port must be 0 (automatic) or between 1 and 65535")
    if not config.model_id.strip():
        raise NInferConfigurationError("model_id must not be empty")
    for name, value in (
        ("device", config.device),
        ("max_context", config.max_context),
        ("max_concurrency", config.max_concurrency),
        ("max_pending_requests", config.max_pending_requests),
        ("prefill_chunk", config.prefill_chunk),
        ("default_max_tokens", config.default_max_tokens),
    ):
        if value < 0:
            raise NInferConfigurationError(f"{name} must not be negative")
    if config.max_context < 1:
        raise NInferConfigurationError("max_context must be positive")
    if config.max_concurrency not in range(1, 9):
        raise NInferConfigurationError("max_concurrency must be between 1 and 8")
    if config.max_pending_requests < 0:
        raise NInferConfigurationError("max_pending_requests must not be negative")
    if config.prefill_chunk < 1:
        raise NInferConfigurationError("prefill_chunk must be positive")
    if config.kv_capacity != "auto":
        if not isinstance(config.kv_capacity, int) or config.kv_capacity < 1:
            raise NInferConfigurationError("kv_capacity must be a positive integer or 'auto'")
        if config.kv_capacity < config.max_context:
            raise NInferConfigurationError("kv_capacity must be at least max_context")
    if config.kv_dtype not in {"bf16", "int8", "rk8v4"}:
        raise NInferConfigurationError("kv_dtype must be bf16, int8, or rk8v4")
    if config.spec_backend not in {None, "mtp", "dflash"}:
        raise NInferConfigurationError("spec_backend must be off, mtp, or dflash")
    if config.spec_backend is None and (
        config.draft_tokens is not None or config.lm_head_draft
    ):
        raise NInferConfigurationError(
            "draft_tokens/lm_head_draft require a speculative backend"
        )
    if config.spec_backend == "dflash" and config.vision:
        raise NInferConfigurationError("NInfer DFlash cannot be combined with vision")
    if config.spec_backend == "mtp":
        if config.draft_tokens is None or not 1 <= config.draft_tokens <= 5:
            raise NInferConfigurationError("MTP draft_tokens must be between 1 and 5")
    if config.spec_backend == "dflash":
        if config.draft_tokens is None or not 1 <= config.draft_tokens <= 15:
            raise NInferConfigurationError("DFlash draft_tokens must be between 1 and 15")
    if config.flag_probe_timeout_s <= 0:
        raise NInferConfigurationError("flag_probe_timeout_s must be positive")

    extra_names = _extra_flag_names(config.extra_flags)
    if any(token and not token.startswith("--") for token in config.extra_flags):
        # Values are allowed after flags, but a standalone positional token is
        # almost always an accidental malformed launch string.  Negative
        # numeric values are valid values and are exempted.
        for index, token in enumerate(config.extra_flags):
            if token.startswith("-") and not token.startswith("--"):
                continue
            if not token.startswith("--"):
                previous = config.extra_flags[index - 1] if index else ""
                if not previous.startswith("--"):
                    raise NInferConfigurationError(
                        f"extra server launch token is not attached to a flag: {token!r}"
                    )
    reserved = _TYPED_FLAG_NAMES
    duplicate = sorted(reserved & extra_names)
    if duplicate:
        raise NInferConfigurationError(
            "extra launch flags duplicate typed settings: " + ", ".join(duplicate)
        )

    if config.validate_flags:
        supported = probe_supported_flags(
            executable, timeout_s=config.flag_probe_timeout_s
        )
        required = _generated_flag_names(config) | extra_names
        missing = sorted(flag for flag in required if flag not in supported)
        if missing:
            raise NInferConfigurationError(
                "installed NInfer does not advertise these configured flags: "
                + ", ".join(missing)
                + ". Run the configured NInfer executable with --help and adjust the node settings."
            )
    return executable, artifact


def _read_stderr(stream: Any, destination: deque[str]) -> None:
    try:
        for line in iter(stream.readline, ""):
            destination.append(line)
    except (OSError, ValueError):
        return
    finally:
        try:
            stream.close()
        except Exception:
            pass


def _initial_descendant_pids(pid: int) -> set[int]:
    try:
        import psutil  # type: ignore

        process = psutil.Process(pid)
        return {child.pid for child in process.children(recursive=True)}
    except (ImportError, OSError, RuntimeError):
        return set()


def start_server(
    config: ServerConfig,
    *,
    baseline_vram: VramSnapshot | None = None,
) -> ServerHandle:
    """Validate and launch one isolated NInfer process group."""

    executable, artifact = validate_server_config(config)
    family = socket.AF_INET6 if ":" in config.host else socket.AF_INET
    probe = socket.socket(family, socket.SOCK_STREAM)
    try:
        probe.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        probe.bind((config.host, config.port))
        selected_port = int(probe.getsockname()[1])
    except OSError as exc:
        raise NInferConfigurationError(
            f"NInfer port {config.host}:{config.port} is unavailable; "
            "choose another port or stop the process using it"
        ) from exc
    finally:
        probe.close()
    effective_config = replace(config, port=selected_port)
    command = build_command(effective_config, executable=executable)
    command[1] = str(artifact)
    kwargs: dict[str, Any] = {
        "stdin": subprocess.DEVNULL,
        "stdout": subprocess.DEVNULL,
        "stderr": subprocess.PIPE,
        "text": True,
        "encoding": "utf-8",
        "errors": "replace",
        "bufsize": 1,
        "close_fds": True,
    }
    if os.name == "nt":
        creationflags = getattr(subprocess, "CREATE_NEW_PROCESS_GROUP", 0)
        creationflags |= getattr(subprocess, "CREATE_NO_WINDOW", 0)
        kwargs["creationflags"] = creationflags
    else:
        kwargs["start_new_session"] = True
    try:
        process = subprocess.Popen(command, **kwargs)
    except OSError as exc:
        raise NInferStartupError(
            f"could not launch NInfer executable {executable}: {exc}"
        ) from exc
    stderr_tail: deque[str] = deque(maxlen=80)
    thread = threading.Thread(
        target=_read_stderr,
        args=(process.stderr, stderr_tail),
        name=f"ninfer-stderr-{process.pid}",
        daemon=True,
    )
    thread.start()
    url_host = effective_config.host
    try:
        if ipaddress.ip_address(url_host).version == 6:
            url_host = f"[{url_host}]"
    except ValueError:
        pass
    return ServerHandle(
        process=process,
        config=effective_config,
        executable=executable,
        artifact=artifact,
        base_url=f"http://{url_host}:{effective_config.port}",
        started_at=time.monotonic(),
        baseline_vram=baseline_vram,
        stderr_tail=stderr_tail,
        descendant_pids=_initial_descendant_pids(process.pid),
        stderr_thread=thread,
    )


def wait_until_ready(
    handle: ServerHandle,
    timeout: float,
    *,
    poll_interval: float = 0.25,
) -> None:
    """Poll ``/health`` until it responds successfully or the process fails."""

    if timeout <= 0:
        raise ValueError("startup timeout must be positive")
    client = NInferClient(handle.base_url, api_key=handle.config.api_key)
    deadline = time.monotonic() + timeout
    last_error: Exception | None = None
    while time.monotonic() < deadline:
        if handle.process.poll() is not None:
            detail = handle.diagnostic_tail()
            suffix = f"; stderr={detail[:800]}" if detail else ""
            raise NInferStartupError(
                f"NInfer exited before /health became ready (code={handle.process.returncode})"
                + suffix
            )
        try:
            remaining = max(0.01, deadline - time.monotonic())
            result = client.request("GET", "/health", timeout_s=min(2.0, remaining))
            if 200 <= result.status < 300:
                if handle.config.verify_model_id:
                    try:
                        models = client.get_json(
                            "/v1/models",
                            timeout_s=min(
                                2.0, max(0.01, deadline - time.monotonic())
                            ),
                        )
                        entries = models.get("data")
                        model_ids = {
                            str(entry.get("id"))
                            for entry in entries
                            if isinstance(entry, dict) and entry.get("id")
                        } if isinstance(entries, list) else set()
                        if not model_ids:
                            raise NInferStartupError(
                                "NInfer is healthy but /v1/models advertised no model ids"
                            )
                        preferred = handle.config.model_id.strip()
                        handle.advertised_model_id = (
                            preferred if preferred in model_ids else sorted(model_ids)[0]
                        )
                    except NInferStartupError:
                        raise
                else:
                    handle.advertised_model_id = handle.config.model_id
                return
            last_error = NInferHTTPError(result.status, "health check failed")
        except (NInferClientError, OSError) as exc:
            last_error = exc
        time.sleep(min(poll_interval, max(0.01, deadline - time.monotonic())))
    detail = f"; last error={last_error}" if last_error else ""
    stderr = handle.diagnostic_tail()
    if stderr:
        detail += f"; stderr={stderr[:800]}"
    raise NInferStartupError(
        f"NInfer did not become healthy within {timeout:.1f}s{detail}"
    )


def _vision_reset_hint(request: ChatRequest, stderr_tail: str, exit_code: int | None) -> str:
    has_images = any(
        isinstance(message.get("content"), list)
        and any(
            isinstance(part, dict) and part.get("type") == "image_url"
            for part in message["content"]
        )
        for message in request.messages
        if isinstance(message, Mapping)
    )
    stderr_l = stderr_tail.lower()
    heap_corrupt = exit_code in {3221226356, -1073740940}  # STATUS_HEAP_CORRUPTION
    if not has_images and "swscaler" not in stderr_l and not heap_corrupt:
        return ""
    return (
        "; ninfer-serve crashed while decoding vision input "
        f"(exit={exit_code}). The node now converts every ComfyUI IMAGE to a "
        "small RGB PNG on the fly; retry with vision_max_side=336/280, one "
        "frame, lower context/kv together, and unload_comfyui_before_launch "
        "enabled. JPEG wire format is rewritten to PNG because FFmpeg/swscaler "
        "JPEG paths can heap-corrupt this Windows runtime."
    )


def complete(
    handle: ServerHandle,
    request: ChatRequest,
    *,
    cancel_check: Any | None = None,
) -> ChatResponse:
    """Send one request through the managed server."""

    client = NInferClient(handle.base_url, api_key=handle.config.api_key)
    last_error: BaseException | None = None
    for attempt in range(2):
        try:
            # No client-side multi-retry: if ninfer dies mid-request the next
            # attempt becomes WinError 10061. We only retry when the process
            # is still alive after a reset.
            return client.complete(request, retries=0, cancel_check=cancel_check)
        except NInferCancelledError:
            raise
        except NInferTimeoutError:
            raise
        except NInferHTTPError:
            raise
        except NInferClientError as exc:
            last_error = exc
            exit_code = handle.process.poll()
            if (
                attempt == 0
                and exit_code is None
                and is_connection_reset_error(exc)
                and not is_connection_refused_error(exc)
            ):
                time.sleep(0.35)
                continue
            tail = handle.diagnostic_tail()
            if (
                exit_code is not None
                or tail
                or is_connection_reset_error(exc)
                or is_connection_refused_error(exc)
            ):
                detail = f"{exc}; ninfer exit={exit_code}"
                if tail:
                    detail += f"; stderr={tail[:1200]}"
                detail += _vision_reset_hint(request, tail, exit_code)
                raise NInferClientError(detail) from exc
            raise
    assert last_error is not None
    raise last_error


def _request_graceful_stop(process: subprocess.Popen[str]) -> bool:
    if process.poll() is not None:
        return False
    try:
        if os.name == "nt":
            # CREATE_NEW_PROCESS_GROUP lets CTRL_BREAK reach a console-aware
            # NInfer process without killing the ComfyUI parent.
            process.send_signal(signal.CTRL_BREAK_EVENT)
        else:
            os.killpg(os.getpgid(process.pid), signal.SIGTERM)
        return True
    except (OSError, ValueError, AttributeError):
        try:
            process.terminate()
            return True
        except OSError:
            return False


def _force_kill_tree(process: subprocess.Popen[str]) -> None:
    if process.poll() is not None:
        return
    if os.name == "nt":
        try:
            subprocess.run(
                ["taskkill", "/PID", str(process.pid), "/T", "/F"],
                stdin=subprocess.DEVNULL,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                check=False,
                timeout=10,
                creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
            )
        except (OSError, subprocess.TimeoutExpired):
            pass
    else:
        try:
            os.killpg(os.getpgid(process.pid), signal.SIGKILL)
        except (OSError, ProcessLookupError):
            pass
    try:
        process.kill()
    except OSError:
        pass


def _force_kill_pid(pid: int) -> None:
    """Force-kill a tracked descendant whose parent already exited."""

    if os.name == "nt":
        try:
            subprocess.run(
                ["taskkill", "/PID", str(pid), "/T", "/F"],
                stdin=subprocess.DEVNULL,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                check=False,
                timeout=10,
                creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
            )
        except (OSError, subprocess.TimeoutExpired):
            pass
    else:
        try:
            os.kill(pid, signal.SIGKILL)
        except (OSError, ProcessLookupError):
            pass


def _wait_for_pids_gone(pids: Iterable[int], timeout_s: float) -> bool | None:
    pids = {int(pid) for pid in pids if int(pid) > 0}
    if not pids:
        # Without psutil there are no descendant identities to check.  The
        # Windows taskkill /T path still covers the tree during force-kill.
        try:
            import psutil  # type: ignore
        except ImportError:
            return True
        return True
    try:
        import psutil  # type: ignore
    except ImportError:
        return None
    deadline = time.monotonic() + timeout_s
    while time.monotonic() < deadline:
        if not any(psutil.pid_exists(pid) for pid in pids):
            return True
        time.sleep(0.05)
    return not any(psutil.pid_exists(pid) for pid in pids)


def stop_server(
    handle: ServerHandle,
    timeouts: LifecycleTimeouts,
    *,
    vram_tolerance_mib: int = 256,
) -> ReclaimReport:
    """Stop the process group and wait for process/VRAM reclamation."""

    timeouts.validate()
    process = handle.process
    if process.poll() is None:
        handle.descendant_pids.update(_initial_descendant_pids(process.pid))
    graceful_requested = _request_graceful_stop(process)
    forced = False
    if graceful_requested:
        try:
            process.wait(timeout=timeouts.graceful_shutdown_s)
        except subprocess.TimeoutExpired:
            forced = True
    if process.poll() is None:
        forced = True
        # A server may create helper children after launch.  Refresh the
        # tracked set immediately before tree-killing so psutil-based
        # postconditions cover those children too.
        handle.descendant_pids.update(_initial_descendant_pids(process.pid))
        _force_kill_tree(process)
        try:
            process.wait(timeout=timeouts.force_kill_s)
        except subprocess.TimeoutExpired:
            pass
    process_exited = process.poll() is not None
    if process_exited and handle.descendant_pids:
        try:
            import psutil  # type: ignore

            live_descendants = [
                pid for pid in handle.descendant_pids if psutil.pid_exists(pid)
            ]
        except ImportError:
            live_descendants = []
        if live_descendants:
            forced = True
            for pid in live_descendants:
                _force_kill_pid(pid)
    descendants_gone = _wait_for_pids_gone(
        handle.descendant_pids, timeouts.force_kill_s
    )
    before = handle.baseline_vram
    vram_result: VramWaitResult | None
    if before is None:
        vram_result = None
        after = snapshot_vram(handle.config.device)
    else:
        vram_result = wait_for_vram_reclaim(
            before,
            timeout_s=timeouts.vram_reclaim_s,
            tolerance_bytes=max(0, vram_tolerance_mib) * 1024 * 1024,
            device=handle.config.device,
        )
        after = vram_result.snapshot
    notes: list[str] = []
    if forced:
        notes.append("graceful stop timed out; force-kill fallback was used")
    if vram_result is not None and vram_result.reclaimed is False:
        notes.append("VRAM did not return within the configured reclaim timeout")
    if not process_exited:
        notes.append("NInfer root process is still running")
    if descendants_gone is False:
        notes.append("one or more tracked descendants are still running")
    report = ReclaimReport(
        graceful_requested=graceful_requested,
        forced=forced,
        process_exited=process_exited,
        descendants_gone=descendants_gone,
        vram_reclaimed=None if vram_result is None else vram_result.reclaimed,
        vram_before=before,
        vram_after=after,
        stderr_tail=handle.diagnostic_tail(),
        notes=tuple(notes),
    )
    if not process_exited:
        raise NInferShutdownError(
            "NInfer did not exit after graceful and force-kill attempts"
            + (f"; stderr={report.stderr_tail[:800]}" if report.stderr_tail else "")
        )
    if descendants_gone is False:
        raise NInferShutdownError("NInfer descendants did not exit after force-kill")
    return report


__all__ = [
    "ChatRequest",
    "ChatResponse",
    "LifecycleTimeouts",
    "NInferConfigurationError",
    "NInferHTTPError",
    "NInferProcessError",
    "NInferShutdownError",
    "NInferStartupError",
    "ReclaimReport",
    "ServerConfig",
    "ServerHandle",
    "build_command",
    "complete",
    "parse_launch_flags",
    "probe_supported_flags",
    "start_server",
    "stop_server",
    "validate_server_config",
    "wait_until_ready",
]
