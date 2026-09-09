"""Helpers for locating ``.ninfer`` artifacts and deriving a launch model id."""

from __future__ import annotations

from pathlib import Path

from .process_manager import NInferConfigurationError

PACKAGE_ROOT = Path(__file__).resolve().parent.parent
_BUNDLED_MODELS_DIR = PACKAGE_ROOT / "artifacts"
# Prefer the node's artifact directory so a freshly installed Linux node can
# discover local models immediately. Keep a conventional per-user fallback
# for installations that intentionally omit the repository artifact folder.
DEFAULT_MODELS_DIR = str(
    _BUNDLED_MODELS_DIR if _BUNDLED_MODELS_DIR.is_dir() else Path.home() / "models"
)
EMPTY_MODEL_PLACEHOLDER = "(no .ninfer files found)"
_MODEL_ID_ALIASES = {
    "qwen3_8_27b": "qwen3.8-27b",
}


def scan_ninfer_models(directory: str) -> list[str]:
    """Return relative paths of ``*.ninfer`` files under ``directory``."""

    root = Path(directory).expanduser()
    if not directory or not root.is_dir():
        return [EMPTY_MODEL_PLACEHOLDER]
    found: list[str] = []
    for path in root.rglob("*.ninfer"):
        if path.is_file():
            found.append(path.relative_to(root).as_posix())
    if not found:
        return [EMPTY_MODEL_PLACEHOLDER]
    found.sort(key=str.lower)
    return found


def derive_model_id(artifact: str | Path) -> str:
    """Launch ``--model-id`` from the artifact filename, with a small alias table."""

    stem = Path(artifact).stem.strip()
    if not stem:
        raise NInferConfigurationError("model artifact filename is empty")
    return _MODEL_ID_ALIASES.get(stem, stem)


def resolve_model_artifact(models_dir: str, model_artifact: str) -> str:
    """Join ``models_dir`` with a selection, including legacy workflow paths.

    ComfyUI serializes widget values into workflows. A workflow created with
    the old Windows default can therefore still contain ``C:\\models`` after
    it is opened on Linux. If that directory is unavailable, use the node's
    default artifact directory when it contains the selected file.
    """

    selected = (model_artifact or "").strip()
    if not selected or selected.startswith("(no .ninfer"):
        raise NInferConfigurationError(
            "No .ninfer artifact selected. Set models_dir and click Refresh."
        )
    path = Path(selected).expanduser()
    if path.is_absolute():
        return str(path)
    root = Path(models_dir).expanduser()
    candidate = root / selected
    if candidate.is_file():
        return str(candidate)
    fallback = Path(DEFAULT_MODELS_DIR) / selected
    if fallback.is_file():
        return str(fallback)
    return str(candidate)
