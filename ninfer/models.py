"""Helpers for locating ``.ninfer`` artifacts and deriving a launch model id."""

from __future__ import annotations

from pathlib import Path

from .process_manager import NInferConfigurationError

DEFAULT_MODELS_DIR = r"C:\models"
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
    """Join ``models_dir`` with a relative selection, or keep an absolute path."""

    selected = (model_artifact or "").strip()
    if not selected or selected.startswith("(no .ninfer"):
        raise NInferConfigurationError(
            "No .ninfer artifact selected. Set models_dir and click Refresh."
        )
    path = Path(selected).expanduser()
    if path.is_absolute():
        return str(path)
    root = Path(models_dir).expanduser()
    return str(root / selected)
