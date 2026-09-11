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


def _comfy_folder_paths():
    try:
        import folder_paths  # type: ignore
    except ImportError:
        return None
    return folder_paths


def register_model_folder() -> None:
    """Register NInfer artifacts with ComfyUI's native model registry."""
    folder_paths = _comfy_folder_paths()
    add_path = getattr(folder_paths, "add_model_folder_path", None) if folder_paths else None
    if callable(add_path):
        add_path("ninfer", str(_BUNDLED_MODELS_DIR), is_default=True)


def native_model_names() -> list[str] | None:
    """Use ComfyUI's registry when available; return None in unit-test mode."""
    folder_paths = _comfy_folder_paths()
    get_names = getattr(folder_paths, "get_filename_list", None) if folder_paths else None
    if not callable(get_names):
        return None
    try:
        names = [str(name) for name in get_names("ninfer") if str(name).lower().endswith(".ninfer")]
    except (KeyError, OSError, RuntimeError):
        return None
    return sorted(names, key=str.lower) or [EMPTY_MODEL_PLACEHOLDER]


def native_model_path(model_artifact: str) -> str | None:
    """Resolve a registry selection through ComfyUI's path containment logic."""
    folder_paths = _comfy_folder_paths()
    get_path = getattr(folder_paths, "get_full_path", None) if folder_paths else None
    if not callable(get_path):
        return None
    try:
        path = get_path("ninfer", model_artifact)
    except (KeyError, OSError, RuntimeError):
        return None
    if not path:
        return None
    resolved = Path(path).resolve()
    return str(resolved) if resolved.suffix.lower() == ".ninfer" and resolved.is_file() else None


def scan_ninfer_models(directory: str) -> list[str]:
    """Return relative paths of ``*.ninfer`` files under ``directory``."""

    root = Path(directory).expanduser()
    if not directory or not root.is_dir():
        return [EMPTY_MODEL_PLACEHOLDER]
    found: list[str] = []
    try:
        for path in root.rglob("*.ninfer"):
            if path.is_file():
                found.append(path.relative_to(root).as_posix())
    except OSError:
        return [EMPTY_MODEL_PLACEHOLDER]
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
    root = Path(models_dir).expanduser().resolve()
    selected_path = Path(selected).expanduser()
    candidate = selected_path if selected_path.is_absolute() else root / selected_path
    candidate = candidate.resolve()

    if not candidate.is_file() and not selected_path.is_absolute():
        fallback_root = Path(DEFAULT_MODELS_DIR).expanduser().resolve()
        fallback = (fallback_root / selected_path).resolve()
        if fallback.is_file():
            root = fallback_root
            candidate = fallback

    try:
        candidate.relative_to(root)
    except ValueError as exc:
        raise NInferConfigurationError(
            "model_artifact must resolve inside models_dir"
        ) from exc
    if candidate.suffix.lower() != ".ninfer":
        raise NInferConfigurationError("model_artifact must be a .ninfer file")
    if not candidate.is_file():
        raise NInferConfigurationError(f"NInfer model artifact was not found: {candidate}")
    return str(candidate)
