"""Helpers for locating ``.ninfer`` artifacts and deriving a launch model id."""

from __future__ import annotations

import os
from pathlib import Path

from .process_manager import NInferConfigurationError

PACKAGE_ROOT = Path(__file__).resolve().parent.parent
_BUNDLED_MODELS_DIR = PACKAGE_ROOT / "artifacts"


def _comfy_folder_paths():
    try:
        import folder_paths  # type: ignore
    except ImportError:
        return None
    return folder_paths


def get_candidate_model_dirs() -> list[Path]:
    """Return existing directories where .ninfer models might be located."""
    candidates: list[Path] = []
    env_dir = os.environ.get("NINFER_MODELS_DIR", "").strip()
    if env_dir:
        candidates.append(Path(env_dir))

    if os.name == "nt":
        candidates.extend(
            [
                Path(r"C:\ninfer\artifacts"),
                Path(r"C:\ninfer\models"),
                Path(r"C:\ninfer"),
            ]
        )
    else:
        candidates.extend(
            [
                Path("/opt/ninfer-3090/artifacts"),
                Path("/opt/ninfer-3090/models"),
                Path("/opt/ninfer-3090/current/models"),
            ]
        )

    folder_paths = _comfy_folder_paths()
    comfy_models = getattr(folder_paths, "models_dir", None) if folder_paths else None
    if comfy_models:
        candidates.append(Path(comfy_models) / "ninfer")

    candidates.append(_BUNDLED_MODELS_DIR)
    candidates.append(Path.home() / "models" / "ninfer")
    candidates.append(Path.home() / "models")
    candidates.append(Path.home() / ".ninfer")

    seen: set[Path] = set()
    existing: list[Path] = []
    for c in candidates:
        try:
            resolved = c.expanduser().resolve()
            if resolved.is_dir() and resolved not in seen:
                seen.add(resolved)
                existing.append(resolved)
        except OSError:
            continue
    return existing


def discover_default_models_dir() -> str:
    """Find the most likely directory containing .ninfer models."""
    candidates = get_candidate_model_dirs()
    for d in candidates:
        try:
            if any(d.glob("*.ninfer")) or any(d.glob("*/*.ninfer")):
                return str(d)
        except OSError:
            continue
    if os.name == "nt" and Path(r"C:\ninfer\artifacts").is_dir():
        return str(Path(r"C:\ninfer\artifacts").resolve())
    if candidates:
        return str(candidates[0])
    return str(
        _BUNDLED_MODELS_DIR if _BUNDLED_MODELS_DIR.is_dir() else Path.home() / "models"
    )


# Prefer directory with existing models, or local installation artifacts.
# Keep a conventional per-user fallback for installations that omit the folder.
DEFAULT_MODELS_DIR = discover_default_models_dir()
EMPTY_MODEL_PLACEHOLDER = "(no .ninfer files found)"
_MODEL_ID_ALIASES = {
    "qwen3_8_27b": "qwen3.8-27b",
}



def register_model_folder() -> None:
    """Register NInfer artifacts with ComfyUI's native model registry."""
    folder_paths = _comfy_folder_paths()
    add_path = getattr(folder_paths, "add_model_folder_path", None) if folder_paths else None
    if not callable(add_path):
        return

    comfy_models = getattr(folder_paths, "models_dir", None)
    if comfy_models:
        add_path("ninfer", str(Path(comfy_models) / "ninfer"), is_default=True)
    seen: set[str] = set()
    for directory in [Path(DEFAULT_MODELS_DIR)] + get_candidate_model_dirs():
        try:
            resolved = str(directory.expanduser().resolve())
            if resolved not in seen:
                seen.add(resolved)
                add_path("ninfer", resolved)
        except OSError:
            continue


def native_model_names() -> list[str] | None:
    """Use ComfyUI's registry when available; return None in unit-test mode."""
    folder_paths = _comfy_folder_paths()
    get_names = getattr(folder_paths, "get_filename_list", None) if folder_paths else None
    if not callable(get_names):
        return None
    try:
        names = [str(name) for name in get_names("ninfer") if str(name).lower().endswith(".ninfer")]
    except (KeyError, OSError, RuntimeError):
        names = []

    if not names:
        scanned = scan_ninfer_models(DEFAULT_MODELS_DIR)
        if scanned != [EMPTY_MODEL_PLACEHOLDER]:
            names.extend(scanned)

    if not names:
        for candidate in get_candidate_model_dirs():
            scanned = scan_ninfer_models(str(candidate))
            if scanned != [EMPTY_MODEL_PLACEHOLDER]:
                names.extend(scanned)
                break

    cleaned: list[str] = []
    seen: set[str] = set()
    for name in names:
        if name and name != EMPTY_MODEL_PLACEHOLDER and name not in seen:
            seen.add(name)
            cleaned.append(name)

    return sorted(cleaned, key=str.lower) or [EMPTY_MODEL_PLACEHOLDER]


def native_model_path(model_artifact: str) -> str | None:
    """Resolve a registry selection through ComfyUI's path containment logic."""
    selected = (model_artifact or "").strip()
    if not selected or selected.startswith("(no .ninfer"):
        return None
    folder_paths = _comfy_folder_paths()
    get_path = getattr(folder_paths, "get_full_path", None) if folder_paths else None
    if not callable(get_path):
        return None
    for candidate_name in (selected, f"artifacts/{selected}", f"models/{selected}"):
        try:
            path = get_path("ninfer", candidate_name)
        except (KeyError, OSError, RuntimeError):
            path = None
        if path:
            resolved = Path(path).resolve()
            if resolved.suffix.lower() == ".ninfer" and resolved.is_file():
                return str(resolved)

    get_folders = getattr(folder_paths, "get_folder_paths", None)
    if callable(get_folders):
        try:
            folders = get_folders("ninfer")
        except (KeyError, OSError, RuntimeError):
            folders = []
        selected_path = Path(selected)
        for folder in folders:
            root = Path(folder).resolve()
            for sub in (
                root / selected_path,
                root / "artifacts" / selected_path,
                root / "models" / selected_path,
            ):
                if sub.suffix.lower() == ".ninfer" and sub.is_file():
                    try:
                        sub.resolve().relative_to(root)
                        return str(sub.resolve())
                    except ValueError:
                        continue
    return None


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

    # If candidate is not a file and selected_path is not absolute, check subfolders under root
    if (
        not candidate.is_file()
        and not selected_path.is_absolute()
        and ".." not in selected_path.parts
    ):
        for sub_dir in ("artifacts", "models"):
            sub_cand = (root / sub_dir / selected_path).resolve()
            if sub_cand.is_file():
                candidate = sub_cand
                break

    # If still not found and selected_path is not absolute, check fallbacks
    if (
        not candidate.is_file()
        and not selected_path.is_absolute()
        and ".." not in selected_path.parts
    ):
        fallback_dirs = [
            Path(DEFAULT_MODELS_DIR).expanduser().resolve()
        ] + get_candidate_model_dirs()
        for fallback_root in fallback_dirs:
            for sub_cand in (
                fallback_root / selected_path,
                fallback_root / "artifacts" / selected_path,
                fallback_root / "models" / selected_path,
            ):
                resolved_sub = sub_cand.resolve()
                if resolved_sub.is_file():
                    try:
                        resolved_sub.relative_to(fallback_root)
                        root = fallback_root
                        candidate = resolved_sub
                        break
                    except ValueError:
                        continue
            if candidate.is_file():
                break

    try:
        candidate.relative_to(root)
    except ValueError as exc:
        raise NInferConfigurationError(
            "model_artifact must resolve inside models_dir"
        ) from exc
    if candidate.suffix.lower() != ".ninfer":
        raise NInferConfigurationError("model_artifact must be a .ninfer file")
    if not candidate.is_file():
        raise NInferConfigurationError(
            f"NInfer model artifact was not found: {candidate}"
        )
    return str(candidate)
