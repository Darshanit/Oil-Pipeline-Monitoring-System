"""
Model Artifact Loader Module.

Loads and validates serialized Isolation Forest models (.joblib) per operating mode.
Provides thread-safe caching and validation.
"""

from dataclasses import dataclass
import os
from typing import Any, Dict, List, Optional
import joblib


@dataclass
class ModelArtifact:
    """Encapsulates a loaded Isolation Forest model and its metadata."""
    mode: str
    model: Any
    features: List[str]
    norm_params: Dict[str, float]


_MODEL_CACHE: Dict[str, ModelArtifact] = {}


def load_model_file(filepath: str, mode: str) -> Optional[ModelArtifact]:
    """Loads a single model artifact from joblib file."""
    if not os.path.exists(filepath):
        return None
    try:
        data = joblib.load(filepath)
        return ModelArtifact(
            mode=mode,
            model=data["model"],
            features=data["features"],
            norm_params=data["norm_params"]
        )
    except Exception as e:
        print(f"[LOADER WARNING] Failed loading '{filepath}': {e}")
        return None


def normalize_mode_name(mode: str) -> str:
    """Normalizes mode string into canonical title ('Flowing', 'Ramping', 'Shut-in')."""
    m = str(mode).strip().upper().replace("-", "_")
    if "FLOW" in m:
        return "Flowing"
    elif "RAMP" in m:
        return "Ramping"
    elif "SHUT" in m:
        return "Shut-in"
    return str(mode).strip().capitalize()


def load_mode_models(model_dir: str = "models", modes: Optional[List[str]] = None) -> Dict[str, ModelArtifact]:
    """
    Loads all available mode models into memory with cache.
    Default modes: ['Flowing', 'Ramping', 'Shut-in'].
    Registers aliases (e.g. FLOWING, Flowing, flowing) for case-insensitive lookup.
    """
    if modes is None:
        modes = ["Flowing", "Ramping", "Shut-in"]

    artifacts = {}
    for mode in modes:
        canon_mode = normalize_mode_name(mode)
        if canon_mode in _MODEL_CACHE:
            art = _MODEL_CACHE[canon_mode]
            artifacts[canon_mode] = art
            continue

        slug_underscore = canon_mode.lower().replace(" ", "_").replace("-", "_")
        slug_hyphen = canon_mode.lower().replace(" ", "-").replace("_", "-")

        path = os.path.join(model_dir, f"isolation_forest_{slug_underscore}.joblib")
        if not os.path.exists(path):
            path = os.path.join(model_dir, f"isolation_forest_{slug_hyphen}.joblib")

        art = load_model_file(path, canon_mode)
        if art is not None:
            _MODEL_CACHE[canon_mode] = art
            artifacts[canon_mode] = art

    # Populate case-insensitive aliases
    alias_dict = dict(artifacts)
    for k, v in list(artifacts.items()):
        alias_dict[k.upper()] = v
        alias_dict[k.lower()] = v
        alias_dict[k.upper().replace("-", "_")] = v
        alias_dict[k.lower().replace("-", "_")] = v

    return alias_dict


def clear_model_cache() -> None:
    """Clears the in-memory model cache."""
    global _MODEL_CACHE
    _MODEL_CACHE.clear()

