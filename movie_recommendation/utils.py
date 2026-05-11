import os
import pickle
from typing import Any, Dict, Optional


def _norm_title(title: str) -> str:
    return str(title or "").strip().lower()


def make_image_url(path: Optional[str], image_base: str) -> Optional[str]:
    if not path:
        return None
    return f"{image_base}{path}"


def load_pickle(path: str) -> Any:
    if not os.path.exists(path):
        raise FileNotFoundError(f"Pickle file not found: {path}")
    with open(path, "rb") as handle:
        return pickle.load(handle)


def build_title_to_idx_map(indices: Any) -> Dict[str, int]:
    title_to_idx: Dict[str, int] = {}

    if isinstance(indices, dict):
        for key, value in indices.items():
            title_to_idx[_norm_title(key)] = int(value)
        return title_to_idx

    try:
        for key, value in indices.items():
            title_to_idx[_norm_title(key)] = int(value)
        return title_to_idx
    except Exception as exc:
        raise RuntimeError("indices.pkl must be dict or pandas Series-like (.items())") from exc
