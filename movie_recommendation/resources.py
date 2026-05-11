from typing import Any, Dict, Optional

import httpx
import pandas as pd

from .config import Config
from .exceptions import AppError
from .utils import build_title_to_idx_map, load_pickle


df: Optional[pd.DataFrame] = None
indices_obj: Optional[Any] = None
tfidf_matrix: Any = None
tfidf_obj: Any = None
TITLE_TO_IDX: Optional[Dict[str, int]] = None
http_client: Optional[httpx.Client] = None


def load_resources() -> None:
    global df, indices_obj, tfidf_matrix, tfidf_obj, TITLE_TO_IDX, http_client

    if http_client is not None:
        return

    df = load_pickle(Config.DF_PATH)
    indices_obj = load_pickle(Config.INDICES_PATH)
    tfidf_matrix = load_pickle(Config.TFIDF_MATRIX_PATH)
    tfidf_obj = load_pickle(Config.TFIDF_PATH)
    TITLE_TO_IDX = build_title_to_idx_map(indices_obj)

    if df is None or "title" not in df.columns:
        raise AppError(
            "df.pkl must contain DataFrame with 'title' column",
            status_code=500,
        )

    http_client = httpx.Client(timeout=20.0)


def get_resources() -> Dict[str, Any]:
    return {
        "df": df,
        "tfidf_matrix": tfidf_matrix,
        "title_to_idx": TITLE_TO_IDX,
        "http_client": http_client,
    }
