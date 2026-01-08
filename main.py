import os
import pickle
from typing import Optional, List, Dict, Any, Tuple

import numpy as np
import pandas as pd
import httpx
from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from dotenv import load_dotenv


# ENV & CONSTANTS
load_dotenv()
TMDB_API_KEY = os.getenv("TMDB_API_KEY")

TMDB_BASE = "https://api.themoviedb.org/3"
TMDB_IMG_500 = "https://image.tmdb.org/t/p/w500"

if not TMDB_API_KEY:
    raise RuntimeError(
        "TMDB_API_KEY missing. Put it in .env as TMDB_API_KEY=xxxx"
    )


# APP SETUP
app = FastAPI(
    title="Movie Recommender API", 
    version="1.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],   # tighten in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# PATHS & GLOBAL RESOURCES
BASE_DIR = os.path.dirname(
    os.path.abspath(__file__)
)

DF_PATH = os.path.join(BASE_DIR, "df.pkl")
INDICES_PATH = os.path.join(BASE_DIR, "indices.pkl")
TFIDF_PATH = os.path.join(BASE_DIR, "tfidf.pkl")
TFIDF_MATRIX_PATH = os.path.join(BASE_DIR, "tfidf_matrix.pkl")

df: Optional[pd.DataFrame] = None
indices_obj: Any = None
tfidf_matrix: Any = None
tfidf_obj: Any = None

TITLE_TO_IDX: Optional[Dict[str, int]] = None

http_client: Optional[httpx.AsyncClient] = None


# MODELS

class TMDBMovieCard(BaseModel):
    """Lightweight movie card for grids and recommendations."""
    tmdb_id: int
    title: str
    poster_url: Optional[str] = None
    release_date: Optional[str] = None
    vote_average: Optional[float] = None


class TMDBMovieDetails(BaseModel):
    """Full movie details used on movie page."""
    tmdb_id: int
    title: str
    overview: Optional[str] = None
    release_date: Optional[str] = None
    poster_url: Optional[str] = None
    backdrop_url: Optional[str] = None
    genres: List[Dict[str, Any]] = Field(default_factory=list)


class TFIDFRecItem(BaseModel):
    """
    TF-IDF recommendation item.
    """

    title: str
    similarity: float
    tmdb: Optional[TMDBMovieCard] = None


class SearchBundleResponse(BaseModel):
    """
    Combined response for movie search + recommendations.
    """

    query: str
    movie_details: TMDBMovieDetails
    tfidf_recommendations: List[TFIDFRecItem]
    genre_recommendations: List[TMDBMovieCard]


# UTILITIES
def _norm_title(t: str) -> str:
    """
    Normalize movie titles for matching.
    """
    return str(t).strip().lower()


def make_image_url(path: Optional[str]) -> Optional[str]:
    """
    Build full TMDB image URL from path.
    """

    if not path:
        return None
    return f"{TMDB_IMG_500}{path}"


async def tmdb_get(path: str, params: Dict[str, Any]) -> Dict[str, Any]:
    """
    Safe TMDB GET request wrapper.

    - Network errors -> HTTP 502
    - TMDB errors -> HTTP 502 with details
    """
    global http_client

    if http_client is None:
        raise HTTPException(
            status_code=500, 
            detail="HTTP client not initialized"
        )

    q = dict(params)
    q["api_key"] = TMDB_API_KEY

    try:
        r = await http_client.get(
            f"{TMDB_BASE}{path}", 
            params=q
        )
    except httpx.RequestError as e:
        raise HTTPException(
            status_code=502,
            detail=f"TMDB request error: {type(e).__name__} | {repr(e)}",
        )

    if r.status_code != 200:
        raise HTTPException(
            status_code=502,
            detail=f"TMDB Error {r.status_code}: {r.text}",
        )

    return r.json()


def tmdb_card_from_results(results: List[dict], limit: int = 20) -> List[TMDBMovieCard]:
    """
    Convert TMDB search/discover results into movie cards.
    """

    out: List[TMDBMovieCard] = []

    for m in (results or [])[:limit]:
        out.append(
            TMDBMovieCard(
                tmdb_id=int(m["id"]),
                title=m.get("title") or m.get("name") or "",
                poster_url=make_image_url(m.get("poster_path")),
                release_date=m.get("release_date"),
                vote_average=m.get("vote_average"),
            )
        )

    return out


async def tmdb_movie_details(movie_id: int) -> TMDBMovieDetails:
    """
    Fetch full TMDB movie details by ID.
    """
    data = await tmdb_get(f"/movie/{movie_id}", {"language": "en-US"})

    return TMDBMovieDetails(
        tmdb_id=int(data["id"]),
        title=data.get("title") or "",
        overview=data.get("overview"),
        release_date=data.get("release_date"),
        poster_url=make_image_url(data.get("poster_path")),
        backdrop_url=make_image_url(data.get("backdrop_path")),
        genres=data.get("genres", []) or [],
    )


async def tmdb_search_movies(query: str, page: int = 1) -> Dict[str, Any]:
    """
    Raw TMDB keyword movie search.
    """
    return await tmdb_get(
        "/search/movie",
        {
            "query": query,
            "include_adult": False,
            "language": "en-US",
            "page": page,
        },
    )


async def tmdb_search_first(query: str) -> Optional[dict]:
    """
    Return first TMDB search result for a query.
    """
    data = await tmdb_search_movies(
        query=query, 
        page=1
    )
    results = data.get(
        "results"
    )
    return results[0] if results else None


def build_title_to_idx_map(indices: Any) -> Dict[str, int]:
    """
    Normalize title->index mapping.

    indices.pkl can be:
    - dict(title -> index)
    - pandas Series (index=title, value=index)
    """
    title_to_idx: Dict[str, int] = {}

    if isinstance(indices, dict):
        for k, v in indices.items():
            title_to_idx[_norm_title(k)] = int(v)
        return title_to_idx

    try:
        for k, v in indices.items():
            title_to_idx[_norm_title(k)] = int(v)
        return title_to_idx
    except Exception:
        raise RuntimeError(
            "indices.pkl must be dict or pandas Series-like (.items())"
        )


def get_local_idx_by_title(title: str) -> int:
    """
    Return TF-IDF row index for a movie title.
    """

    global TITLE_TO_IDX

    if TITLE_TO_IDX is None:
        raise HTTPException(
            status_code=500, 
            detail="TF-IDF index map not initialized"
        )

    key = _norm_title(title)
    if key in TITLE_TO_IDX:
        return int(TITLE_TO_IDX[key])

    raise HTTPException(
        status_code=404, 
        detail=f"Title not found: {title}"
    )


def tfidf_recommend_title(
    query_title: str, 
    top_n: int = 10
) -> List[Tuple[str, float]]:
    """
    Recommend similar titles using cosine similarity on TF-IDF matrix.

    Returns:
        List of (title, similarity_score)
    """
    global df, tfidf_matrix

    if df is None or tfidf_matrix is None:
        raise HTTPException(
            status_code=500, 
            detail="TF-IDF resources not loaded"
        )

    idx = get_local_idx_by_title(query_title)

    qv = tfidf_matrix[idx]
    score = (tfidf_matrix @ qv.T).toarray().ravel()

    order = np.argsort(-score)

    out: List[Tuple[str, float]] = []

    for i in order:
        if int(i) == int(idx):
            continue
        try:
            title_i = str(df.iloc[int(i)]["title"])
        except Exception:
            continue

        out.append((title_i, float(score[i])))

        if len(out) >= top_n:
            break

    return out


async def attach_tmdb_card_by_title(title: str) -> Optional[TMDBMovieCard]:
    """
    Attach TMDB poster/info to local dataset titles.
    """

    try:
        m = await tmdb_search_first(title)
        if not m:
            return None

        return TMDBMovieCard(
            tmdb_id=int(m["id"]),
            title=m.get("title") or title,
            poster_url=make_image_url(m.get("poster_path")),
            release_date=m.get("release_date"),
            vote_average=m.get("vote_average"),
        )
    except Exception as e:
        print("attach_tmdb_card_by_title failed:", e)
        return None


# STARTUP / SHUTDOWN
@app.on_event("startup")
async def startup():
    """
    Load models and create HTTP client.
    """
    global df, indices_obj, tfidf_matrix, tfidf_obj, TITLE_TO_IDX, http_client

    http_client = httpx.AsyncClient(timeout=20)

    with open(DF_PATH, "rb") as f:
        df = pickle.load(f)

    with open(INDICES_PATH, "rb") as f:
        indices_obj = pickle.load(f)

    with open(TFIDF_MATRIX_PATH, "rb") as f:
        tfidf_matrix = pickle.load(f)

    with open(TFIDF_PATH, "rb") as f:
        tfidf_obj = pickle.load(f)

    TITLE_TO_IDX = build_title_to_idx_map(indices_obj)

    if df is None or "title" not in df.columns:
        raise RuntimeError(
            "df.pkl must contain DataFrame with 'title' column"
        )


@app.on_event("shutdown")
async def shutdown():
    """Close HTTP client."""
    global http_client
    if http_client:
        await http_client.aclose()


# ROUTES
@app.get("/health")
def health():
    """Health check."""
    return {"status": "ok"}


@app.get("/home", response_model=List[TMDBMovieCard])
async def home(
    category: str = Query("popular"), 
    limit: int = Query(24, ge=1, le=50)
):
    """
    Home feed for poster grid.

    Categories:
    - trending
    - popular
    - top_rated
    - upcoming
    - now_playing
    """
    if category == "trending":
        data = await tmdb_get("/trending/movie/day", {"language": "en-US"})
        return tmdb_card_from_results(data.get("results", []), limit)

    if category not in {"popular", "top_rated", "upcoming", "now_playing"}:
        raise HTTPException(
            status_code=400, 
            detail="Invalid category"
        )

    data = await tmdb_get(f"/movie/{category}", {"language": "en-US", "page": 1})
    return tmdb_card_from_results(data.get("results", []), limit)


@app.get("/tmdb/search")
async def tmdb_search(
    query: str = Query(..., min_length=1), 
    page: int = Query(1, ge=1, le=10)
):
    """
    Raw TMDB keyword search (multiple results).
    """
    return await tmdb_search_movies(query=query, page=page)


@app.get("/movie/id/{tmdb_id}", response_model=TMDBMovieDetails)
async def movie_details_routes(tmdb_id: int):
    """
    Fetch movie details by TMDB ID.
    """

    return await tmdb_movie_details(tmdb_id)


@app.get("/recommand/genre", response_model=List[TMDBMovieCard])
async def recommand_genre(
    tmdb_id: int = Query(...), 
    limit: int = Query(18, ge=1, le=50)
):
    """
    Recommend popular movies from same genre as selected movie.
    """
    details = await tmdb_movie_details(tmdb_id)

    if not details.genres:
        return []

    genre_id = details.genres[0]["id"]

    discover = await tmdb_get(
        "/discover/movie",
        {
            "with_genres": genre_id,
            "language": "en-US",
            "sort_by": "popularity.desc",
            "page": 1,
        },
    )

    cards = tmdb_card_from_results(discover.get("results", []), limit)
    return [c for c in cards if c.tmdb_id != tmdb_id]


@app.get("/recommand/tfidf")
async def recommand_tfidf(
    title: str = Query(..., min_length=1), 
    top_n: int = Query(10, ge=1, le=50)
):
    """
    TF-IDF only recommendations.
    """
    recs = tfidf_recommend_title(title, top_n=top_n)
    return [{"title": t, "score": s} for t, s in recs]


@app.get("/movie/search", response_model=SearchBundleResponse)
async def search_bundle(
    query: str = Query(..., min_length=1),
    tfidf_top_n: int = Query(12, ge=1, le=30),
    genres_limit: int = Query(12, ge=1, le=30),
):
    """
    Full search bundle:
    - Best TMDB match
    - TF-IDF similar movies (with posters)
    - Genre-based recommendations
    """
    best = await tmdb_search_first(query)

    if not best:
        raise HTTPException(status_code=404, detail=f"No TMDB movie found for: {query}")

    tmdb_id = int(best["id"])
    details = await tmdb_movie_details(tmdb_id)

    # ---------- TF-IDF RECS ----------
    tfidf_items: List[TFIDFRecItem] = []

    try:
        recs = tfidf_recommend_title(details.title, top_n=tfidf_top_n)
    except Exception:
        try:
            recs = tfidf_recommend_title(query, top_n=tfidf_top_n)
        except Exception:
            recs = []

    for title, score in recs:
        card = await attach_tmdb_card_by_title(title)
        tfidf_items.append(TFIDFRecItem(
            title=title, 
            similarity=score, 
            tmdb=card)
        )

    # ---------- GENRE RECS ----------
    genre_recs: List[TMDBMovieCard] = []

    if details.genres:
        genre_id = details.genres[0]["id"]

        discover = await tmdb_get(
            "/discover/movie",
            {
                "with_genres": genre_id,
                "language": "en-US",
                "sort_by": "popularity.desc",
                "page": 1,
            },
        )

        genre_recs = tmdb_card_from_results(discover.get("results", []), genres_limit)
        genre_recs = [c for c in genre_recs if c.tmdb_id != tmdb_id]

    return SearchBundleResponse(
        query=query,
        movie_details=details,
        tfidf_recommendations=tfidf_items,
        genre_recommendations=genre_recs,
    )
