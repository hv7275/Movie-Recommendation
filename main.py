import os
import pickle
from typing import Optional, List, Dict, Any, Tuple

import numpy as np
import pandas as pd
import httpx
from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from dotenv import load_dotenv

# .env and cros config
load_dotenv()
TMDB_API_KEY = os.getenv('TMDB_API_KEY')

TMDB_BASE = "https://api.themoviedb.org/3"
TMDB_IMG_500 = "https://image.tmdb.org/t/p/w500"

if not TMDB_API_KEY:
    # Don't crash import-time in production if you prefered; but for better fail early
    raise RuntimeError("TMDB_API_KEY missing. put in .env as TMDB_API_KEY=xxxx")


app = FastAPI(title='Movie Recommender API', version="1.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_header=["*"],
)

# path config and global vars config
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

DF_PATH = os.path.join(BASE_DIR, 'df.pkl')
INDICES_PATH = os.path.join(BASE_DIR, 'indices.pkl')
TFIDF_PATH = os.path.join(BASE_DIR, 'tfidf.pkl')
TFIDF_MATRIX_PATH = os.path.join(BASE_DIR, 'tfidf_matrix.pkl')


df: Optional[pd.DataFrame] = None
indices_obj: Any = None
tfidf_marix: Any = None
tfidf_obj: Any = None

TITLE_TO_IDX = Optional[Dict[str, int]] = None

# Models 
class TMDBMovieCard(BaseModel):
    tmdb_id:int
    title:str
    poster_url: Optional[str] = None
    release_date: Optional[str] = None
    vote_average: Optional[str] = None

class TMDBMovieDetails(BaseModel):
    tmdb_id: int
    title: str
    overview: Optional[str] = None
    release_date: Optional[str] = None
    poster_url: Optional[str] = None
    backdrop_url: Optional[str] = None
    genres: List[Dict] = []

class TFIDFRecTiem(BaseModel):
    title: str
    similarity: float
    tmdb: Optional[TMDBMovieCard] = None

class SearchBundleResponse(BaseModel):
    query: str
    movie_details: TMDBMovieDetails
    tfidf_recommendations: List[TFIDFRecTiem]
    genre_recommendations: List[TMDBMovieCard]

# Utilities Functions
def _norm_title(t: str) -> str:
    return str(t).strip().lower()

def make_image_url(path: Optional[str]) -> Optional[str]:
    if not path:
        return None
    else:
        return f"{TMDB_IMG_500}{path}"
    
async def tmdb_get(path: str, params: Dict[str, Any]) -> Dict[str, Any]:
    """

    safe TMDB Get
    -- Network Error -> 502
    -- TMDB API Error -> 502 With detail

    """
    q = dict(params)
    q['api_key'] = TMDB_API_KEY

    try:
        async with httpx.AsyncClient(timeout=20) as client:
            r = await client.get(f"{TMDB_BASE}{path}", params=q)
    except httpx.RequestError as e:
        raise HTTPException(
            status_code=502,
            detail=f"TMDB request error: {type(e).__name__} | {repr(e)}",
        )
    
    if r.status_code != 200:
        raise HTTPException(
            status_code=502,
            detail=f"TMDB Error {r.status_code}: {r.text}"
        )
    
    return r.json()


async def tmdb_card_from_results(result: List[dict], limit: int = 20) -> List[TMDBMovieCard]:
    out: List[TMDBMovieCard] = []
    for m in (result or [])[:limit]:
        out.append(
            TMDBMovieCard(
                tmdb_id=int(m['id']),
                title=m.get('title') or m.get('name') or "",
                poster_url=make_image_url(m.get('poster_path')),
                release_date=m.get('release_date'),
                vote_average=m.get('vote_average'),
            )
        )
    return out

async def tmdb_search_movies(query: str, page: int=1) -> Dict[str, Any]:
    """
    Docstring for tmdb_search_movies
    
    :param querry: Description
    :type querry: str
    :param page: Description
    :type page: int
    :return: Description
    :rtype: Dict[str, Any]

    Raw TMDB response for keyword search (MULTIPLE results).
    Streamlit will use this for suggestions and grid.
    """

    return await tmdb_get(
        "/search/movie",
        {
            'query':query,
            'include_adult':False,
            'language':'en-US',
            'page':page
        }
    )

async def tmdb_search_first(query: str) -> Optional[dict]:
    data = await tmdb_search_movies(query=query, page=1)
    results = data.get('results')
    return results[0] if results else None


def build_title_to_idx_map(indices: Any) -> Dict[str, int]:
    '''
    Docstring for build_title_to_idx_map
    
    :param indices: Description
    :type indices: Any
    :return: Description
    :rtype: Dict[str, int]

    indices.pkl can be:
    -- dict(title -> index)
    -- pandas series(index=title, value=index)
    We can normlize into TITLE_TO_IDX

    '''

    title_to_idx: Dict[str, int] = {}

    if isinstance(indices, dict):
        for k, v in indices.items():
            title_to_idx[_norm_title(k)] = int(v)
        return title_to_idx
    
    # Pandas Series or similar mapping
    try:
        for k, v in indices.items():
            title_to_idx[_norm_title(k)] = int(v)
        return title_to_idx
    except Exception:
        # Last resort: if it's a List Like etc...
        raise RuntimeError(
            "indcies.pkl must be dict or pandas series-Like (with .items())"
        )

def get_local_idx_by_title(title: str) -> int:
    global TITLE_TO_IDX
    
    if TITLE_TO_IDX is None:
        raise HTTPException(
            status_code=500,
            detail="TF-IDF index map not intialized"
        )
    
    key = _norm_title(title)
    if key in TITLE_TO_IDX:
        return int(TITLE_TO_IDX[key])
    raise HTTPException(
        status_code=404,
        detail=f"Title not found in local dataset: {title}"
    )


def tfidf_recommend_title(query_title: str, top_n: int = 10) -> List[Tuple[str, float]]:
    """
    Docstring for tfidf_recommend_title
    
    :param query_title: Description
    :type query_title: str
    :param top_n: Description
    :type top_n: int
    :return: Description
    :rtype: List[Tuple[str, float]]

    Return List of (title, score) from local df using cosine similarity on TF-IDF matrix
    Safe Againts missing columns

    """

    global df, tfidf_marix

    if df is None or tfidf_marix is None:
        raise HTTPException(
            status_code=500,
            detail='TF-IDF resource not loaded'
        )
    
    idx = get_local_idx_by_title(query_title)

    # query vector
    qv = tfidf_marix[idx]
    score = (tfidf_marix @ qv.T).toarray().ravel()

    # sort descending
    order = np.argsort(-score)

    out: List[Tuple[str, float]] = []
    for i in order:
        if int(i) == int(idx):
            continue
        try:
            title_i = str(df.iloc[int(i)]['title'])
        except Exception:
            continue

        if len(out) >= top_n:
            break

    return out

async def attach_tmdb_card_by_title(title: str) -> Optional[TMDBMovieCard]:
    """
    Docstring for attach_tmdb_card_by_title
    
    :param title: Description
    :type title: str
    :return: Description
    :rtype: TMDBMovieCard | None

    Uses TMDB search by title to fatch poster for a local title.
    If not found, returns None (never crashes the endpoints)
    """

    try:
        m = await tmdb_search_first(title)

        if not m:
            return None
        
        return TMDBMovieCard(
            tmdb_id=int(m['id']),
            title=m.get('title') or title,
            poster_url=make_image_url(m.get('poster_path')),
            release_date=m.get('release_date'),
            vote_average=m.get('vote_average')
        )
    
    except Exception:
        return None


# STARTUP: LOAD PICKLES
@app.on_event('startup')
def load_pickle():
    global df, indices_obj, tfidf_marix, tfidf_obj, TITLE_TO_IDX

    # Load df
    with open(DF_PATH, 'rb') as f:
        df = pickle.load(f)
    
    # Load Indices

    with open(INDICES_PATH, 'rb') as f:
        indices_obj = pickle.load(f)

    # Load TF-IDF Matrix (Usually  scipy sparse)
    with open(TFIDF_MATRIX_PATH, 'rb') as f:
        tfidf_marix = pickle.load(f)

    # Load TF-IDF  vectorizer (optional, not used directly here)
    with open(TFIDF_PATH, 'rb') as f:
        tfidf_obj = pickle.load(f)

    TITLE_TO_IDX = build_title_to_idx_map(indices_obj)

    # sansity
    if df is None or 'title' not in df.columns:
        raise RecursionError(
            "df.pkl must contain a DataFrame with a 'title' column"
        )


# HEALTH ROUTES
@app.get("/health")
def health():
    return {
        "status":"ok"
    }

# HOME ROUTES
@app.get('/home', response_model=List[TMDBMovieCard])
async def home(category: str = Query('popular'), limit: int = Query(24, ge=1, le=50)):
    """
    Docstring for home
    
    :param category: Description
    :type category: str
    :param limit: Description
    :type limit: int

    Home feed for Streamlit (posters).
    category:
        -- Tranding (trending/movie/day)
        -- Popular, top_rated, upcomming, now_playing (movie/{category})
    """

    try:
        if category == 'trending':
            data = await tmdb_get('/trending/movie/day', {'language':'en-US'})
            return await tmdb_card_from_results(data.get('results', []), limit=limit)
        
        if category not in {'popular', 'top_rated', 'upcoming', 'now_playing'}:
            raise HTTPException(
                status_code=400,
                detail='Invailid Category'
            )
        
        data = await tmdb_get(f'/movie/{category}', {'language':'en-US', 'page':1})
        return await tmdb_card_from_results(data.get('results', []), limit=limit)
    
    except HTTPException:
        raise

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Home route failed: {e}")


# SEARCH MULTIPLE VIA KEYWORDS
@app.get('/tmdb/search')
async def tmdb_search(
    qurry: str = Query(..., min_length=1),
    page: int = Query(1, ge=1, le=10)
):
    """
    Docstring for tmdb_search
    
    :param qurry: Description
    :type qurry: str
    :param page: Description
    :type page: int

    RETURN RAW TMDB shape with 'results' List.
    Streamlit will use it for:
        -- dropdown suggestions
        -- grid result
    """

    return await tmdb_search_movies(query=qurry, page=page)

