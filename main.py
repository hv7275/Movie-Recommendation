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


async def tmdb_card_fromm_results(result: List[dict], limit: int = 20) -> List[TMDBMovieCard]:
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