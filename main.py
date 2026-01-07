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
TMD_IMG_500 = "https://image.tmdb.org/t/p/w500"

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