import os

from dotenv import load_dotenv

load_dotenv()


class Config:
    SECRET_KEY = os.getenv("SECRET_KEY", "dev-secret-key-change-me")
    TMDB_API_KEY = os.getenv("TMDB_API_KEY")
    if not TMDB_API_KEY:
        raise RuntimeError("TMDB_API_KEY missing. Put it in .env as TMDB_API_KEY=xxxx")

    TMDB_BASE = "https://api.themoviedb.org/3"
    TMDB_IMG_500 = "https://image.tmdb.org/t/p/w500"

    BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    DF_PATH = os.path.join(BASE_DIR, "df.pkl")
    INDICES_PATH = os.path.join(BASE_DIR, "indices.pkl")
    TFIDF_PATH = os.path.join(BASE_DIR, "tfidf.pkl")
    TFIDF_MATRIX_PATH = os.path.join(BASE_DIR, "tfidf_matrix.pkl")
    DATABASE_PATH = os.path.join(BASE_DIR, "movie_recommendation.sqlite3")

    LOG_DIR = os.path.join(BASE_DIR, "logs")
    CORS_ORIGINS = ["*"]
