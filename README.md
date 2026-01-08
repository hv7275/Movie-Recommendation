# Movie Recommender

>A simple movie recommendation app composed of a Streamlit frontend and a FastAPI backend. Uses TMDB for movie metadata and a TF-IDF based recommender built from the included dataset.

## Project structure

- `app.py` — Streamlit frontend UI
- `main.py` — FastAPI backend serving recommendations
- `movies.ipynb` — Notebook used to prepare datasets and TF-IDF artifacts
- `movies_metadata.csv` — Original metadata CSV used to build models
- `requirements.txt` — Python dependencies

## Features

- Search movies via TMDB
- Movie details and poster display
- TF-IDF and genre-based recommendations

## Prerequisites

- Python 3.10+ recommended
- A TMDB API key (get one from https://www.themoviedb.org)

## Setup

1. Create and activate a virtual environment:

```powershell
python -m venv .venv
.\.venv\Scripts\activate
```

2. Install dependencies:

```powershell
pip install -r requirements.txt
```

3. Create a `.env` file in the project root containing your TMDB key:

```
TMDB_API_KEY=your_tmdb_api_key_here
```

4. Generate the model artifacts required by the backend (`df.pkl`, `indices.pkl`, `tfidf.pkl`, `tfidf_matrix.pkl`). Either:

- Run the preparation notebook: [movies.ipynb](movies.ipynb) to produce the `.pkl` files, or
- Obtain precomputed `.pkl` files if available for this project.

## Running the app

1. Start the backend API (from the project root):

```powershell
uvicorn main:app --reload --port 8000
```

The backend depends on the pickled artifacts described above and the `TMDB_API_KEY` in `.env`.

2. Start the Streamlit frontend:

```powershell
streamlit run app.py
```

Note: `app.py` defaults to an external `API_BASE` hosted URL. To point the frontend to your local backend, open `app.py` and set `API_BASE = "http://127.0.0.1:8000"`.

## Troubleshooting

- If you see errors about missing `.pkl` files, generate them via `movies.ipynb`.
- If TMDB requests fail, ensure `TMDB_API_KEY` is present and valid in `.env`.
- If Streamlit shows stale state, try restarting the app or clearing the browser cache.

## Development notes

- The backend is implemented with FastAPI in `main.py` and exposes endpoints consumed by the Streamlit UI in `app.py`.
- Use the notebook to experiment with TF-IDF parameters and regenerate artifacts.

---

If you'd like, I can also add a sample `.env.example`, or help generate the `.pkl` artifacts from the notebook—tell me which you'd prefer next.
