# Movie Recommender

>A simple movie recommendation app composed of a Flask frontend and a Flask backend. Uses TMDB for movie metadata and a TF-IDF based recommender built from the included dataset.

## Project structure

- `movie_recommendation/` — Flask application package with frontend templates and API routes
- `main.py` — Flask entrypoint serving the UI and API
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

1. Start the Flask application from the project root:

```powershell
python main.py
```

2. Open your browser at:

```text
http://127.0.0.1:8000/
```

The app depends on the pickled artifacts described above and the `TMDB_API_KEY` in `.env`.

## Troubleshooting

- If you see errors about missing `.pkl` files, generate them via `movies.ipynb`.
- If TMDB requests fail, ensure `TMDB_API_KEY` is present and valid in `.env`.
- If the web UI does not load, verify that `python main.py` is running and that port `8000` is available.

## Development notes

- The full app is now a Flask application served from `main.py`.
- The frontend uses Jinja templates in `movie_recommendation/templates/`.
- Use the notebook to experiment with TF-IDF parameters and regenerate artifacts.

---

If you'd like, I can also add a sample `.env.example`, or help generate the `.pkl` artifacts from the notebook—tell me which you'd prefer next.
