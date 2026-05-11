from typing import Any, Dict, List, Optional, Tuple

import httpx
import numpy as np
from flask import Blueprint, current_app, jsonify, request

from .config import Config
from .exceptions import ExternalAPIError, InvalidUsage, NotFoundError
from .models import (
    SearchBundleResponse,
    TFIDFRecItem,
    TMDBMovieCard,
    TMDBMovieDetails,
)
from . import resources
from .utils import _norm_title, make_image_url

api_bp = Blueprint("api", __name__)


def tmdb_get(path: str, params: Dict[str, Any]) -> Dict[str, Any]:
    query = dict(params)
    query["api_key"] = Config.TMDB_API_KEY

    try:
        response = resources.http_client.get(f"{Config.TMDB_BASE}{path}", params=query, timeout=20.0)
    except httpx.RequestError as exc:
        raise ExternalAPIError(
            f"TMDB request failed: {type(exc).__name__} | {exc}",
        )

    if response.status_code != 200:
        raise ExternalAPIError(
            f"TMDB returned {response.status_code}: {response.text}",
        )

    return response.json()


def tmdb_card_from_results(results: List[Dict[str, Any]], limit: int = 20) -> List[TMDBMovieCard]:
    cards: List[TMDBMovieCard] = []
    for raw in (results or [])[:limit]:
        cards.append(
            TMDBMovieCard(
                tmdb_id=int(raw["id"]),
                title=raw.get("title") or raw.get("name") or "",
                poster_url=make_image_url(raw.get("poster_path"), Config.TMDB_IMG_500),
                release_date=raw.get("release_date"),
                vote_average=raw.get("vote_average"),
            )
        )
    return cards


def tmdb_movie_details(movie_id: int) -> TMDBMovieDetails:
    data = tmdb_get("/movie/{movie_id}".format(movie_id=movie_id), {"language": "en-US"})
    return TMDBMovieDetails(
        tmdb_id=int(data["id"]),
        title=data.get("title") or "",
        overview=data.get("overview"),
        release_date=data.get("release_date"),
        poster_url=make_image_url(data.get("poster_path"), Config.TMDB_IMG_500),
        backdrop_url=make_image_url(data.get("backdrop_path"), Config.TMDB_IMG_500),
        runtime=data.get("runtime"),
        tagline=data.get("tagline"),
        genres=data.get("genres", []) or [],
    )


def tmdb_search_movies(query: str, page: int = 1) -> Dict[str, Any]:
    return tmdb_get(
        "/search/movie",
        {
            "query": query,
            "include_adult": False,
            "language": "en-US",
            "page": page,
        },
    )


def tmdb_search_first(query: str) -> Optional[Dict[str, Any]]:
    data = tmdb_search_movies(query=query, page=1)
    results = data.get("results")
    return results[0] if results else None


def get_local_idx_by_title(title: str) -> int:
    if resources.TITLE_TO_IDX is None:
        raise InvalidUsage("TF-IDF index map is not initialized")

    key = _norm_title(title)
    if key in resources.TITLE_TO_IDX:
        return int(resources.TITLE_TO_IDX[key])

    raise NotFoundError(f"Title not found: {title}")


def tfidf_recommend_title(query_title: str, top_n: int = 10) -> List[Tuple[str, float]]:
    if resources.df is None or resources.tfidf_matrix is None:
        raise InvalidUsage("TF-IDF resources are not loaded")

    idx = get_local_idx_by_title(query_title)
    qv = resources.tfidf_matrix[idx]
    score = (resources.tfidf_matrix @ qv.T).toarray().ravel()
    order = np.argsort(-score)

    recommendations: List[Tuple[str, float]] = []
    for row_idx in order:
        if int(row_idx) == int(idx):
            continue
        try:
            title_value = str(resources.df.iloc[int(row_idx)]["title"])
        except Exception:
            continue

        recommendations.append((title_value, float(score[row_idx])))
        if len(recommendations) >= top_n:
            break

    return recommendations


def attach_tmdb_card_by_title(title: str) -> Optional[TMDBMovieCard]:
    result = tmdb_search_first(title)
    if not result:
        return None

    return TMDBMovieCard(
        tmdb_id=int(result["id"]),
        title=result.get("title") or title,
        poster_url=make_image_url(result.get("poster_path"), Config.TMDB_IMG_500),
        release_date=result.get("release_date"),
        vote_average=result.get("vote_average"),
    )


@api_bp.route("/health", methods=["GET"])
def health():
    return jsonify({"status": "ok"})


@api_bp.route("/home", methods=["GET"])
def home():
    category = request.args.get("category", "popular")
    try:
        limit = int(request.args.get("limit", 24))
    except ValueError:
        raise InvalidUsage("limit must be an integer")

    allowed = {"popular", "top_rated", "upcoming", "now_playing", "trending"}
    if category not in allowed:
        raise InvalidUsage("Invalid category")

    if category == "trending":
        data = tmdb_get("/trending/movie/day", {"language": "en-US"})
    else:
        data = tmdb_get(f"/movie/{category}", {"language": "en-US", "page": 1})

    cards = tmdb_card_from_results(data.get("results", []), limit)
    return jsonify([card.to_dict() for card in cards])


@api_bp.route("/tmdb/search", methods=["GET"])
def tmdb_search():
    query = request.args.get("query")
    if not query:
        raise InvalidUsage("query is required")
    try:
        page = int(request.args.get("page", 1))
    except ValueError:
        raise InvalidUsage("page must be an integer")

    return jsonify(tmdb_search_movies(query=query, page=page))


@api_bp.route("/movie/id/<int:tmdb_id>", methods=["GET"])
def movie_details_route(tmdb_id: int):
    return jsonify(tmdb_movie_details(tmdb_id).to_dict())


@api_bp.route("/recommand/genre", methods=["GET"])
def recommand_genre():
    tmdb_id = request.args.get("tmdb_id")
    if not tmdb_id:
        raise InvalidUsage("tmdb_id is required")
    try:
        tmdb_id_value = int(tmdb_id)
    except ValueError:
        raise InvalidUsage("tmdb_id must be an integer")

    limit = int(request.args.get("limit", 18))
    details = tmdb_movie_details(tmdb_id_value)
    if not details.genres:
        return jsonify([])

    genre_id = details.genres[0].get("id")
    discover = tmdb_get(
        "/discover/movie",
        {
            "with_genres": genre_id,
            "language": "en-US",
            "sort_by": "popularity.desc",
            "page": 1,
        },
    )
    cards = tmdb_card_from_results(discover.get("results", []), limit)
    filtered = [card.to_dict() for card in cards if card.tmdb_id != tmdb_id_value]
    return jsonify(filtered)


@api_bp.route("/recommand/tfidf", methods=["GET"])
def recommand_tfidf():
    title = request.args.get("title")
    if not title:
        raise InvalidUsage("title is required")
    try:
        top_n = int(request.args.get("top_n", 10))
    except ValueError:
        raise InvalidUsage("top_n must be an integer")

    recommendations = tfidf_recommend_title(title, top_n=top_n)
    return jsonify([{"title": title_value, "score": score} for title_value, score in recommendations])


@api_bp.route("/movie/search", methods=["GET"])
def search_bundle():
    query = request.args.get("query")
    if not query:
        raise InvalidUsage("query is required")

    try:
        tfidf_top_n = int(request.args.get("tfidf_top_n", 12))
        genres_limit = int(request.args.get("genres_limit", 12))
    except ValueError:
        raise InvalidUsage("tfidf_top_n and genres_limit must be integers")

    best = tmdb_search_first(query)
    if not best:
        raise NotFoundError(f"No TMDB movie found for: {query}")

    tmdb_id = int(best["id"])
    details = tmdb_movie_details(tmdb_id)

    tfidf_items: List[TFIDFRecItem] = []
    try:
        recs = tfidf_recommend_title(details.title, top_n=tfidf_top_n)
    except Exception:
        try:
            recs = tfidf_recommend_title(query, top_n=tfidf_top_n)
        except Exception:
            recs = []

    for title_value, score in recs:
        tfidf_items.append(
            TFIDFRecItem(
                title=title_value,
                similarity=score,
                tmdb=attach_tmdb_card_by_title(title_value),
            )
        )

    genre_recommendations: List[TMDBMovieCard] = []
    if details.genres:
        genre_id = details.genres[0].get("id")
        discover = tmdb_get(
            "/discover/movie",
            {
                "with_genres": genre_id,
                "language": "en-US",
                "sort_by": "popularity.desc",
                "page": 1,
            },
        )
        genre_recommendations = [
            card for card in tmdb_card_from_results(discover.get("results", []), genres_limit)
            if card.tmdb_id != tmdb_id
        ]

    bundle = SearchBundleResponse(
        query=query,
        movie_details=details,
        tfidf_recommendations=tfidf_items,
        genre_recommendations=genre_recommendations,
    )
    return jsonify(bundle.to_dict())
