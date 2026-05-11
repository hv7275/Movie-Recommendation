from typing import List

import sqlite3

from flask import Blueprint, flash, redirect, render_template, request, session, url_for

from .routes import (
    attach_tmdb_card_by_title,
    tmdb_card_from_results,
    tmdb_get,
    tmdb_movie_details,
    tmdb_search_movies,
    tfidf_recommend_title,
)
from .storage import (
    create_review,
    create_user,
    get_reviews_for_movie,
    get_user_by_id,
    get_user_by_username,
    password_matches,
)

frontend_bp = Blueprint(
    "frontend",
    __name__,
    template_folder="templates",
    static_folder="static",
)

CATEGORIES = [
    ("popular", "Popular"),
    ("trending", "Trending"),
    ("top_rated", "Top Rated"),
    ("upcoming", "Upcoming"),
    ("now_playing", "Now Playing"),
]


@frontend_bp.context_processor
def inject_current_user():
    user_id = session.get("user_id")
    return {"current_user": get_user_by_id(user_id) if user_id else None}


@frontend_bp.route("/register", methods=["GET", "POST"])
def register():
    if request.method == "POST":
        username = request.form.get("username", "").strip()
        password = request.form.get("password", "")
        confirm_password = request.form.get("confirm_password", "")

        if not username or not password:
            flash("Username and password are required.", "error")
        elif len(username) < 3:
            flash("Username must be at least 3 characters.", "error")
        elif len(password) < 6:
            flash("Password must be at least 6 characters.", "error")
        elif password != confirm_password:
            flash("Passwords do not match.", "error")
        else:
            try:
                create_user(username, password)
            except sqlite3.IntegrityError:
                flash("That username is already taken.", "error")
            else:
                user = get_user_by_username(username)
                session.clear()
                session["user_id"] = user["id"]
                flash("Account created. You are logged in.", "success")
                return redirect(url_for("frontend.home"))

    return render_template("register.html")


@frontend_bp.route("/login", methods=["GET", "POST"])
def login():
    if request.method == "POST":
        username = request.form.get("username", "").strip()
        password = request.form.get("password", "")
        user = get_user_by_username(username)

        if user and password_matches(user, password):
            session.clear()
            session["user_id"] = user["id"]
            flash("Welcome back.", "success")
            next_url = request.args.get("next")
            if next_url and next_url.startswith("/"):
                return redirect(next_url)
            return redirect(url_for("frontend.home"))

        flash("Invalid username or password.", "error")

    return render_template("login.html")


@frontend_bp.route("/logout")
def logout():
    session.clear()
    flash("You have been logged out.", "success")
    return redirect(url_for("frontend.home"))


@frontend_bp.route("/")
def home():
    query = request.args.get("query", "").strip()
    category = request.args.get("category", "popular")
    if category not in {item[0] for item in CATEGORIES}:
        category = "popular"

    cards = []
    error_message = None
    heading = "Popular Movies"

    if query:
        heading = f"Search results for \"{query}\""
        try:
            data = tmdb_search_movies(query=query, page=1)
            cards = tmdb_card_from_results(data.get("results", []), limit=24)
        except Exception as exc:
            error_message = str(exc)
    else:
        try:
            if category == "trending":
                data = tmdb_get("/trending/movie/day", {"language": "en-US"})
            else:
                data = tmdb_get(f"/movie/{category}", {"language": "en-US", "page": 1})
            cards = tmdb_card_from_results(data.get("results", []), limit=24)
            heading = dict(CATEGORIES).get(category, "Popular")
        except Exception as exc:
            error_message = str(exc)

    return render_template(
        "home.html",
        query=query,
        category=category,
        categories=CATEGORIES,
        cards=cards,
        heading=heading,
        error_message=error_message,
    )


@frontend_bp.route("/movie/<int:tmdb_id>")
def movie_detail(tmdb_id: int):
    details = None
    error_message = None

    try:
        details = tmdb_movie_details(tmdb_id)
    except Exception as exc:
        error_message = str(exc)

    tfidf_cards = []
    genre_cards = []

    if details:
        try:
            recommendations = tfidf_recommend_title(details.title, top_n=12)
            for title, _score in recommendations:
                card = attach_tmdb_card_by_title(title)
                if card:
                    tfidf_cards.append(card)
                    if len(tfidf_cards) >= 12:
                        break
        except Exception:
            tfidf_cards = []

        if details.genres:
            genre_id = details.genres[0].get("id")
            try:
                discover = tmdb_get(
                    "/discover/movie",
                    {
                        "with_genres": genre_id,
                        "language": "en-US",
                        "sort_by": "popularity.desc",
                        "page": 1,
                    },
                )
                genre_cards = [
                    card
                    for card in tmdb_card_from_results(discover.get("results", []), limit=12)
                    if card.tmdb_id != tmdb_id
                ]
            except Exception:
                genre_cards = []

    return render_template(
        "movie_detail.html",
        details=details,
        tfidf_cards=tfidf_cards,
        genre_cards=genre_cards,
        reviews=get_reviews_for_movie(tmdb_id),
        error_message=error_message,
    )


@frontend_bp.route("/movie/<int:tmdb_id>/reviews", methods=["POST"])
def add_review(tmdb_id: int):
    user_id = session.get("user_id")
    if not user_id:
        flash("Please log in to leave a review.", "error")
        return redirect(url_for("frontend.login", next=url_for("frontend.movie_detail", tmdb_id=tmdb_id)))

    rating_raw = request.form.get("rating", "")
    review_text = request.form.get("review_text", "").strip()

    try:
        rating = int(rating_raw)
    except ValueError:
        rating = 0

    if rating < 1 or rating > 10:
        flash("Rating must be between 1 and 10.", "error")
    elif not review_text:
        flash("Review text is required.", "error")
    else:
        create_review(tmdb_id, user_id, rating, review_text)
        flash("Review added.", "success")

    return redirect(url_for("frontend.movie_detail", tmdb_id=tmdb_id))
