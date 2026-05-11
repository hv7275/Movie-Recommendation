import sqlite3
from typing import Dict, List, Optional

from flask import current_app, g
from werkzeug.security import check_password_hash, generate_password_hash


def get_db() -> sqlite3.Connection:
    if "db" not in g:
        g.db = sqlite3.connect(current_app.config["DATABASE_PATH"])
        g.db.row_factory = sqlite3.Row
    return g.db


def close_db(error=None) -> None:
    db = g.pop("db", None)
    if db is not None:
        db.close()


def init_db() -> None:
    db = get_db()
    db.executescript(
        """
        CREATE TABLE IF NOT EXISTS users (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            username TEXT NOT NULL UNIQUE,
            password_hash TEXT NOT NULL,
            created_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP
        );

        CREATE TABLE IF NOT EXISTS reviews (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            tmdb_id INTEGER NOT NULL,
            user_id INTEGER NOT NULL,
            rating INTEGER NOT NULL CHECK (rating BETWEEN 1 AND 10),
            review_text TEXT NOT NULL,
            created_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY (user_id) REFERENCES users (id)
        );
        """
    )
    db.commit()


def init_app(app) -> None:
    app.teardown_appcontext(close_db)
    with app.app_context():
        init_db()


def create_user(username: str, password: str) -> None:
    get_db().execute(
        "INSERT INTO users (username, password_hash) VALUES (?, ?)",
        (username, generate_password_hash(password)),
    )
    get_db().commit()


def get_user_by_username(username: str) -> Optional[Dict]:
    user = get_db().execute(
        "SELECT id, username, password_hash FROM users WHERE username = ?",
        (username,),
    ).fetchone()
    return dict(user) if user else None


def get_user_by_id(user_id: int) -> Optional[Dict]:
    user = get_db().execute(
        "SELECT id, username FROM users WHERE id = ?",
        (user_id,),
    ).fetchone()
    return dict(user) if user else None


def password_matches(user: Dict, password: str) -> bool:
    return check_password_hash(user["password_hash"], password)


def create_review(tmdb_id: int, user_id: int, rating: int, review_text: str) -> None:
    get_db().execute(
        """
        INSERT INTO reviews (tmdb_id, user_id, rating, review_text)
        VALUES (?, ?, ?, ?)
        """,
        (tmdb_id, user_id, rating, review_text),
    )
    get_db().commit()


def get_reviews_for_movie(tmdb_id: int) -> List[Dict]:
    rows = get_db().execute(
        """
        SELECT reviews.id, reviews.tmdb_id, reviews.rating, reviews.review_text,
               reviews.created_at, users.username
        FROM reviews
        JOIN users ON users.id = reviews.user_id
        WHERE reviews.tmdb_id = ?
        ORDER BY reviews.created_at DESC
        """,
        (tmdb_id,),
    ).fetchall()
    return [dict(row) for row in rows]
