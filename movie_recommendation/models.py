from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List, Optional


@dataclass
class TMDBMovieCard:
    tmdb_id: int
    title: str
    poster_url: Optional[str] = None
    release_date: Optional[str] = None
    vote_average: Optional[float] = None

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class TMDBMovieDetails:
    tmdb_id: int
    title: str
    overview: Optional[str] = None
    release_date: Optional[str] = None
    poster_url: Optional[str] = None
    backdrop_url: Optional[str] = None
    runtime: Optional[int] = None
    tagline: Optional[str] = None
    genres: List[Dict[str, Any]] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class TFIDFRecItem:
    title: str
    similarity: float
    tmdb: Optional[TMDBMovieCard] = None

    def to_dict(self) -> Dict[str, Any]:
        result = {
            "title": self.title,
            "similarity": self.similarity,
            "tmdb": self.tmdb.to_dict() if self.tmdb else None,
        }
        return result


@dataclass
class SearchBundleResponse:
    query: str
    movie_details: TMDBMovieDetails
    tfidf_recommendations: List[TFIDFRecItem] = field(default_factory=list)
    genre_recommendations: List[TMDBMovieCard] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "query": self.query,
            "movie_details": self.movie_details.to_dict(),
            "tfidf_recommendations": [item.to_dict() for item in self.tfidf_recommendations],
            "genre_recommendations": [item.to_dict() for item in self.genre_recommendations],
        }
