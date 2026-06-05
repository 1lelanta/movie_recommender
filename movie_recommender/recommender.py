"""
Movie Recommendation Engine using Content-Based Filtering.

This module implements a content-based recommendation system using TF-IDF vectorization
and cosine similarity on movie genres. It loads movie metadata and provides recommendations
based on genre similarity.

Classes:
    MovieRecommender: Main class for movie recommendations.

Example:
    >>> recommender = MovieRecommender('data/raw/movies.csv', 'data/raw/ratings.csv')
    >>> recommendations = recommender.recommend_movies('The Matrix', top_n=10)
    >>> recommender.display_recommendations(recommendations)
"""

import os
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from typing import List, Tuple, Optional, Dict


class MovieRecommender:
    """Content-based recommender that uses movie genres as the feature space."""

    def __init__(self, data_dir: Optional[Path | str] = None) -> None:
        """Initialize the recommender with the directory containing MovieLens CSVs.

        Args:
            data_dir: Directory containing movies.csv and ratings.csv.
                If omitted, defaults to data/raw relative to the project root.
        """

        project_root = Path(__file__).resolve().parents[1]
        default_data_dir = project_root / "data" / "raw"
        self.data_dir = Path(data_dir).expanduser().resolve() if data_dir else default_data_dir

        self.movies: pd.DataFrame | None = None
        self.ratings: pd.DataFrame | None = None
        self.movie_features: pd.DataFrame | None = None
        self.similarity_matrix = None
        self.vectorizer: TfidfVectorizer | None = None

    def load_data(self) -> None:
        """Load movies.csv and ratings.csv, then clean and standardize the data."""

        movies_path = self.data_dir / "movies.csv"
        ratings_path = self.data_dir / "ratings.csv"

        if not movies_path.exists():
            raise FileNotFoundError(
                f"Missing movies.csv at {movies_path}. Place the MovieLens file in data/raw/."
            )

        if not ratings_path.exists():
            raise FileNotFoundError(
                f"Missing ratings.csv at {ratings_path}. Place the MovieLens file in data/raw/."
            )

        movies = pd.read_csv(movies_path)
        ratings = pd.read_csv(ratings_path)

        self.movies = self._clean_movies(movies)
        self.ratings = self._clean_ratings(ratings)

    def build_model(self) -> None:
        """Create the TF-IDF matrix and cosine similarity model from movie genres."""

        if self.movies is None or self.ratings is None:
            self.load_data()

        assert self.movies is not None
        assert self.ratings is not None

        movie_stats = (
            self.ratings.groupby("movieId", as_index=False)
            .agg(average_rating=("rating", "mean"), rating_count=("rating", "count"))
        )

        features = self.movies.merge(movie_stats, on="movieId", how="left")
        features["average_rating"] = features["average_rating"].fillna(0.0)
        features["rating_count"] = features["rating_count"].fillna(0).astype(int)
        features["genres"] = features["genres"].fillna("Unknown").replace("(no genres listed)", "Unknown")
        features["combined_features"] = features["genres"].str.replace("|", " ", regex=False)

        self.vectorizer = TfidfVectorizer(stop_words=None)
        tfidf_matrix = self.vectorizer.fit_transform(features["combined_features"])
        self.similarity_matrix = cosine_similarity(tfidf_matrix, tfidf_matrix)
        self.movie_features = features.reset_index(drop=True)

    def recommend_movies(self, movie_title: str, top_n: int = 10) -> pd.DataFrame:
        """Return the top N most similar movies for a given title.

        Args:
            movie_title: The title to search for in the dataset.
            top_n: Number of recommendations to return.

        Returns:
            A DataFrame with movie titles and similarity scores.

        Raises:
            ValueError: If the title is not found in the dataset.
            RuntimeError: If the model has not been built.
        """

        self._ensure_model_is_ready()

        assert self.movie_features is not None
        assert self.similarity_matrix is not None

        normalized_title = movie_title.strip().lower()
        title_matches = self.movie_features.index[
            self.movie_features["title"].str.lower() == normalized_title
        ].tolist()

        if not title_matches:
            close_matches = get_close_matches(movie_title, self.movie_features["title"].tolist(), n=5)
            suggestion_text = f" Did you mean: {', '.join(close_matches)}?" if close_matches else ""
            raise ValueError(f"Movie '{movie_title}' was not found in the dataset.{suggestion_text}")

        movie_index = title_matches[0]
        similarity_scores = list(enumerate(self.similarity_matrix[movie_index]))
        similarity_scores = sorted(similarity_scores, key=lambda item: item[1], reverse=True)

        ranked_movies = []
        for candidate_index, score in similarity_scores:
            if candidate_index == movie_index:
                continue

            row = self.movie_features.iloc[candidate_index]
            ranked_movies.append(
                Recommendation(
                    title=row["title"],
                    genres=row["genres"],
                    similarity_score=float(score),
                    average_rating=float(row["average_rating"]),
                    rating_count=int(row["rating_count"]),
                )
            )

            if len(ranked_movies) >= top_n:
                break

        recommendations = pd.DataFrame([item.__dict__ for item in ranked_movies])
        if recommendations.empty:
            return pd.DataFrame(columns=["title", "genres", "similarity_score", "average_rating", "rating_count"])

        return recommendations

    def _clean_movies(self, movies: pd.DataFrame) -> pd.DataFrame:
        """Clean the movies table and normalize the most important columns."""

        required_columns = {"movieId", "title", "genres"}
        missing_columns = required_columns.difference(movies.columns)
        if missing_columns:
            raise ValueError(f"movies.csv is missing required columns: {sorted(missing_columns)}")

        cleaned = movies.copy()
        cleaned["movieId"] = pd.to_numeric(cleaned["movieId"], errors="coerce")
        cleaned["title"] = cleaned["title"].fillna("Unknown Title").astype(str).str.strip()
        cleaned["genres"] = cleaned["genres"].fillna("Unknown").astype(str).str.strip()

        cleaned = cleaned.dropna(subset=["movieId"])
        cleaned["movieId"] = cleaned["movieId"].astype(int)
        cleaned = cleaned.drop_duplicates(subset=["movieId"], keep="first")
        cleaned = cleaned[cleaned["title"].ne("")].reset_index(drop=True)

        return cleaned

    def _clean_ratings(self, ratings: pd.DataFrame) -> pd.DataFrame:
        """Clean the ratings table and remove invalid rows."""

        required_columns = {"userId", "movieId", "rating"}
        missing_columns = required_columns.difference(ratings.columns)
        if missing_columns:
            raise ValueError(f"ratings.csv is missing required columns: {sorted(missing_columns)}")

        cleaned = ratings.copy()
        cleaned["userId"] = pd.to_numeric(cleaned["userId"], errors="coerce")
        cleaned["movieId"] = pd.to_numeric(cleaned["movieId"], errors="coerce")
        cleaned["rating"] = pd.to_numeric(cleaned["rating"], errors="coerce")

        cleaned = cleaned.dropna(subset=["userId", "movieId", "rating"])
        cleaned = cleaned[(cleaned["rating"] >= 0) & (cleaned["rating"] <= 5)]

        cleaned["userId"] = cleaned["userId"].astype(int)
        cleaned["movieId"] = cleaned["movieId"].astype(int)

        if "timestamp" in cleaned.columns:
            cleaned["timestamp"] = pd.to_numeric(cleaned["timestamp"], errors="coerce").fillna(0).astype(int)

        return cleaned.reset_index(drop=True)

    def _ensure_model_is_ready(self) -> None:
        """Guard against calling recommend_movies before the model is built."""

        if self.movie_features is None or self.similarity_matrix is None:
            raise RuntimeError("The recommendation model is not ready. Call build_model() first.")


def format_recommendations(recommendations: pd.DataFrame) -> str:
    """Format recommendations for terminal output."""

    if recommendations.empty:
        return "No recommendations found."

    display_frame = recommendations.copy()
    display_frame["similarity_score"] = display_frame["similarity_score"].map(lambda value: f"{value:.4f}")
    display_frame["average_rating"] = display_frame["average_rating"].map(lambda value: f"{value:.2f}")
    return display_frame.to_string(index=False)
