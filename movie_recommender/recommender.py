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
    """
    A content-based movie recommendation system using TF-IDF vectorization and cosine similarity.

    This class loads movie metadata and ratings, preprocesses the data, and provides
    recommendations based on genre similarity using TF-IDF vectorization.

    Attributes:
        movies_df (pd.DataFrame): DataFrame containing movie data.
        ratings_df (pd.DataFrame): DataFrame containing rating data.
        tfidf_matrix (sparse matrix): TF-IDF vectorized genres.
        similarity_matrix (np.ndarray): Cosine similarity matrix between movies.
    """

    def __init__(self, movies_path: str, ratings_path: str) -> None:
        """
        Initialize the MovieRecommender with movie and rating data.

        Args:
            movies_path (str): Path to the movies.csv file.
            ratings_path (str): Path to the ratings.csv file.

        Raises:
            FileNotFoundError: If either CSV file is not found.
            ValueError: If required columns are missing from the CSV files.
        """
        self.movies_df = None
        self.ratings_df = None
        self.tfidf_matrix = None
        self.similarity_matrix = None

        # Load and preprocess data
        self._load_data(movies_path, ratings_path)
        self._preprocess_data()
        self._compute_similarity_matrix()

    def _load_data(self, movies_path: str, ratings_path: str) -> None:
        """
        Load movies and ratings CSV files.

        Args:
            movies_path (str): Path to the movies.csv file.
            ratings_path (str): Path to the ratings.csv file.

        Raises:
            FileNotFoundError: If either file doesn't exist.
            ValueError: If required columns are missing.
        """
        # Check if files exist
        if not os.path.exists(movies_path):
            raise FileNotFoundError(f"Movies file not found: {movies_path}")
        if not os.path.exists(ratings_path):
            raise FileNotFoundError(f"Ratings file not found: {ratings_path}")

        print(f"Loading data from {movies_path} and {ratings_path}...")

        try:
            # Load CSV files
            self.movies_df = pd.read_csv(movies_path)
            self.ratings_df = pd.read_csv(ratings_path)

            # Validate required columns
            required_movie_cols = ['movieId', 'title', 'genres']
            if not all(col in self.movies_df.columns for col in required_movie_cols):
                raise ValueError(f"Movies CSV missing required columns: {required_movie_cols}")

            required_rating_cols = ['userId', 'movieId', 'rating']
            if not all(col in self.ratings_df.columns for col in required_rating_cols):
                raise ValueError(f"Ratings CSV missing required columns: {required_rating_cols}")

            print(f"✓ Loaded {len(self.movies_df)} movies and {len(self.ratings_df)} ratings")

        except pd.errors.ParserError as e:
            raise ValueError(f"Error parsing CSV files: {str(e)}")

    def _preprocess_data(self) -> None:
        """
        Clean and preprocess the data.

        This method handles:
        - Missing value detection
        - Genre standardization
        - Duplicate movie removal (keeping first occurrence)
        - Data validation
        """
        print("Preprocessing data...")

        # Check for missing values in critical columns
        missing_genres = self.movies_df['genres'].isna().sum()
        missing_titles = self.movies_df['title'].isna().sum()

        if missing_genres > 0:
            print(f"⚠ Warning: {missing_genres} movies have missing genres (filling with 'Unknown')")
            self.movies_df['genres'] = self.movies_df['genres'].fillna('Unknown')

        if missing_titles > 0:
            print(f"⚠ Warning: {missing_titles} movies have missing titles (removing)")
            self.movies_df = self.movies_df.dropna(subset=['title'])

        # Remove duplicates (keep first occurrence)
        initial_count = len(self.movies_df)
        self.movies_df = self.movies_df.drop_duplicates(subset=['movieId'], keep='first')
        duplicates_removed = initial_count - len(self.movies_df)

        if duplicates_removed > 0:
            print(f"⚠ Warning: Removed {duplicates_removed} duplicate movies")

        # Reset index for consistency
        self.movies_df = self.movies_df.reset_index(drop=True)
        print(f"✓ Data preprocessing complete: {len(self.movies_df)} movies ready for recommendations")

    def _compute_similarity_matrix(self) -> None:
        """
        Compute TF-IDF vectorization and cosine similarity matrix.

        This method:
        1. Initializes TF-IDF Vectorizer with genre data
        2. Transforms genre strings into TF-IDF vectors
        3. Computes pairwise cosine similarity
        """
        print("Computing TF-IDF and similarity matrix...")

        # Initialize TF-IDF Vectorizer
        # max_features limits vocabulary to top features
        # min_df=1 includes genres that appear at least once
        tfidf = TfidfVectorizer(
            analyzer='char',
            ngram_range=(1, 2),
            min_df=1,
            max_features=100
        )

        # Fit and transform genre data
        self.tfidf_matrix = tfidf.fit_transform(self.movies_df['genres'].fillna('Unknown'))

        # Compute cosine similarity between all pairs
        self.similarity_matrix = cosine_similarity(self.tfidf_matrix)

        print(f"✓ Similarity matrix computed ({self.similarity_matrix.shape[0]} x {self.similarity_matrix.shape[1]})")

    def _find_movie_index(self, movie_title: str) -> Optional[int]:
        """
        Find the index of a movie by title (case-insensitive).

        Args:
            movie_title (str): Title of the movie to find.

        Returns:
            Optional[int]: Index of the movie if found, None otherwise.
        """
        # Case-insensitive partial matching
        matches = self.movies_df[
            self.movies_df['title'].str.lower().str.contains(
                movie_title.lower(),
                na=False,
                regex=False
            )
        ]

        if len(matches) == 0:
            return None
        elif len(matches) == 1:
            return matches.index[0]
        else:
            # Return the first exact match, or first partial match
            exact_match = self.movies_df[
                self.movies_df['title'].str.lower() == movie_title.lower()
            ]
            if len(exact_match) > 0:
                return exact_match.index[0]
            return matches.index[0]

    def recommend_movies(
        self,
        movie_title: str,
        top_n: int = 10
    ) -> List[Tuple[str, float, str]]:
        """
        Get top N movie recommendations based on genre similarity.

        Uses cosine similarity to find movies with similar genre profiles to the
        input movie.

        Args:
            movie_title (str): Title of the reference movie.
            top_n (int): Number of recommendations to return (default: 10).

        Returns:
            List[Tuple[str, float, str]]: List of tuples containing
                (movie_title, similarity_score, genres).

        Raises:
            ValueError: If the movie is not found in the dataset.
        """
        # Find the movie index
        movie_index = self._find_movie_index(movie_title)

        if movie_index is None:
            raise ValueError(
                f"Movie '{movie_title}' not found in dataset. "
                "Please check the title and try again."
            )

        # Get the similarity scores for the target movie
        similarity_scores = self.similarity_matrix[movie_index]

        # Get indices of top similar movies (excluding the movie itself)
        # argsort gives indices in ascending order, so we reverse and skip the first
        similar_indices = np.argsort(similarity_scores)[::-1][1:top_n + 1]

        # Build recommendations list
        recommendations = []
        for idx in similar_indices:
            title = self.movies_df.loc[idx, 'title']
            score = similarity_scores[idx]
            genres = self.movies_df.loc[idx, 'genres']
            recommendations.append((title, float(score), genres))

        return recommendations

    def get_movie_info(self, movie_title: str) -> Optional[Dict]:
        """
        Retrieve detailed information about a movie.

        Args:
            movie_title (str): Title of the movie.

        Returns:
            Optional[Dict]: Dictionary with movie info, or None if not found.
        """
        movie_index = self._find_movie_index(movie_title)

        if movie_index is None:
            return None

        movie_data = self.movies_df.iloc[movie_index]
        return {
            'title': movie_data['title'],
            'genres': movie_data['genres'],
            'movieId': movie_data['movieId']
        }

    @staticmethod
    def display_recommendations(
        recommendations: List[Tuple[str, float, str]],
        reference_movie: str = ""
    ) -> None:
        """
        Display recommendations in a formatted table.

        Args:
            recommendations: List of recommendation tuples (title, score, genres).
            reference_movie: Optional title of the reference movie.
        """
        if reference_movie:
            print(f"\n{'=' * 80}")
            print(f"Recommendations based on: {reference_movie}")
            print(f"{'=' * 80}\n")
        else:
            print(f"\n{'=' * 80}")
            print("Movie Recommendations")
            print(f"{'=' * 80}\n")

        if not recommendations:
            print("No recommendations found.")
            return

        # Print header
        print(f"{'Rank':<6} {'Movie Title':<50} {'Similarity':<12} {'Genres'}")
        print("-" * 80)

        # Print recommendations
        for idx, (title, score, genres) in enumerate(recommendations, 1):
            # Truncate title if too long
            title_display = title[:47] + "..." if len(title) > 50 else title
            print(f"{idx:<6} {title_display:<50} {score:.4f}       {genres}")

        print(f"{'=' * 80}\n")

    def get_all_movies(self) -> List[str]:
        """
        Get a list of all movie titles in the dataset.

        Returns:
            List[str]: List of all movie titles.
        """
        return self.movies_df['title'].tolist()

    def search_movies(self, query: str) -> List[str]:
        """
        Search for movies by partial title match (case-insensitive).

        Args:
            query (str): Search query.

        Returns:
            List[str]: List of matching movie titles.
        """
        matches = self.movies_df[
            self.movies_df['title'].str.lower().str.contains(
                query.lower(),
                na=False,
                regex=False
            )
        ]
        return matches['title'].tolist()


if __name__ == "__main__":
    # Example usage
    try:
        recommender = MovieRecommender('data/raw/movies.csv', 'data/raw/ratings.csv')
        recommendations = recommender.recommend_movies('The Matrix', top_n=10)
        recommender.display_recommendations(recommendations, 'The Matrix')
    except FileNotFoundError as e:
        print(f"Error: {e}")
    except ValueError as e:
        print(f"Error: {e}")
