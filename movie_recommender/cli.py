"""Command-line interface for the movie recommender."""

from __future__ import annotations

from pathlib import Path

from .recommender import MovieRecommender, format_recommendations


def main() -> None:
    """Run an interactive terminal experience for the recommender."""

    recommender = MovieRecommender(data_dir=Path(__file__).resolve().parents[1] / "data" / "raw")

    try:
        recommender.build_model()
    except (FileNotFoundError, ValueError) as exc:
        print(f"Error: {exc}")
        print("Place movies.csv and ratings.csv in data/raw/ and try again.")
        return

    print("Movie Recommender ready.")
    print("Type a movie title exactly as it appears in the dataset, or press Enter to exit.\n")

    while True:
        movie_title = input("Enter a movie title: ").strip()
        if not movie_title:
            print("Exiting movie recommender.")
            break

        try:
            recommendations = recommender.recommend_movies(movie_title, top_n=10)
            print(f"\nTop recommendations for '{movie_title}':")
            print(format_recommendations(recommendations))
            print()
        except ValueError as exc:
            print(f"\n{exc}\n")
        except RuntimeError as exc:
            print(f"\nError: {exc}\n")
