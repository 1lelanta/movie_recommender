"""
Command-line interface for the movie recommendation system.

This module provides an interactive CLI for users to get movie recommendations
through the terminal.
"""

from movie_recommender.recommender import MovieRecommender
from typing import Optional


def main() -> None:
    """
    Main entry point for the CLI application.

    Initializes the recommender and provides an interactive interface for users
    to search for movies and get recommendations.
    """
    print("\n" + "=" * 80)
    print("🎬 Welcome to the Movie Recommender System")
    print("=" * 80)

    try:
        # Initialize the recommender
        recommender = MovieRecommender('data/raw/movies.csv', 'data/raw/ratings.csv')
    except FileNotFoundError as e:
        print(f"\n❌ Error: {e}")
        print("\nPlease ensure you have:")
        print("  - data/raw/movies.csv")
        print("  - data/raw/ratings.csv")
        print("\nYou can download the MovieLens dataset from:")
        print("  https://grouplens.org/datasets/movielens/")
        return
    except ValueError as e:
        print(f"\n❌ Error: {e}")
        return

    # Interactive loop
    while True:
        print("\nOptions:")
        print("  1. Get movie recommendations")
        print("  2. Search for a movie")
        print("  3. Get movie info")
        print("  4. Exit")

        choice = input("\nEnter your choice (1-4): ").strip()

        if choice == "1":
            get_recommendations(recommender)
        elif choice == "2":
            search_movie(recommender)
        elif choice == "3":
            get_movie_info(recommender)
        elif choice == "4":
            print("\nThank you for using the Movie Recommender System! 👋\n")
            break
        else:
            print("❌ Invalid choice. Please enter 1-4.")


def get_recommendations(recommender: MovieRecommender) -> None:
    """
    Get and display movie recommendations based on user input.

    Args:
        recommender (MovieRecommender): The recommender instance.
    """
    print("\n" + "-" * 80)
    print("Get Movie Recommendations")
    print("-" * 80)

    movie_title = input("Enter a movie title: ").strip()

    if not movie_title:
        print("❌ Movie title cannot be empty.")
        return

    # Get the top N recommendations
    top_n_input = input("How many recommendations do you want? (default: 10): ").strip()
    try:
        top_n = int(top_n_input) if top_n_input else 10
        if top_n <= 0:
            raise ValueError("Number must be positive")
        top_n = min(top_n, 100)  # Cap at 100
    except ValueError:
        print("❌ Invalid input. Using default (10 recommendations).")
        top_n = 10

    try:
        # Get recommendations
        recommendations = recommender.recommend_movies(movie_title, top_n=top_n)

        # Display results
        recommender.display_recommendations(recommendations, movie_title)

    except ValueError as e:
        print(f"\n❌ Error: {e}")

        # Suggest similar titles
        print("\nDid you mean one of these?")
        suggestions = recommender.search_movies(movie_title[:3])[:5]
        for i, suggestion in enumerate(suggestions, 1):
            print(f"  {i}. {suggestion}")


def search_movie(recommender: MovieRecommender) -> None:
    """
    Search for movies by partial title match.

    Args:
        recommender (MovieRecommender): The recommender instance.
    """
    print("\n" + "-" * 80)
    print("Search for Movies")
    print("-" * 80)

    query = input("Enter search query: ").strip()

    if not query:
        print("❌ Search query cannot be empty.")
        return

    matches = recommender.search_movies(query)

    if not matches:
        print(f"❌ No movies found matching '{query}'")
    else:
        print(f"\n✓ Found {len(matches)} matching movies (showing first 20):\n")
        for i, title in enumerate(matches[:20], 1):
            print(f"  {i:2d}. {title}")

        if len(matches) > 20:
            print(f"\n  ... and {len(matches) - 20} more matches")


def get_movie_info(recommender: MovieRecommender) -> None:
    """
    Display information about a specific movie.

    Args:
        recommender (MovieRecommender): The recommender instance.
    """
    print("\n" + "-" * 80)
    print("Get Movie Information")
    print("-" * 80)

    movie_title = input("Enter a movie title: ").strip()

    if not movie_title:
        print("❌ Movie title cannot be empty.")
        return

    movie_info = recommender.get_movie_info(movie_title)

    if movie_info is None:
        print(f"❌ Movie '{movie_title}' not found.")
        print("\nDid you mean one of these?")
        suggestions = recommender.search_movies(movie_title[:3])[:5]
        for i, suggestion in enumerate(suggestions, 1):
            print(f"  {i}. {suggestion}")
    else:
        print("\n" + "=" * 80)
        print(f"Title:   {movie_info['title']}")
        print(f"Genres:  {movie_info['genres']}")
        print(f"MovieID: {movie_info['movieId']}")
        print("=" * 80)


if __name__ == "__main__":
    main()
