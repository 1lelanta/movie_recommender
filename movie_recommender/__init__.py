"""
Movie Recommender System Package.

A production-grade movie recommendation engine using content-based filtering
with TF-IDF vectorization and cosine similarity.

Classes:
    MovieRecommender: Core recommendation engine.
"""

from .recommender import MovieRecommender

__version__ = "1.0.0"
__author__ = "Your Name"
__all__ = ["MovieRecommender"]
