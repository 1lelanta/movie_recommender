"""
MovieLens dataset bootstrap utilities.

This module provides a small, dependency-free bootstrap flow for preparing the
MovieLens Latest Small dataset in the local data directory.
"""

from __future__ import annotations

import shutil
import tempfile
import urllib.error
import urllib.request
import zipfile
from pathlib import Path

MOVIELENS_LATEST_SMALL_URL = "https://files.grouplens.org/datasets/movielens/ml-latest-small.zip"


def ensure_movielens_data(data_dir: Path | str) -> tuple[Path, Path]:
    """
    Ensure that movies.csv and ratings.csv exist in the target directory.

    If either file is missing, the MovieLens Latest Small archive is downloaded
    and extracted into the directory.

    Args:
        data_dir: Directory where the MovieLens CSV files should live.

    Returns:
        A tuple containing the resolved movies.csv and ratings.csv paths.

    Raises:
        RuntimeError: If the dataset cannot be downloaded or extracted.
    """
    resolved_dir = Path(data_dir).expanduser().resolve()
    movies_path = resolved_dir / "movies.csv"
    ratings_path = resolved_dir / "ratings.csv"

    if movies_path.exists() and ratings_path.exists():
        return movies_path, ratings_path

    download_movielens_latest_small(resolved_dir)

    if not movies_path.exists() or not ratings_path.exists():
        raise RuntimeError(
            "The MovieLens dataset was downloaded, but the expected CSV files were not found."
        )

    return movies_path, ratings_path


def download_movielens_latest_small(data_dir: Path | str) -> None:
    """
    Download and extract the MovieLens Latest Small dataset.

    Args:
        data_dir: Target directory for movies.csv and ratings.csv.

    Raises:
        RuntimeError: If the download or extraction fails.
    """
    target_dir = Path(data_dir).expanduser().resolve()
    target_dir.mkdir(parents=True, exist_ok=True)

    with tempfile.TemporaryDirectory() as temp_dir_name:
        temp_dir = Path(temp_dir_name)
        archive_path = temp_dir / "ml-latest-small.zip"

        try:
            with urllib.request.urlopen(MOVIELENS_LATEST_SMALL_URL) as response, archive_path.open("wb") as output_file:
                shutil.copyfileobj(response, output_file)
        except urllib.error.URLError as exc:
            raise RuntimeError(
                "Could not download the MovieLens dataset. Check your internet connection and try again."
            ) from exc

        try:
            with zipfile.ZipFile(archive_path) as zip_file:
                zip_file.extractall(temp_dir)
        except zipfile.BadZipFile as exc:
            raise RuntimeError("The downloaded MovieLens archive is not a valid zip file.") from exc

        extracted_dir = temp_dir / "ml-latest-small"
        movies_source = extracted_dir / "movies.csv"
        ratings_source = extracted_dir / "ratings.csv"

        if not movies_source.exists() or not ratings_source.exists():
            raise RuntimeError("The downloaded archive did not contain the expected CSV files.")

        shutil.copy2(movies_source, target_dir / "movies.csv")
        shutil.copy2(ratings_source, target_dir / "ratings.csv")
