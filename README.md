# Movie Recommender

This project implements a content-based movie recommendation system using the MovieLens dataset.

It loads `movies.csv` and `ratings.csv`, cleans the data, builds TF-IDF features from movie genres, and computes cosine similarity to return movies that are most similar to a given title.

## Project Structure

```text
movie_recommender/
├── main.py
├── requirements.txt
├── README.md
├── data/
│   └── raw/
│       ├── movies.csv
│       └── ratings.csv
└── movie_recommender/
    ├── __init__.py
    ├── cli.py
    └── recommender.py
```

## Setup

1. Create and activate a virtual environment.
2. Install dependencies:

```bash
pip install -r requirements.txt
```

3. Download a MovieLens dataset version that contains `movies.csv` and `ratings.csv`.
4. Place both files in `data/raw/`.

## Run the CLI

```bash
python main.py
```

Then enter a movie title from the terminal to get the top 10 similar movies.

## Design Notes

- Data is loaded with pandas and validated before modeling.
- Missing values are handled explicitly during preprocessing.
- The recommendation engine is implemented as a `MovieRecommender` class.
- TF-IDF is applied to genres, and cosine similarity is used to rank candidates.
- Results include movie titles, genre labels, similarity scores, and rating statistics.
