# 🎬 Movie Recommendation System

A production-grade content-based movie recommendation engine using TF-IDF vectorization and cosine similarity. This system analyzes movie genres to recommend similar films based on user preferences.

## Table of Contents

- [Features](#features)
- [Project Structure](#project-structure)
- [Installation](#installation)
- [Dataset](#dataset)
- [Usage](#usage)
- [How It Works](#how-it-works)
- [API Documentation](#api-documentation)
- [Code Quality](#code-quality)

---

## Features

✅ **Content-Based Filtering**: Recommendations based on genre similarity  
✅ **TF-IDF Vectorization**: Intelligent genre feature extraction  
✅ **Cosine Similarity**: Fast and accurate similarity computation  
✅ **Interactive CLI**: User-friendly command-line interface  
✅ **Robust Error Handling**: Graceful handling of edge cases  
✅ **Production-Quality Code**: Clean architecture, docstrings, and comments  
✅ **Object-Oriented Design**: Extensible `MovieRecommender` class  
✅ **Data Validation**: Comprehensive data cleaning and preprocessing  

---

## Project Structure

```
movie_recommender/
├── main.py                 # Application entry point
├── README.md              # Project documentation (this file)
├── requirements.txt       # Python dependencies
├── data/
│   └── raw/
│       ├── movies.csv     # Movie metadata (id, title, genres)
│       └── ratings.csv    # User ratings (userId, movieId, rating)
└── movie_recommender/
    ├── __init__.py        # Package initialization
    ├── recommender.py     # Core MovieRecommender class
    └── cli.py            # Command-line interface
```

### Key Files Explained

#### `main.py`
The application entry point that imports and runs the CLI.

#### `movie_recommender/recommender.py`
Contains the `MovieRecommender` class with:
- Data loading and validation
- Data preprocessing and cleaning
- TF-IDF vectorization
- Cosine similarity computation
- Recommendation generation
- Movie search and info retrieval

#### `movie_recommender/cli.py`
Provides an interactive command-line interface with:
- Main menu loop
- Movie recommendation generation
- Movie search functionality
- Movie information display

#### `data/raw/`
Contains MovieLens dataset CSV files:
- **movies.csv**: movieId, title, genres
- **ratings.csv**: userId, movieId, rating, timestamp

---

## Installation

### Prerequisites
- Python 3.8 or higher
- pip (Python package manager)

### Step 1: Clone or Navigate to Project

```bash
cd movie_recommender
```

### Step 2: Install Dependencies

```bash
pip install -r requirements.txt
```

This installs:
- **pandas**: Data loading and manipulation
- **numpy**: Numerical operations
- **scikit-learn**: TF-IDF Vectorizer and cosine similarity

---

## Dataset

### Downloading MovieLens Dataset

This project uses the MovieLens dataset. Download it from:  
**https://grouplens.org/datasets/movielens/**

Recommended: Use the **"MovieLens Latest Small"** (~1MB) for testing or **"MovieLens 25M"** for comprehensive analysis.

### Dataset Setup

1. Download the dataset (e.g., `ml-latest-small.zip`)
2. Extract it
3. Place `movies.csv` and `ratings.csv` in `data/raw/` directory:

```bash
data/
└── raw/
    ├── movies.csv
    └── ratings.csv
```

### CSV File Format

**movies.csv**:
```
movieId,title,genres
1,Toy Story (1995),Adventure|Animation|Children|Comedy|Fantasy
2,Jumanji (1995),Adventure|Children|Fantasy
...
```

**ratings.csv**:
```
userId,movieId,rating,timestamp
1,1,4.0,964982703
1,3,4.0,964981247
...
```

---

## Usage

### Running the Application

```bash
python main.py
```

### Interactive Menu

Once launched, you'll see the main menu:

```
================================================================================
🎬 Welcome to the Movie Recommender System
================================================================================

Options:
  1. Get movie recommendations
  2. Search for a movie
  3. Get movie info
  4. Exit

Enter your choice (1-4):
```

### Example: Get Recommendations

```
Enter your choice (1-4): 1

Enter a movie title: The Matrix
How many recommendations do you want? (default: 10): 5

================================================================================
Recommendations based on: The Matrix
================================================================================

Rank   Movie Title                                        Similarity   Genres
--------------------------------------------------------------------------------
1      The Matrix Reloaded (2003)                        0.8234       Action|Sci-Fi|Thriller
2      The Matrix Revolutions (2003)                     0.8105       Action|Sci-Fi|Thriller
3      Johnny Mnemonic (1995)                            0.7654       Action|Sci-Fi|Thriller
4      Hackers (1995)                                    0.7421       Crime|Sci-Fi|Thriller
5      Total Recall (1990)                               0.7312       Action|Sci-Fi|Thriller
================================================================================
```

### Example: Search for a Movie

```
Enter your choice (1-4): 2

Enter search query: Toy Story

✓ Found 4 matching movies (showing first 20):

   1. Toy Story (1995)
   2. Toy Story 2 (1999)
   3. Toy Story 3 (2010)
   4. Toy Story 4 (2019)
```

### Example: Get Movie Info

```
Enter your choice (1-4): 3

Enter a movie title: Inception

================================================================================
Title:   Inception (2010)
Genres:  Action|Sci-Fi|Thriller
MovieID: 27205
================================================================================
```

---

## How It Works

### 1. Data Loading
- Loads `movies.csv` and `ratings.csv` from the data directory
- Validates presence of required columns (movieId, title, genres)
- Validates data integrity

### 2. Data Preprocessing
- **Missing Value Handling**: Fills missing genres with "Unknown"
- **Duplicate Removal**: Keeps first occurrence of duplicate movies
- **Data Cleaning**: Removes movies with missing titles
- Provides status messages for transparency

### 3. TF-IDF Vectorization
- Converts movie genre strings into numerical feature vectors
- Uses character n-grams (1-2 characters) for better genre matching
- Parameters:
  - `analyzer='char'`: Uses character-level features
  - `ngram_range=(1, 2)`: Unigrams and bigrams
  - `min_df=1`: Includes genres appearing at least once
  - `max_features=100`: Limits vocabulary to 100 top features

### 4. Similarity Computation
- Computes cosine similarity between all pairs of movies
- Formula: `similarity = (A · B) / (||A|| × ||B||)`
- Results in an N×N matrix where N = number of movies
- Ranges from 0 (no similarity) to 1 (identical genres)

### 5. Recommendation Generation
1. Find the index of the query movie (case-insensitive)
2. Retrieve its similarity scores with all other movies
3. Sort by similarity in descending order
4. Exclude the query movie itself
5. Return top N recommendations with scores

### 6. Error Handling
- **Movie Not Found**: Provides helpful suggestions by searching for partial matches
- **File Not Found**: Clear error message with setup instructions
- **Invalid Input**: Graceful handling with user-friendly messages

---

## API Documentation

### MovieRecommender Class

#### Constructor
```python
recommender = MovieRecommender(movies_path, ratings_path)
```

**Parameters**:
- `movies_path` (str): Path to movies.csv
- `ratings_path` (str): Path to ratings.csv

**Raises**:
- `FileNotFoundError`: If CSV files don't exist
- `ValueError`: If required columns are missing

#### Methods

##### `recommend_movies()`
```python
recommendations = recommender.recommend_movies(movie_title, top_n=10)
```

Returns top N movie recommendations based on genre similarity.

**Parameters**:
- `movie_title` (str): Title of the reference movie
- `top_n` (int): Number of recommendations (default: 10)

**Returns**:
- `List[Tuple[str, float, str]]`: List of (title, similarity_score, genres)

**Raises**:
- `ValueError`: If movie not found

**Example**:
```python
recommendations = recommender.recommend_movies('Inception', top_n=5)
for title, score, genres in recommendations:
    print(f"{title}: {score:.4f} - {genres}")
```

##### `get_movie_info()`
```python
info = recommender.get_movie_info(movie_title)
```

Get detailed information about a movie.

**Returns**:
- `Optional[Dict]`: Dictionary with 'title', 'genres', 'movieId'

##### `search_movies()`
```python
matches = recommender.search_movies(query)
```

Search for movies by partial title match (case-insensitive).

**Returns**:
- `List[str]`: List of matching movie titles

##### `display_recommendations()`
```python
recommender.display_recommendations(recommendations, reference_movie)
```

Static method to display recommendations in a formatted table.

##### `get_all_movies()`
```python
all_movies = recommender.get_all_movies()
```

Get a list of all movie titles in the dataset.

---

## Code Quality

### Design Principles

1. **Object-Oriented Programming**
   - Encapsulation of related functionality in `MovieRecommender` class
   - Clear separation of concerns (data, processing, UI)

2. **Error Handling**
   - Comprehensive exception handling
   - User-friendly error messages
   - Input validation at all levels

3. **Documentation**
   - Module-level docstrings
   - Class and method docstrings
   - Inline comments for complex logic
   - Clear parameter and return type hints

4. **Data Validation**
   - File existence checking
   - Column presence validation
   - Missing value handling
   - Duplicate detection and removal

5. **Performance Optimization**
   - Vectorized operations using NumPy
   - Sparse matrix representation for TF-IDF
   - Efficient cosine similarity computation

### Type Hints

The code uses Python type hints throughout:
```python
def recommend_movies(
    self,
    movie_title: str,
    top_n: int = 10
) -> List[Tuple[str, float, str]]:
```

This improves:
- Code readability
- IDE autocomplete support
- Static type checking with tools like mypy

### Docstring Format

All functions follow Google-style docstrings:
```python
def method(param1: str, param2: int) -> List[str]:
    """
    Brief description.
    
    Longer description if needed.
    
    Args:
        param1: Description
        param2: Description
    
    Returns:
        Description of return value
    
    Raises:
        ExceptionType: When this occurs
    """
```

---

## Example: Using the Library Programmatically

```python
from movie_recommender import MovieRecommender

# Initialize recommender
recommender = MovieRecommender('data/raw/movies.csv', 'data/raw/ratings.csv')

# Get recommendations
recommendations = recommender.recommend_movies('The Shawshank Redemption', top_n=10)

# Display results
recommender.display_recommendations(recommendations, 'The Shawshank Redemption')

# Search for movies
similar_titles = recommender.search_movies('Avatar')
print(f"Found {len(similar_titles)} movies matching 'Avatar':")
for title in similar_titles:
    print(f"  - {title}")

# Get specific movie info
info = recommender.get_movie_info('Avatar (2009)')
if info:
    print(f"Genres: {info['genres']}")
```

---

## Extension Ideas

### Future Enhancements

1. **Collaborative Filtering**
   - Combine with user-user or item-item similarity
   - Use matrix factorization (SVD)

2. **Hybrid Approach**
   - Combine content-based and collaborative filtering
   - Weight different recommendation strategies

3. **Advanced Features**
   - Director and actor similarity
   - Release year and IMDB rating consideration
   - Temporal trends and recent movies boost

4. **Web Interface**
   - Create a Flask/Django web application
   - Build a React frontend

5. **Database Integration**
   - Store computed similarity matrix
   - Cache recommendations
   - Track user preferences over time

6. **Performance Improvements**
   - Implement approximate nearest neighbors (ANN)
   - Use faiss for faster similarity search

---

## Troubleshooting

### Error: "Movies file not found"
**Solution**: Ensure `data/raw/movies.csv` exists. Download from MovieLens dataset.

### Error: "Movie not found in dataset"
**Solution**: The exact movie title doesn't exist. Use option 2 (Search) to find similar titles.

### Slow Recommendations on Large Datasets
**Solution**: Consider using the MovieLens 100K dataset first, or optimize with ANN libraries.

### Module Import Errors
**Solution**: Ensure you're running from the project root:
```bash
cd movie_recommender
python main.py
```

---

## License

This project is provided as-is for educational purposes.

---

## Author

Created as a portfolio project demonstrating:
- Machine Learning fundamentals
- Clean Python code practices
- OOP design principles
- Data processing and analysis
- CLI development

---

## References

- **MovieLens Dataset**: https://grouplens.org/datasets/movielens/
- **Scikit-learn TF-IDF**: https://scikit-learn.org/stable/modules/generated/sklearn.feature_extraction.text.TfidfVectorizer.html
- **Cosine Similarity**: https://en.wikipedia.org/wiki/Cosine_similarity
- **Content-Based Filtering**: https://en.wikipedia.org/wiki/Recommender_system#Content-based_filtering

---

**Happy Recommending! 🍿**
