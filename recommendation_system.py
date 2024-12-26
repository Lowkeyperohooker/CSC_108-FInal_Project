"""
MovieRecommender: A comprehensive movie recommendation system

A sophisticated recommendation engine combining content-based filtering and collaborative 
filtering techniques to suggest similar movies. Utilizes movie metadata (overview, cast, 
crew, genres) and user ratings for personalized recommendations.

Key Features:
- Content-based filtering using TF-IDF or Count vectorization
- Weighted rating calculation based on vote counts and averages 
- Multiple similarity computation methods (overview-based, metadata-based)
- Enhanced feature processing for cast, crew, keywords, and genres
- Comprehensive error handling and logging

Required Data Format:
- credits.csv: Contains movie credits with cast and crew information
    Columns: id, title, cast (JSON), crew (JSON)
- movies.csv: Contains movie metadata
    Columns: id, title, overview, vote_count, vote_average, genres (JSON), keywords (JSON)

Dependencies:
    - pandas: Data manipulation and DataFrame operations
    - numpy: Numerical computations and array operations
    - scikit-learn: TF-IDF, Count vectorization, cosine similarity
    - ast.literal_eval: Safe conversion of string representations to Python objects
"""

import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from ast import literal_eval
from typing import Union, List
import logging

logging.basicConfig(level=logging.DEBUG, format='%(levelname)s: %(message)s')
pd.set_option('display.max_columns', None)

class MovieRecommender:
    """
    A class to create and manage movie recommendations based on content similarity
    and user ratings.

    Attributes:
        df_movies (pd.DataFrame): Combined movie and credits data
        cosine_sim (np.ndarray): Cosine similarity matrix
        indices (pd.Series): Movie title to index mapping
    """
    
    def __init__(self, credits_path: str, movies_path: str):
        """
        Initialize MovieRecommender with data paths.

        Args:
            credits_path (str): Path to credits CSV file containing cast and crew info
            movies_path (str): Path to movies CSV file containing metadata

        Raises:
            FileNotFoundError: If either CSV file is not found
            pd.errors.EmptyDataError: If either CSV file is empty
            pd.errors.ParserError: If CSV files have invalid format
        """
        self.df_movies = self._load_and_merge_data(credits_path, movies_path)
        self.cosine_sim = None
        self.indices = None

    def _load_and_merge_data(self, credits_path: str, movies_path: str) -> pd.DataFrame:
        """
        Load and merge credits and movies datasets.

        Args:
            credits_path (str): Path to credits CSV file
            movies_path (str): Path to movies CSV file

        Returns:
            pd.DataFrame: Merged dataset with all movie information

        Raises:
            FileNotFoundError: If either file path is invalid
            pd.errors.EmptyDataError: If either file is empty
        """
        df_credits = pd.read_csv(credits_path)
        df_movies = pd.read_csv(movies_path)

        df_credits.columns = ['id', 'tittle', 'cast', 'crew']
        df_movies = df_movies.merge(df_credits, on='id')

        return df_movies

    def show_movies(self) -> pd.Series:
        """
        Get alphabetically sorted list of all movie titles.

        Returns:
            pd.Series: Sorted movie titles

        Note:
            Useful for exploring available movies and checking exact titles
            for recommendation queries.
        """
        return self.df_movies['title'].sort_values()

    def _calculate_weighted_rating(self, x: pd.Series, m: float, C: float) -> float:
        """
        Calculate weighted rating using IMDB formula: (v/(v+m) * R) + (m/(m+v) * C)

        Args:
            x (pd.Series): Row containing vote_count and vote_average
            m (float): Minimum votes required (vote count threshold)
            C (float): Mean vote across all movies

        Returns:
            float: Weighted rating score

        Formula Components:
            - v: Number of votes for the movie
            - m: Minimum votes required to be listed
            - R: Average rating of the movie
            - C: Mean vote across the whole dataset

        Note:
            This method reduces impact of movies with very few votes
            while considering both popularity and rating.
        """
        v = x['vote_count']
        R = x['vote_average']
        return (v / (v + m) * R) + (m / (m + v) * C)

    def prepare_data(self):
        """
        Prepare movie data for recommendation generation.

        Process:
        1. Calculate mean rating (C) across all movies
        2. Determine minimum vote threshold (m) using 90th percentile
        3. Filter movies below vote threshold
        4. Calculate weighted ratings using IMDB formula
        5. Sort movies by weighted rating

        Modifies:
            - df_movies: Updates DataFrame with qualified movies only
            - Adds 'score' column with weighted ratings

        Note:
            Should be called after initialization and before building similarity matrix
        """
        C = self.df_movies['vote_average'].mean()
        m = self.df_movies['vote_count'].quantile(0.9)

        qualified_movies = self.df_movies.copy().loc[self.df_movies['vote_count'] >= m]
        qualified_movies['score'] = qualified_movies.apply(
            self._calculate_weighted_rating, axis=1, m=m, C=C
        )
        self.df_movies = qualified_movies.sort_values('score', ascending=False).reset_index(drop=True)

    def get_recommendations(self, title: str) -> pd.Series:
        """
        Get movie recommendations based on content similarity.

        Args:
            title (str): Title of the movie to base recommendations on

        Returns:
            pd.Series: Series of 10 recommended movie titles, ordered by similarity

        Raises:
            ValueError: If similarity matrix hasn't been built
            KeyError: If movie title not found in dataset
            Exception: For other processing errors

        Process:
        1. Convert title to lowercase for matching
        2. Get movie index from title
        3. Calculate similarity scores with all other movies
        4. Sort by similarity and get top 10 (excluding input movie)
        5. Return titles of recommended movies

        Note:
            build_content_similarity() must be called before using this method
        """
        if self.cosine_sim is None or self.indices is None:
            logging.error("Cosine similarity matrix not built. Call 'build_content_similarity' first.")
            raise ValueError("Cosine similarity matrix not built.")

        title = title.lower()

        if title not in self.indices:
            logging.warning(f"Title '{title}' not found in indices.")
            raise KeyError(f"Movie '{title}' not found. Please check the title and try again.")

        idx = self.indices[title]
        logging.debug(f"Processing title: {title}, Index: {idx}")

        try:
            sim_scores = list(enumerate(self.cosine_sim[idx]))
            sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
            sim_scores = sim_scores[1:11]  # Top 10 similar movies
            movie_indices = [i[0] for i in sim_scores]

            logging.info(f"Recommendations successfully retrieved for '{title}'.")
            return self.df_movies['title'].iloc[movie_indices]
        except Exception as e:
            logging.error(f"Error while processing '{title}': {e}")
            raise

    def _get_director(self, crew: list) -> str:
        """
        Extract director's name from crew list.

        Args:
            crew (list): List of crew members and their roles
                Format: [{'name': str, 'job': str, ...}, ...]

        Returns:
            str: Director's name or np.nan if no director found

        Note:
            Assumes 'Director' is the exact job title in crew data
        """
        for member in crew:
            if member['job'] == 'Director':
                return member['name']
        return np.nan

    def _get_list(self, x: list) -> list:
        """
        Extract up to three names from a list of dictionaries.

        Args:
            x (list): List of dictionaries with 'name' key
                Format: [{'name': str, ...}, ...]

        Returns:
            list: Up to three names from the input list

        Note:
            Limits to three names to reduce noise in similarity calculations
            while maintaining most relevant information
        """
        if isinstance(x, list):
            names = [i['name'] for i in x]
            return names[:3] if len(names) > 3 else names
        return []

    def _clean_data(self, x: Union[list, str]) -> Union[list, str]:
        """
        Clean data by converting to lowercase and removing spaces.

        Args:
            x (Union[list, str]): Input data to clean

        Returns:
            Union[list, str]: Cleaned data in same format as input

        Process:
        - Convert to lowercase
        - Remove all spaces
        - Handle both string and list inputs
        """
        if isinstance(x, list):
            return [str.lower(i.replace(" ", "")) for i in x]
        if isinstance(x, str):
            return str.lower(x.replace(" ", ""))
        return ''

    def enhance_features(self):
        """
        Enhance movie features for improved similarity calculation.

        Process:
        1. Convert string representations to Python objects
        2. Extract director information from crew
        3. Limit cast, keywords, genres to top entries
        4. Clean and standardize all text data
        5. Create consolidated 'soup' of features

        Modifies:
            - df_movies: Updates multiple columns with processed data
            - Adds 'director' column
            - Adds 'soup' column for similarity calculation

        Note:
            Should be called before building similarity matrix
        """
        features = ['cast', 'crew', 'keywords', 'genres']
        for feature in features:
            self.df_movies[feature] = self.df_movies[feature].apply(literal_eval)

        self.df_movies['director'] = self.df_movies['crew'].apply(self._get_director)

        for feature in ['cast', 'keywords', 'genres']:
            self.df_movies[feature] = self.df_movies[feature].apply(self._get_list)

        for feature in ['cast', 'keywords', 'director', 'genres']:
            self.df_movies[feature] = self.df_movies[feature].apply(self._clean_data)

        self.df_movies['soup'] = self.df_movies.apply(
            lambda x: ' '.join(x['keywords']) + ' ' + ' '.join(x['cast']) + ' ' + 
                     x['director'] + ' ' + ' '.join(x['genres']),
            axis=1
        )

    def build_content_similarity(self):
        """
        Build content-based similarity matrix.

        Process:
        1. Create CountVectorizer with English stop words removed
        2. Transform 'soup' text into count matrix
        3. Calculate cosine similarity between all movies
        4. Create title to index mapping

        Modifies:
            - cosine_sim: Updates with new similarity matrix
            - indices: Updates with new title-to-index mapping

        Note:
            - enhance_features() should be called first
            - Memory usage scales quadratically with number of movies
        """
        count = CountVectorizer(stop_words='english')
        count_matrix = count.fit_transform(self.df_movies['soup'])

        self.cosine_sim = cosine_similarity(count_matrix, count_matrix)
        self.indices = pd.Series(
            self.df_movies.index, 
            index=self.df_movies['title'].str.lower()
        ).drop_duplicates()