"""
MovieRecommender: A comprehensive movie recommendation system

This class implements a movie recommendation engine that uses content-based filtering
and collaborative filtering techniques to suggest similar movies. It combines movie
metadata (overview, cast, crew, genres) with user ratings to generate personalized
recommendations.

Key Features:
- Content-based filtering using TF-IDF or Count vectorization
- Weighted rating calculation based on vote counts and averages
- Multiple similarity computation methods (overview-based, metadata-based)
- Enhanced feature processing for cast, crew, keywords, and genres
- Comprehensive error handling and logging

Usage:
    recommender = MovieRecommender('credits.csv', 'movies.csv')
    recommender.prepare_data()                # Process and clean the data
    recommender.build_cosine_similarity()     # Build similarity matrix
    recommendations = recommender.get_recommendations('Movie Title')

Required Data Format:
- credits.csv: Contains movie credits with cast and crew information
- movies.csv: Contains movie metadata (title, overview, ratings, etc.)

Dependencies:
    - pandas: Data manipulation
    - numpy: Numerical operations
    - scikit-learn: TF-IDF, Count vectorization, and similarity calculations
    - ast.literal_eval: Safe string to list/dict conversion
"""

import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity, linear_kernel
from ast import literal_eval
from typing import Union, List
import logging

# Configure logging with timestamp and log level
logging.basicConfig(level=logging.DEBUG, format='%(levelname)s: %(message)s')

# Set pandas display options for better debugging
pd.set_option('display.max_columns', None)  # Show all columns

class MovieRecommender:
    """
    A class to create and manage movie recommendations based on content similarity
    and user ratings.
    """
    
    def __init__(self, credits_path: str, movies_path: str):
        """
        Initialize the MovieRecommender with movie and credits data.

        Args:
            credits_path (str): Path to the credits CSV file
            movies_path (str): Path to the movies CSV file

        Attributes:
            df_movies (pd.DataFrame): Combined movie and credits data
            cosine_sim (np.ndarray): Cosine similarity matrix (initialized as None)
            indices (pd.Series): Movie title to index mapping (initialized as None)
        """
        self.df_movies = self._load_and_merge_data(credits_path, movies_path)
        self.cosine_sim = None
        self.indices = None

    def _load_and_merge_data(self, credits_path: str, movies_path: str) -> pd.DataFrame:
        """
        Load and merge credits and movies data from CSV files.

        Args:
            credits_path (str): Path to the credits CSV file
            movies_path (str): Path to the movies CSV file

        Returns:
            pd.DataFrame: Merged dataset containing both movie and credits information
        """
        df_credits = pd.read_csv(credits_path)
        df_movies = pd.read_csv(movies_path)

        df_credits.columns = ['id', 'tittle', 'cast', 'crew']
        df_movies = df_movies.merge(df_credits, on='id')

        return df_movies

    def show_movies(self) -> pd.Series:
        """
        Get a sorted list of all movie titles in the dataset.

        Returns:
            pd.Series: Alphabetically sorted list of movie titles
        """
        return self.df_movies['title'].sort_values()

    def _calculate_weighted_rating(self, x: pd.Series, m: float, C: float) -> float:
        """
        Calculate the weighted rating for a movie using the IMDB weighted rating formula.
        
        Formula: (v/(v+m) * R) + (m/(m+v) * C)
        where:
        - v is the number of votes for the movie
        - m is the minimum votes required to be listed
        - R is the average rating of the movie
        - C is the mean vote across the whole dataset

        Args:
            x (pd.Series): Row of movie data containing vote_count and vote_average
            m (float): Minimum votes required (quantile cutoff)
            C (float): Mean vote across all movies

        Returns:
            float: Weighted rating score
        """
        v = x['vote_count']
        R = x['vote_average']
        return (v / (v + m) * R) + (m / (m + v) * C)

    def prepare_data(self):
        """
        Prepare the movie data by:
        1. Calculating the mean rating across all movies
        2. Setting a minimum vote threshold (90th percentile)
        3. Filtering movies based on minimum votes
        4. Calculating weighted ratings
        5. Sorting movies by weighted rating

        This method modifies the df_movies attribute in place.
        """
        C = self.df_movies['vote_average'].mean()
        m = self.df_movies['vote_count'].quantile(0.9)

        qualified_movies = self.df_movies.copy().loc[self.df_movies['vote_count'] >= m]
        qualified_movies['score'] = qualified_movies.apply(self._calculate_weighted_rating, axis=1, m=m, C=C)
        self.df_movies = qualified_movies.sort_values('score', ascending=False).reset_index(drop=True)

    # def build_cosine_similarity(self, feature: str = 'overview', use_tfidf: bool = True):
    #     """
    #     Build a cosine similarity matrix based on the specified text feature.

    #     Args:
    #         feature (str): Column name to use for similarity calculation (default: 'overview')
    #         use_tfidf (bool): Whether to use TF-IDF vectorization (True) or Count vectorization (False)

    #     The method creates and stores:
    #         - cosine_sim: Similarity matrix
    #         - indices: Mapping of movie titles to matrix indices
    #     """
    #     self.df_movies[feature] = self.df_movies[feature].fillna('')

    #     if use_tfidf:
    #         vectorizer = TfidfVectorizer(stop_words='english')
    #     else:
    #         vectorizer = CountVectorizer(stop_words='english')

    #     feature_matrix = vectorizer.fit_transform(self.df_movies[feature])
    #     self.cosine_sim = linear_kernel(feature_matrix, feature_matrix)
    #     self.indices = pd.Series(self.df_movies.index, index=self.df_movies['title'].str.lower()).drop_duplicates()

    def get_recommendations(self, title: str) -> pd.Series:
        """
        Get movie recommendations based on similarity to the input movie.

        Args:
            title (str): Title of the movie to base recommendations on

        Returns:
            pd.Series: Series of 10 recommended movie titles

        Raises:
            ValueError: If cosine similarity matrix hasn't been built
            KeyError: If the movie title isn't found in the dataset
        """
        if self.cosine_sim is None or self.indices is None:
            logging.error("Cosine similarity matrix not built. Call 'build_content_similarity' first.")
            raise ValueError("Cosine similarity matrix not built.")

        # Normalize title case
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
        Extract the director's name from the crew list.

        Args:
            crew (list): List of crew members and their roles

        Returns:
            str: Director's name or np.nan if no director is found
        """
        for member in crew:
            if member['job'] == 'Director':
                return member['name']
        return np.nan

    def _get_list(self, x: list) -> list:
        """
        Extract up to three names from a list of dictionaries.

        Args:
            x (list): List of dictionaries containing 'name' keys

        Returns:
            list: Up to three names from the input list
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
            Union[list, str]: Cleaned data in the same format as input
        """
        if isinstance(x, list):
            return [str.lower(i.replace(" ", "")) for i in x]
        if isinstance(x, str):
            return str.lower(x.replace(" ", ""))
        return ''

    def enhance_features(self):
        """
        Enhance movie features by:
        1. Converting string representations of lists to actual lists
        2. Extracting director information
        3. Limiting cast, keywords, and genres to top entries
        4. Cleaning and standardizing text data
        5. Creating a consolidated 'soup' of features for similarity calculation

        This method modifies the df_movies attribute in place.
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
            lambda x: ' '.join(x['keywords']) + ' ' + ' '.join(x['cast']) + ' ' + x['director'] + ' ' + ' '.join(x['genres']),
            axis=1
        )

    def build_content_similarity(self):
        """
        Build content-based similarity matrix using the enhanced feature 'soup'.
        This method provides a more comprehensive similarity measure based on
        multiple features (keywords, cast, director, genres) combined.

        The method creates and stores:
            - cosine_sim: Similarity matrix based on the enhanced features
            - indices: Mapping of movie titles to matrix indices
        """
        count = CountVectorizer(stop_words='english')
        count_matrix = count.fit_transform(self.df_movies['soup'])

        self.cosine_sim = cosine_similarity(count_matrix, count_matrix)
        self.indices = pd.Series(self.df_movies.index, index=self.df_movies['title'].str.lower()).drop_duplicates()