import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity, linear_kernel
from ast import literal_eval
import logging

# Configure logging
logging.basicConfig(level=logging.DEBUG, format='%(levelname)s: %(message)s')

# Set pandas options for debugging
pd.set_option('display.max_columns', None)  # Show all columns

class MovieRecommender:
    def __init__(self, credits_path, movies_path):
        self.df_movies = self._load_and_merge_data(credits_path, movies_path)
        self.cosine_sim = None
        self.indices = None

    def _load_and_merge_data(self, credits_path, movies_path):
        df_credits = pd.read_csv(credits_path)
        df_movies = pd.read_csv(movies_path)

        df_credits.columns = ['id', 'tittle', 'cast', 'crew']
        df_movies = df_movies.merge(df_credits, on='id')

        return df_movies

    def show_movies(self):
        return self.df_movies['title']

    def _calculate_weighted_rating(self, x, m, C):
        v = x['vote_count']
        R = x['vote_average']
        return (v / (v + m) * R) + (m / (m + v) * C)

    def prepare_data(self):
        C = self.df_movies['vote_average'].mean()
        m = self.df_movies['vote_count'].quantile(0.9)

        qualified_movies = self.df_movies.copy().loc[self.df_movies['vote_count'] >= m]
        qualified_movies['score'] = qualified_movies.apply(self._calculate_weighted_rating, axis=1, m=m, C=C)
        self.df_movies = qualified_movies.sort_values('score', ascending=False).reset_index(drop=True)

    def build_cosine_similarity(self, feature='overview', use_tfidf=True):
        self.df_movies[feature] = self.df_movies[feature].fillna('')

        if use_tfidf:
            vectorizer = TfidfVectorizer(stop_words='english')
        else:
            vectorizer = CountVectorizer(stop_words='english')

        feature_matrix = vectorizer.fit_transform(self.df_movies[feature])
        self.cosine_sim = linear_kernel(feature_matrix, feature_matrix)
        self.indices = pd.Series(self.df_movies.index, index=self.df_movies['title'].str.lower()).drop_duplicates()

    def get_recommendations(self, title):
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

    def _get_director(self, crew):
        for member in crew:
            if member['job'] == 'Director':
                return member['name']
        return np.nan

    def _get_list(self, x):
        if isinstance(x, list):
            names = [i['name'] for i in x]
            return names[:3] if len(names) > 3 else names
        return []

    def _clean_data(self, x):
        if isinstance(x, list):
            return [str.lower(i.replace(" ", "")) for i in x]
        if isinstance(x, str):
            return str.lower(x.replace(" ", ""))
        return ''

    def enhance_features(self):
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
        count = CountVectorizer(stop_words='english')
        count_matrix = count.fit_transform(self.df_movies['soup'])

        self.cosine_sim = cosine_similarity(count_matrix, count_matrix)
        self.indices = pd.Series(self.df_movies.index, index=self.df_movies['title'].str.lower()).drop_duplicates()