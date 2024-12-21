from flask import Flask, jsonify, request
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.metrics.pairwise import linear_kernel, cosine_similarity
from ast import literal_eval

app = Flask(__name__)

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
    
    def prepare_data(self):
        C = self.df_movies['vote_average'].mean()
        m = self.df_movies['vote_count'].quantile(0.9)
        qualified_movies = self.df_movies.copy().loc[self.df_movies['vote_count'] >= m]
        qualified_movies['score'] = qualified_movies.apply(
            lambda x: (x['vote_count'] / (x['vote_count'] + m) * x['vote_average']) +
                      (m / (m + x['vote_count']) * C),
            axis=1
        )
        self.df_movies = qualified_movies.sort_values('score', ascending=False)

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
            lambda x: ' '.join(x['keywords']) + ' ' + ' '.join(x['cast']) +
                      ' ' + x['director'] + ' ' + ' '.join(x['genres']),
            axis=1
        )

    def build_content_similarity(self):
        count = CountVectorizer(stop_words='english')
        count_matrix = count.fit_transform(self.df_movies['soup'])
        self.cosine_sim = cosine_similarity(count_matrix, count_matrix)
        self.indices = pd.Series(self.df_movies.index, index=self.df_movies['title']).drop_duplicates()

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

    def get_recommendations(self, title):
        if self.cosine_sim is None or self.indices is None:
            raise ValueError("Cosine similarity matrix not built.")
        idx = self.indices[title]
        sim_scores = list(enumerate(self.cosine_sim[idx]))
        sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)[1:11]
        movie_indices = [i[0] for i in sim_scores]
        return self.df_movies['title'].iloc[movie_indices].tolist()

# Initialize the recommender
recommender = MovieRecommender("tmdb-movie-metadata/tmdb_5000_credits.csv", "tmdb-movie-metadata/tmdb_5000_movies.csv")
recommender.prepare_data()
recommender.enhance_features()
recommender.build_content_similarity()

@app.route('/')
def home():
    return "Welcome to the Movie Recommendation API! Use /movies to list movies or /recommend to get recommendations."

@app.route('/movies', methods=['GET'])
def show_movies():
    movies = recommender.df_movies['title'].tolist()
    return jsonify({"movies": movies})

@app.route('/recommend', methods=['POST'])
def recommend():
    data = request.json
    title = data.get('title', '')
    if title not in recommender.indices:
        return jsonify({"error": "Movie not found. Check the title and try again."}), 400
    recommendations = recommender.get_recommendations(title)
    return jsonify({"recommendations": recommendations})

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)
