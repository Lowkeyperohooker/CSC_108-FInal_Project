from flask import Flask, request, jsonify, render_template
from recommendation_system import MovieRecommender  # Import your MovieRecommender class
from flask_caching import Cache
import os
import pickle
import logging

# Initialize Flask app and cache
app = Flask(__name__)
cache = Cache(app, config={'CACHE_TYPE': 'SimpleCache'})

# Logging setup
logging.basicConfig(level=logging.INFO)

# Initialize or load the precomputed recommender
if not os.path.exists('precomputed_data.pkl'):
    logging.info("Precomputing data...")
    recommender = MovieRecommender("tmdb-movie-metadata/tmdb_5000_credits.csv", 
                                   "tmdb-movie-metadata/tmdb_5000_movies.csv")
    recommender.prepare_data()
    recommender.enhance_features()
    recommender.build_content_similarity()
    with open('precomputed_data.pkl', 'wb') as f:
        pickle.dump(recommender, f)
    logging.info("Data precomputed and saved to 'precomputed_data.pkl'.")
else:
    logging.info("Loading precomputed data from 'precomputed_data.pkl'...")
    with open('precomputed_data.pkl', 'rb') as f:
        recommender = pickle.load(f)
    logging.info("Precomputed data loaded successfully.")


@app.route('/')
def home():
    """Render the home page."""
    return render_template('index.html')


@app.route('/movies', methods=['GET'])
@cache.cached(timeout=300)  # Cache the response for 5 minutes
def get_movies():
    """Return a list of available movies."""
    try:
        movies = recommender.show_movies().tolist()
        return render_template('movies.html', movies=movies)
    except Exception as e:
        logging.error(f"Error fetching movies: {str(e)}")
        return jsonify({'status': 'error', 'message': 'Failed to fetch movies.'}), 500


@app.route('/recommend', methods=['POST'])
def recommend():
    """Return movie recommendations for a given title."""
    data = request.json
    title = data.get('title', '').strip()

    if not title:
        return jsonify({'status': 'error', 'message': 'Movie title cannot be empty.'}), 400

    try:
        recommendations = recommender.get_recommendations(title).tolist()
        return jsonify({'status': 'success', 'recommendations': recommendations})
    except KeyError:
        # Suggest similar titles if the movie is not found
        similar_titles = recommender.get_similar_titles(title)
        return jsonify({
            'status': 'error',
            'message': f'Movie "{title}" not found. Did you mean one of these?',
            'suggestions': similar_titles
        }), 404
    except Exception as e:
        logging.error(f"Unexpected error during recommendation: {str(e)}")
        return jsonify({'status': 'error', 'message': 'An unexpected error occurred.'}), 500


if __name__ == '__main__':
    PORT = int(os.environ.get("PORT", 10000))  # Render provides the PORT env
