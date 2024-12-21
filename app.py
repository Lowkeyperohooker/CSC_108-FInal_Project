from flask import Flask, request, jsonify, render_template
from recommendation_system import MovieRecommender  # Import your MovieRecommender class

app = Flask(__name__)

# Initialize the recommender
recommender = MovieRecommender("tmdb-movie-metadata/tmdb_5000_credits.csv", "tmdb-movie-metadata/tmdb_5000_movies.csv")
recommender.prepare_data()
recommender.enhance_features()
recommender.build_content_similarity()


@app.route('/')
def home():
    """Render the home page."""
    return render_template('index.html')


@app.route('/movies', methods=['GET'])
def get_movies():
    """Return a list of available movies."""
    movies = recommender.show_movies().tolist()
    return render_template('movies.html', movies=movies)


@app.route('/recommend', methods=['POST'])
def recommend():
    """Return movie recommendations for a given title."""
    data = request.json
    title = data.get('title', '').strip()  # Get and sanitize the input

    if not title:
        return jsonify({'status': 'error', 'message': 'Movie title cannot be empty.'}), 400

    try:
        recommendations = recommender.get_recommendations(title).tolist()
        return jsonify({'status': 'success', 'recommendations': recommendations})
    except KeyError:
        return jsonify({'status': 'error', 'message': f'Movie "{title}" not found. Please check the title and try again.'}), 404
    except Exception as e:
        return jsonify({'status': 'error', 'message': f'An unexpected error occurred: {str(e)}'}), 500



if __name__ == '__main__':
    app.run(debug=True)
