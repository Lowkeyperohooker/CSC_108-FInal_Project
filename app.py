from flask import Flask, jsonify, request
from movie_recommender import MovieRecommender

app = Flask(__name__)

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
    movies = recommender.show_movies().tolist()
    return jsonify({"movies": movies})

@app.route('/recommend', methods=['POST'])
def recommend():
    try:
        data = request.json
        title = data.get('title', '')
        recommendations = recommender.get_recommendations(title)
        return jsonify({"recommendations": recommendations.tolist()})
    except KeyError:
        return jsonify({"error": "Movie not found. Check the title and try again."}), 400

if __name__ == '__main__':
    app.run(debug=True)
