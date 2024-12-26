from recommendation_system import MovieRecommender 
import os
import pickle
import logging

class Data:
    def __init__(self):
        self.recommender = self._load_data()

    def _load_data(self):
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
        
        return recommender
    
    def get_data(self):
        return self.recommender



