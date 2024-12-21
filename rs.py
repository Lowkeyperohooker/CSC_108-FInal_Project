import pandas as pd

from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.metrics.pairwise import linear_kernel, cosine_similarity

# Load datasets
df_credits = pd.read_csv("tmdb-movie-metadata/tmdb_5000_credits.csv")
df_movies = pd.read_csv("tmdb-movie-metadata/tmdb_5000_movies.csv")

print(df_credits.head())
print(df_movies.head())