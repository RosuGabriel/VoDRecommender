import pandas as pd

# File importing
# Movie Lens 2018 dataset
movieLensPath = "../../Datasets/movie lens 2018/ml-latest-small/"
ratings_df = pd.read_csv(movieLensPath + "ratings.csv")   # userId, movieId, rating, timestamp
movies_df = pd.read_csv(movieLensPath + "movies.csv")     # movieId, title, genres

# Variables
moviesIDs = movies_df['movieId']

allGenres = ["Action",
            "Adventure",
            "Animation",
            "Children's",
            "Comedy",
            "Crime",
            "Documentary",
            "Drama",
            "Fantasy",
            "Film-Noir",
            "Horror",
            "Musical",
            "Mystery",
            "Romance",
            "Sci-Fi",
            "Thriller",
            "War",
            "Western",
            "(no genres listed)"]

# Funtions
def genres_to_vector(genres):
    return [1.0 if g in genres else 0.0 for g in allGenres]

def user_rating_of_movie(userId: int, movieId: int):
    filteredRatings = ratings_df[(ratings_df['userId'] == userId) & (ratings_df['movieId'] == movieId)]
    if filteredRatings.empty:
        return 0
    return filteredRatings['rating'].values[0]

def genres_of_movie(movieId: int):
    filteredMovies = movies_df[movies_df['movieId'] == movieId]
    if filteredMovies.empty:
        return []
    return filteredMovies['genres'].values[0].split('|')
