# Imports
import pandas as pd
import re

# File importing - Movie Lens 2018 dataset
movieLensPath = "../../../Datasets/movie lens 2018/ml-latest-small/"
ratingsDf = pd.read_csv(movieLensPath + "ratings.csv")   # userId, movieId (year), rating, timestamp
moviesDf = pd.read_csv(movieLensPath + "movies.csv")     # movieId, title, genres

# Variables
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
def split_title_year(title):
    match = re.match(r"^(.*) \((\d{4})\)$", title)
    if match:
        name, year = match.groups()
        return pd.Series([name, int(year)])
    else:
        return pd.Series([title, None])  # fallback

def genres_to_vector(genres):
    return [1.0 if g in genres else 0.0 for g in allGenres]

# Splits the title and year from the title column in moviesDf
moviesDf[['title', 'year']] = moviesDf['title'].apply(split_title_year)

# Create movie feature vectors (movieId, year, hot vector genres)
moviesFeatures = moviesDf[['movieId', 'genres', 'year']].copy()
moviesFeatures['genres'] = moviesFeatures['genres'].apply(genres_to_vector)
genresDf = pd.DataFrame(moviesFeatures['genres'].tolist())
moviesFeatures = pd.concat([moviesFeatures['movieId'],  moviesFeatures['year'], genresDf], axis=1)

# Data class for easier manipulation and recovery - this is the one intended to be imported
class Data:
    def __init__(self):
        self.ratingsDf = ratingsDf
        self.moviesDf = moviesDf
        self.allGenres = allGenres
        self.utilityMatrix = self.calculate_utility_matrix()

    def user_rating_of_movie(self, userId: int, movieId: int):
        filteredRatings = self.ratingsDf[(self.ratingsDf['userId'] == userId) & (self.ratingsDf['movieId'] == movieId)]
        if filteredRatings.empty:
            return None
        return filteredRatings['rating'].values[0]

    def genres_of_movie(self, movieId: int):
        filteredMovies = self.moviesDf[self.moviesDf['movieId'] == movieId]
        if filteredMovies.empty:
            return []
        return filteredMovies['genres'].values[0].split('|')

    def title_of_movie(self, movieId: int):
        filteredMovies = self.moviesDf[self.moviesDf['movieId'] == movieId]
        if filteredMovies.empty:
            return []
        return filteredMovies['title'].values[0]
    
    def calculate_utility_matrix(self):
        return self.ratingsDf.pivot_table(index='userId', columns='movieId', values='rating')

    def data_sparsity(self):
        return (1.0 - (len(self.ratingsDf) / (len(self.utilityMatrix) * len(self.utilityMatrix.columns))))*100.0
    
    def reset_data(self):
        self.ratingsDf = ratingsDf
        self.moviesDf = moviesDf
        self.allGenres = allGenres

    def remove_movies_with_less_than_k_ratings(self, k: int):
        filtered_movies = self.ratingsDf['movieId'].value_counts()
        movies_to_keep = filtered_movies[filtered_movies >= k].index
        self.ratingsDf = self.ratingsDf[self.ratingsDf['movieId'].isin(movies_to_keep)]
        self.moviesDf = self.moviesDf[self.moviesDf['movieId'].isin(movies_to_keep)]
        self.utilityMatrix = self.calculate_utility_matrix()

    def user_ratings(self, userId: int):
        filteredRatings = self.ratingsDf[self.ratingsDf['userId'] == userId]
        if filteredRatings.empty:
            return []
        return filteredRatings[['movieId', 'rating']].values.tolist()
    
    def movie_ratings(self, movieId: int):
        filteredRatings = self.ratingsDf[self.ratingsDf['movieId'] == movieId]
        if filteredRatings.empty:
            return []
        return filteredRatings[['userId', 'rating']].values.tolist()
