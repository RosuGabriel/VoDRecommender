#%%
# Imports
import pandas as pd
import re
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.feature_extraction.text import TfidfVectorizer



# Constants
MOVIE_LENS_PATH = "ml-latest-small/"
ALL_GENRES = ["Action",
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
            "Western"]


# Functions
def split_title_year(title):
    match = re.match(r"^(.*) \((\d{4})\)$", title)
    if match:
        name, year = match.groups()
        return pd.Series([name, int(year)])
    else:
        return pd.Series([title, None])  # fallback

def genres_to_vector(genres):
    return [1.0 if g in genres else 0.0 for g in ALL_GENRES]



# Data class for easier manipulation and recovery - this is the one intended to be imported
class MovieLensData:
    def __init__(self, includeMyRatings: bool = False):
        self.ratingsDf, self.moviesDf, self.moviesFeatures = self.initialize_data()
        self.allGenres = ALL_GENRES
        if includeMyRatings:
            self.myRatings = self.add_my_ratings()
        else:
            self.mergeRatingsWithFeatures()
            self.calculate_utility_matrix()


    def reset_data(self, includeMyRatings: bool = False):
        self.__init__(includeMyRatings)


    def add_my_ratings(self):
        myRatings = pd.read_csv(MOVIE_LENS_PATH + "my ratings.csv")
        self.add_ratings(myRatings)
        return myRatings


    def initialize_data(self):
        ratingsDf = pd.read_csv(MOVIE_LENS_PATH + "ratings.csv")   # userId, movieId (year), rating, timestamp
        moviesDf = pd.read_csv(MOVIE_LENS_PATH + "movies.csv")     # movieId, title, genres
        
        # Splits the title and year from the title column in moviesDf
        moviesDf[['title', 'year']] = moviesDf['title'].apply(split_title_year)

        # Create movie feature vectors (movieId, year, hot vector genres)
        genresDf = moviesDf[['genres']].copy()
        genresDf['genres'] = genresDf['genres'].apply(genres_to_vector)
        genresDf = pd.DataFrame(genresDf['genres'].tolist())
        moviesFeatures = pd.concat([moviesDf['movieId'], moviesDf['title'], moviesDf['year'], genresDf], axis=1)

        moviesFeatures.columns = ['movieId', 'title', 'year'] + ALL_GENRES
        # Normalize years
        scaler = StandardScaler()
        yearIndex = moviesFeatures.columns.get_loc('year')
        moviesFeatures.insert(yearIndex + 1, 'yearScaled', scaler.fit_transform(moviesFeatures[['year']]))

        # Exract title features
        tfidf = TfidfVectorizer(stop_words='english')
        titleFeatures = tfidf.fit_transform(moviesDf['title'])

        titleFeatures = pd.DataFrame(titleFeatures.toarray(), columns=tfidf.get_feature_names_out())
        titleFeatures.columns = [ col + ' in title' for col in tfidf.get_feature_names_out()]
        moviesFeatures = pd.concat([moviesFeatures.reset_index(drop=True), titleFeatures.reset_index(drop=True)], axis=1)

        return ratingsDf, moviesDf, moviesFeatures


    def mergeRatingsWithFeatures(self):
        self.mergedDf = pd.merge(self.ratingsDf, self.moviesFeatures, on='movieId', how='inner')
        self.mergedDf.drop(columns=['timestamp'], inplace=True)


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
        self.utilityMatrix = self.ratingsDf.pivot_table(index='userId', columns='movieId', values='rating')


    def data_sparsity(self):
        return (1.0 - (len(self.ratingsDf) / (len(self.utilityMatrix) * len(self.utilityMatrix.columns))))*100.0


    def remove_movies_with_less_than_k_ratings(self, k: int):
        filtered_movies = self.ratingsDf['movieId'].value_counts()
        movies_to_keep = filtered_movies[filtered_movies >= k].index
        self.ratingsDf = self.ratingsDf[self.ratingsDf['movieId'].isin(movies_to_keep)]
        self.moviesDf = self.moviesDf[self.moviesDf['movieId'].isin(movies_to_keep)]
        self.mergedDf = self.mergedDf[self.mergedDf['movieId'].isin(movies_to_keep)]
        self.calculate_utility_matrix()


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
    

    def add_ratings(self, newRatingsDf: pd.DataFrame):
        self.ratingsDf = pd.concat([self.ratingsDf, newRatingsDf], ignore_index=True)
        self.ratingsDf = self.ratingsDf.drop_duplicates(subset=['userId', 'movieId'], keep='first') # remove duplicates
        self.mergeRatingsWithFeatures()
        self.calculate_utility_matrix()
