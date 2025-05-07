#%%
# Imports
import pandas as pd
import re
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.feature_extraction.text import TfidfVectorizer
from utils.paths import MOVIE_LENS_DIR, ADDITIONAL_DIR


# Constants
ALL_GENRES = ["Action",
            "Adventure",
            "Animation",
            "Children",
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
    match = re.match(r"^(.*)\s\((\d{4})\)\s*$", title)
    if match:
        name, year = match.groups()
        return pd.Series([name, int(year)])
    else:
        return pd.Series([title, ])  # fallback

def genres_to_vector(genres):
    return [1.0 if g in genres else 0.0 for g in ALL_GENRES]



# Data class for easier manipulation and recovery - this is the one intended to be imported
class MovieLensData:
    def __init__(self, includeMyRatings: bool = False, includeEstimatedRatings: bool = False):
        self.ratingsDf = pd.read_csv(MOVIE_LENS_DIR / "ratings.csv")   # userId, movieId, rating, timestamp

        if includeEstimatedRatings:
            estimatedRatings = pd.read_csv(ADDITIONAL_DIR / "hybridPredictedRatings.csv") # userId, movieId, rating
            self.ratingsDf = pd.concat([self.ratingsDf, estimatedRatings], axis=0, ignore_index=True)
            
        self.ratingsDf.drop(['timestamp'], axis=1, inplace=True)

        self.moviesDf = pd.read_csv(MOVIE_LENS_DIR / "movies.csv")     # movieId, title (year), genres
        self.moviesDf = self.moviesDf[self.moviesDf['movieId'].isin(self.ratingsDf['movieId'])].dropna().reset_index(drop=True) # remove movies not rated by any user

        self.calculate_movie_features()
        self.allGenres = ALL_GENRES

        if not includeEstimatedRatings:
            if includeMyRatings:
                self.myRatings = self.add_my_ratings()
            else:
                self.calculate_utility_matrix()


    def add_my_ratings(self):
        myRatings = pd.read_csv(ADDITIONAL_DIR / "myRatings.csv")
        self.add_ratings(myRatings)
        return myRatings


    def calculate_movie_features(self):
        # Splits the title and year from the title column in moviesDf
        self.moviesDf[['title', 'year']] = self.moviesDf['title'].apply(split_title_year)

        # Create movie feature vectors (movieId, title, year, yearScaled, hot vector genres, title features)
        genresDf = self.moviesDf[['genres']].copy()
        genresDf['genres'] = genresDf['genres'].apply(genres_to_vector)
        genresDf = pd.DataFrame(genresDf['genres'].tolist())
        moviesFeatures = pd.concat([self.moviesDf['movieId'], self.moviesDf['title'], self.moviesDf['year'], genresDf], axis=1)

        moviesFeatures.columns = ['movieId', 'title', 'year'] + ALL_GENRES
        
        # Normalize years
        scaler = StandardScaler()
        yearIndex = moviesFeatures.columns.get_loc('year')
        moviesFeatures.insert(yearIndex + 1, 'yearScaled', scaler.fit_transform(moviesFeatures[['year']]))

        # Exract title features
        tfidf = TfidfVectorizer(stop_words='english')
        titleFeatures = tfidf.fit_transform(self.moviesDf['title'])

        titleFeatures = pd.DataFrame(titleFeatures.toarray(), columns=tfidf.get_feature_names_out())
        titleFeatures.columns = [ col + ' in title' for col in tfidf.get_feature_names_out()]
        moviesFeatures = pd.concat([moviesFeatures.reset_index(drop=True), titleFeatures.reset_index(drop=True)], axis=1)
        moviesFeatures.loc[moviesFeatures['yearScaled'].isna(), 'yearScaled'] = 0
        
        self.moviesFeatures = moviesFeatures


    def calculate_user_profile(self, userId: int, showUserRatings: bool=False, genresWeights: list=[1], yearWeight: float=1, ratingsWeights: list=[1.3, 1, 0.7, -1, -2], titleWeight: float=1):
        # Get liked movies of a specific user
        userRatings = self.user_ratings_with_movie_features(userId=userId)

        if showUserRatings:
            print(f"User {userId} ratings:")
            print(userRatings[['movieId', 'title', 'rating', 'year']])
        
        # Verify for correct genres weights
        if len(genresWeights) == 1:
            genresWeights = [genresWeights[0]] * len(self.allGenres)
        elif len(genresWeights) != len(self.allGenres):
            raise ValueError(f"genresWeight must be a list of length {len(self.allGenres)} or a list of one.")
        
        # Verify for correct ratings weights
        if len(ratingsWeights) != 5:
            raise ValueError("ratingsWeights must be a list of length 5.")
        
        # Features that are multiplied by ratings weights
        mainFeatures = self.allGenres.copy()
        mainFeatures.append('yearScaled')
        
        # Add importance based on rating
        bestMovies = userRatings[userRatings['rating'] >= 4.5].dropna()
        bestMovies[mainFeatures] *= ratingsWeights[0]
        goodMovies = userRatings[(userRatings['rating'] >= 4) & (userRatings['rating'] < 4.5)].dropna()
        goodMovies[mainFeatures] *= ratingsWeights[1]
        mehMovies = userRatings[(userRatings['rating'] >= 3) & (userRatings['rating'] < 4)].dropna()
        mehMovies[mainFeatures] *= ratingsWeights[2]
        dislikedMovies = userRatings[(userRatings['rating'] >= 2) & (userRatings['rating'] < 3)].dropna()
        dislikedMovies[mainFeatures] *= ratingsWeights[3]
        hatedMovies = userRatings[(userRatings['rating'] < 2)].dropna()
        hatedMovies[mainFeatures] *= ratingsWeights[4]
        
        # Concatenate all watched movies
        watchedMoviesFeatures = pd.concat([bestMovies, goodMovies, mehMovies, dislikedMovies, hatedMovies], axis=0)
        
        # Create user profile by averaging the features of liked movies
        # userId yearScaled genres titleFeatures
        userProfile = watchedMoviesFeatures.drop(['movieId', 'title', 'rating', 'year'], axis=1).mean().values
        
        # Add weights to features
        userProfile[1] *= yearWeight
        userProfile[2:20] = np.array(userProfile[2:20]) * genresWeights

        if not titleWeight:
            return userProfile[:20]

        userProfile[20:] =  np.array(userProfile[20:]) * titleWeight

        return userProfile
    

    def get_movie_features(self, movieId: int):
        filteredMovies = self.moviesFeatures[self.moviesFeatures['movieId'] == movieId].drop(['movieId', 'title', 'year'], axis=1)
        if filteredMovies.empty:
            return None
        return filteredMovies.values[0]


    def calculate_all_user_profiles(self, genresWeights: list=[1], yearWeight: float=1, ratingsWeights: list=[1.3, 1, 0.7, -1, -2], titleWeight: float=1, saveToCSV: bool=False):
        users = self.ratingsDf['userId'].unique()
        userProfiles = []

        for userId in users:
            userProfile = self.calculate_user_profile(userId=userId, genresWeights=genresWeights, yearWeight=yearWeight, ratingsWeights=ratingsWeights, titleWeight=titleWeight)
            userProfiles.append(userProfile)
        userProfilesDf = pd.DataFrame(userProfiles, columns=['userId'] + ['yearScaled'] + self.moviesFeatures.columns[4:].tolist())
        
        if saveToCSV:
            userProfilesDf.to_csv(ADDITIONAL_DIR / 'userProfiles.csv', index=False)
        
        return userProfilesDf


    def get_user_profile_from_csv(self, userId: int):
        userProfilesDf = pd.read_csv(ADDITIONAL_DIR / 'userProfiles.csv')
        userProfile = userProfilesDf[userProfilesDf['userId'] == userId].values[0]
        
        return userProfile


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


    @property
    def data_sparsity(self):
        return (1.0 - (len(self.ratingsDf) / (len(self.utilityMatrix) * len(self.utilityMatrix.columns))))*100.0


    def remove_movies_with_less_than_k_ratings(self, k: int):
        filtered_movies = self.ratingsDf['movieId'].value_counts()
        movies_to_keep = filtered_movies[filtered_movies >= k].index
        self.ratingsDf = self.ratingsDf[self.ratingsDf['movieId'].isin(movies_to_keep)]
        self.moviesDf = self.moviesDf[self.moviesDf['movieId'].isin(movies_to_keep)]
        self.calculate_utility_matrix()


    def user_ratings_with_movie_features(self, userId: int):
        userRatings = self.user_ratings(userId=userId).copy()
        return pd.merge(userRatings, self.moviesFeatures, on='movieId', how='inner').dropna()


    def user_ratings(self, userId: int):
        return self.ratingsDf[self.ratingsDf['userId'] == userId]


    def movie_ratings(self, movieId: int):
        return self.ratingsDf[self.ratingsDf['movieId'] == movieId]
    

    @property
    def all_users(self):
        return self.ratingsDf['userId'].unique()


    def add_ratings(self, newRatingsDf: pd.DataFrame):
        self.ratingsDf = pd.concat([self.ratingsDf, newRatingsDf], ignore_index=True)
        self.ratingsDf = self.ratingsDf.drop_duplicates(subset=['userId', 'movieId'], keep='first') # remove duplicates
        self.calculate_utility_matrix()
