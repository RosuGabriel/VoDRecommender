#%%
# Imports
from data import MovieLensData
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import MinMaxScaler
import pandas as pd
import numpy as np

#%%
# Recommender function
def get_recommendations(userId: int, data: MovieLensData, showUserRatings: bool=False, recomandationsNum: int=-1, genresWeight: list=[1], yearWeight: float=1, ratingsWeights: list=[1,1,1], titleWeight: int=1):
    # Get liked movies of a specific user
    userRatings = data.mergedDf[data.mergedDf['userId'] == userId].dropna()
    if showUserRatings:
        print(f"User {userId} ratings:")
        print(userRatings[['movieId', 'title', 'rating', 'year']])

    if len(genresWeight) == 1:
        genresWeight = [genresWeight[0]] * len(data.allGenres)
    elif len(genresWeight) != len(data.allGenres):
        raise ValueError(f"genresWeight must be a list of length {len(data.allGenres)} or a list of one.")
    
    # Features that are multiplied by ratings weights
    mainFeatures = data.allGenres.copy()
    mainFeatures.append('yearScaled')

    # Add importance based on rating
    bestMovies = userRatings[userRatings['rating'] >= 4.5].dropna()
    bestMovies[mainFeatures] *= ratingsWeights[0]
    goodMovies = userRatings[(userRatings['rating'] >= 3.5) & (userRatings['rating'] < 4.5)].dropna()
    goodMovies[mainFeatures] *= ratingsWeights[1]
    mehMovies = userRatings[(userRatings['rating'] >= 2.5) & (userRatings['rating'] < 3.5)].dropna()
    mehMovies[mainFeatures] *= ratingsWeights[2]
    likedMovies = pd.concat([bestMovies, goodMovies, mehMovies], axis=0).drop_duplicates(subset=['movieId'])
    
    # Create user profile by averaging the features of liked movies
    userProfile = likedMovies.drop(['userId', 'movieId', 'title', 'rating', 'year'], axis=1).mean().values
    # Add weights to features
    userProfile[0] *= yearWeight
    userProfile[1:19] = np.array(userProfile[1:19]) * genresWeight
    userProfile[19:] =  np.array(userProfile[19:]) * titleWeight

    # Calculate similarities between user profile and movies not watched by the user
    notWatchedMovies = data.moviesFeatures[~data.moviesFeatures['movieId'].isin(userRatings['movieId'])].dropna() # "~" is the negation operator for a boolean mask
    similarities = cosine_similarity([userProfile], notWatchedMovies.drop(['movieId', 'title', 'year'], axis=1).values)[0]

    betaScaler = MinMaxScaler(feature_range=(0, 1))
    ratingScaler = MinMaxScaler(feature_range=(0.5, 5))
    notWatchedMovies['similarity'] = betaScaler.fit_transform(similarities.reshape(-1, 1))
    notWatchedMovies['predictedRating'] = ratingScaler.fit_transform(notWatchedMovies[['similarity']])

    # Return recommendations sorted
    recommendations = notWatchedMovies.sort_values(by='similarity', ascending=False)[:recomandationsNum]
    
    return recommendations

#%%
# # Usage example
# # Get dataset
# data = MovieLensData(includeMyRatings=True)
# # Get recommendations for user 611
# get_recommendations(612, data, recomandationsNum=20, yearWeight=0.6, titleWeight=4.5, genresWeight=[1], ratingsWeights=[1, 1, 0.6])