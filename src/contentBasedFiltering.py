#%%
# Imports
from data import MovieLensData
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import MinMaxScaler
import pandas as pd
import numpy as np



#%%
# Recommender function
def get_user_profile(userId: int, data: MovieLensData, showUserRatings: bool=False, genresWeights: list=[1], yearWeight: float=1, ratingsWeights: list=[1,1,1,1,1], titleWeight: float=1):
    # Get liked movies of a specific user
    userRatings = data.mergedDf[data.mergedDf['userId'] == userId].dropna().copy()
    if showUserRatings:
        print(f"User {userId} ratings:")
        print(userRatings[['movieId', 'title', 'rating', 'year']])
    
    # Verify for correct genres weights
    if len(genresWeights) == 1:
        genresWeights = [genresWeights[0]] * len(data.allGenres)
    elif len(genresWeights) != len(data.allGenres):
        raise ValueError(f"genresWeight must be a list of length {len(data.allGenres)} or a list of one.")
    
    # Verify for correct ratings weights
    if len(ratingsWeights) != 5:
        raise ValueError("ratingsWeights must be a list of length 5.")
    
    # Features that are multiplied by ratings weights
    mainFeatures = data.allGenres.copy()
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
    userProfile = watchedMoviesFeatures.drop(['movieId', 'title', 'rating', 'year'], axis=1).mean().values
    
    # Add weights to features
    userProfile[0] *= yearWeight
    userProfile[1:19] = np.array(userProfile[1:19]) * genresWeights
    userProfile[19:] =  np.array(userProfile[19:]) * titleWeight
    return userProfile


# Get recommendations based on user profile
def get_recommendations(data: MovieLensData, userProfile: list, recomandationsNum: int=-1):
    # Take user watched movies
    userRatings = data.ratingsDf[data.ratingsDf['userId'] == userProfile[0]].dropna()
    
    # Calculate similarities between user profile and movies not watched by the user
    notWatchedMovies = data.moviesFeatures[~data.moviesFeatures['movieId'].isin(userRatings['movieId'])].dropna() # "~" is the negation operator for a boolean mask
    similarities = cosine_similarity([userProfile[1:]], notWatchedMovies.drop(['movieId', 'title', 'year'], axis=1).values)[0] # userProfile[1:] (without the userId)
    
    # Prepare scalers
    betaScaler = MinMaxScaler(feature_range=(0, 1))
    ratingScaler = MinMaxScaler(feature_range=(0.5, 5))
    
    # Scale similarities to [0, 1] and transform them into ratings [0.5, 5]
    notWatchedMovies['similarity'] = betaScaler.fit_transform(similarities.reshape(-1, 1))
    notWatchedMovies['predictedRatingCBF'] = ratingScaler.fit_transform(notWatchedMovies[['similarity']])
    
    # Return recommendations sorted
    return notWatchedMovies.sort_values(by='similarity', ascending=False)[:recomandationsNum]



#%%
# # Usage example
# # Get dataset
# data = MovieLensData(includeMyRatings=True)
# # Get recommendations for user 611
# userProfile = get_user_profile(userId=611, data=data, showUserRatings=True, genresWeights=[1], yearWeight=1, ratingsWeights=[1,0.9,0.7,-0.4,-1], titleWeight=2)
# recommendations = get_recommendations(data, userProfile, recomandationsNum=10)