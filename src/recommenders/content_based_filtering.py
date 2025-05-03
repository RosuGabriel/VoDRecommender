#%%
# Imports
from data_loader.data_movie_lens import MovieLensData
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import MinMaxScaler



#%%
# Get recommendations based on user profile
def get_recommendations(data: MovieLensData, userId: int=1, userProfileFromCSV: str=None, recomandationsNum: int=-1, showUserRatings: bool=False, genresWeights: list=[1], yearWeight: float=1, ratingsWeights: list=[1,1,1,1,1], titleWeight: float=1):
    # Get user profile
    if userProfileFromCSV is not None:
        userProfile = data.get_user_profile_from_csv(userId=userId)
    else:
        userProfile = data.calculate_user_profile(userId=userId, showUserRatings=showUserRatings, genresWeights=genresWeights, yearWeight=yearWeight, ratingsWeights=ratingsWeights, titleWeight=titleWeight)
    
     # Take user watched movies
    userRatings = data.user_ratings(userId=userId)
    
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



# #%%
# # Usage example
# # Get dataset
# data = MovieLensData(includeMyRatings=True)
# # Get recommendations for user 611
# recommendations = get_recommendations(data, userId=611, showUserRatings=True, genresWeights=[1], yearWeight=1, ratingsWeights=[1,0.9,0.7,-0.4,-1], titleWeight=2, recomandationsNum=10)
# recommendations