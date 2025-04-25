#%%
# Imports
from data import MovieLensData
from contentBasedFiltering import get_recommendations as content_based_recommender
from collaborativeFiltering import model, train_recommender, get_recommendations as collaborative_recommender
import pandas as pd

#%%
# Prepare data
data = MovieLensData(includeMyRatings=True)
trainedModel = train_recommender(data, model)

#%%
# Set the user that will get recommendations
userId = 612

#%%
# Get collaborative recommendations
cfRecommendations = collaborative_recommender(userId, data, trainedModel, recomandationsNum=-1)[['movieId', 'title', 'year', 'predictedRating']]
cfRecommendations

#%%
# Get content based recommendations
cbfRecommendations = content_based_recommender(userId, data, ratingsWeights=[1,0.8,0.4], titleWeight=3)[['movieId', 'title', 'year', 'predictedRating', 'similarity']]
cbfRecommendations.rename(columns={'predictedRating': 'predictedRatingCB'}, inplace=True)
cbfRecommendations

#%%
# Make hybrid recommendations
hybridRecommendations = pd.merge(cbfRecommendations, cfRecommendations, on=['movieId','title','year'], how='inner')

collaborativeWeight = 0.7
contentBasedWeight = 1 - collaborativeWeight

hybridRecommendations['hybridRating'] = hybridRecommendations['predictedRating']*collaborativeWeight + hybridRecommendations['predictedRatingCB']*contentBasedWeight
hybridRecommendations.sort_values(by='hybridRating', ascending=False, inplace=True)
hybridRecommendations[:20]

#%%
# Prediction example
movieId = 4369
print(data.title_of_movie(movieId))
hybridRecommendations[hybridRecommendations['movieId'] == movieId][['predictedRating','predictedRatingCB','hybridRating']]
