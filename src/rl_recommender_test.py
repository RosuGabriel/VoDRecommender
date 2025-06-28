#%%
# Imports
from recommender import Recommender
import numpy as np



#%%
# Initialize recommender and rating averages list
recommender = Recommender()
ratingAverages = []


#%%
# Get recommendations for each user
for userId in range(1,611,10):
    recommender.reset(userId=userId, method='RL')

    recommendations = recommender.get_recommendations(recommendationsNum=10)

    ratings = [recommender.env.data.user_rating_of_movie(userId=userId, movieId=movieId) for movieId in recommendations['movieId'].values]
    ratings = [rating for rating in ratings if rating is not None]
    ratingAverages.append(np.mean(ratings))


# Show average performance
print(f"Average rating of recommendations: {np.mean(ratingAverages):.2f}")
