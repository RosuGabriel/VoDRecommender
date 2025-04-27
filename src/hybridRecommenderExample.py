#%%
# Imports
from hybridFiltering import HybridRecommender
import time



#%% Create recommender
# Recommender object
start = time.time()

recommender = HybridRecommender(includeMyRatings=True)

end = time.time()
print(f"Exec time: {end - start} s")


#%%
# Create recommender
start = time.time()
recommender.initialize_recommender(userId=613, collaborativeWeight=0.6, titleWeight=2, ratingsWeights=[1.2,1,0.6,-0.7,-2], genresWeights=[2])
end = time.time()
print(f"Exec time: {end - start} s")


#%% Usage Examples
# User ratings
recommender.user_real_ratings()

#%%
# Get recommendations
recommender.get_top_n_recommendations(25)

#%%
# Predict rating
recommender.get_movie_predicted_rating(movieId=114713, printMovie=True)

#%%
# Change user
start = time.time()
recommender.change_user(userId=611)
print("User changed to:", recommender.userId)
end = time.time()
print(f"Exec time: {end - start} s")
