#%%
# Imports
from recommenders.hybrid_filtering import HybridRecommender
from utils.paths import ADDITIONAL_DIR
import time



#%%
# Recommender object
start = time.time()

recommender = HybridRecommender(includeMyRatings=False, collaborativeWeight=0.6, titleWeight=2, ratingsWeights=[1.2,1,0.6,-0.7,-2], genresWeights=[2])

end = time.time()
print(f"Exec time: {end - start} s")


#%%
# Choose user
start = time.time()

recommender.choose_user(userId=610)
print("Selected user:", recommender.userId)

end = time.time()
print(f"Exec time: {end - start} s")


#%%
# User ratings
recommender.user_real_ratings()


#%%
# Get recommendations
recommender.get_top_n_recommendations()


#%%
# Predict rating
recommender.get_movie_predicted_rating(movieId=296, printMovie=True)
