#%%
# Imports
from utils.paths import BASE_DIR
from data_loader.movie_lens_data import MovieLensData
from recommenders.collaborative_filtering import import_model, get_recommendations as collaborativeRecommendations
from recommenders.content_based_filtering import get_recommendations as contentBasedRecommendations
from recommenders.hybrid_filtering import HybridRecommender
from environment.agent import Agent
from environment.movie_lens_env import MovieLensEnv
import torch



#%%
# Get data
data = MovieLensData(includeMyRatings=True)
env = MovieLensEnv(data = MovieLensData(includeEstimatedRatings=True))

#%%
# Set user and method
userId = 611
method = 'content_based'  # 'collaborative', 'content_based', or 'hybrid'

#%%
# Make recommendations
if method == 'collaborative':
    # Load collaborative filtering model
    model = import_model()
    recommendations = collaborativeRecommendations(data=data, userId=userId, model=model, recomandationsNum=10)
elif method == 'content_based':
    recommendations = contentBasedRecommendations(data=data, userId=userId, recomandationsNum=10, ratingsWeights=[1,0.9,0.7,-0.4,-1])
elif method == 'hybrid':
    recommender = HybridRecommender(data=data, includeMyRatings=False, collaborativeWeight=0.6, titleWeight=2, ratingsWeights=[1.2,1,0.6,-0.7,-2], genresWeights=[2])
    recommender.choose_user(userId=userId)
    recommendations = recommender.get_top_n_recommendations(10)
elif method == 'rl':
    # Load RL agent
    actorModelName = "actor_10May0028.pt"
    pretrainedActorModel = torch.load(BASE_DIR / f"models/pretrained/{actorModelName}")
    criticModelName = "critic_10May0028.pt"
    pretrainedCriticModel = torch.load(BASE_DIR / f"models/pretrained/{criticModelName}")
    agent = Agent(actionsNum=env.action_space.n, observationDim=env.observation_space.shape[0], pretrainedActor=pretrainedActorModel, pretrainedCritic=pretrainedCriticModel)
    agent.load_models('../good models/papc_3_350_0.001_0.99_18-05-2025_00-32')
    # Get user profile and make recommendations
    userProfile = data.get_user_profile_from_csv(611)[1:20]
    recommendations = []
    for i in range(10):
        index = agent.choose_action(observation=userProfile)
        recommendations.append(env.data.moviesDf.iloc[index]['title'])

print(recommendations)

