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
import pandas as pd



# Recommender class
class Recommender:
    data = MovieLensData(includeMyRatings=True)
    def __init__(self, method='content_based', userId=1):
        self.method = method
        self.userId = userId
        self.env = MovieLensEnv(data = MovieLensData(includeEstimatedRatings=True))

        # Load collaborative filtering model
        self.svdModel = import_model()
        
        # Load hybrid recommender
        self.hybridRecommender = HybridRecommender(data=self.data, includeMyRatings=False, collaborativeWeight=0.6, titleWeight=2, ratingsWeights=[1.2,1,0.6,-0.7,-2], genresWeights=[2])
        self.hybridRecommender.choose_user(userId=userId)
        
        # Load RL agent
        actorModelName = "actor_10May0028.pt"
        pretrainedActorModel = torch.load(BASE_DIR / f"models/pretrained/{actorModelName}")
        criticModelName = "critic_10May0028.pt"
        pretrainedCriticModel = torch.load(BASE_DIR / f"models/pretrained/{criticModelName}")
        self.agent = Agent(actionsNum=self.env.action_space.n, observationDim=self.env.observation_space.shape[0], pretrainedActor=pretrainedActorModel, pretrainedCritic=pretrainedCriticModel)
        self.agent.load_models('../good models/3_24440_0.00015_0.99_11-06-2025_00-29')


    def reset(self, userId, method='content_based'):
        self.userId = userId
        self.method = method


    def get_recommendations(self, recommendationsNum=12):
        if self.method == 'Collaborative':
            recommendations = collaborativeRecommendations(data=self.data, userId=self.userId, model=self.svdModel, recomandationsNum=recommendationsNum)
        
        elif self.method == 'Content-Based':
            recommendations = contentBasedRecommendations(data=self.data, userId=self.userId, recomandationsNum=recommendationsNum, ratingsWeights=[1,0.9,0.7,-0.4,-1])
        
        elif self.method == 'Hybrid':
            self.hybridRecommender.choose_user(userId=self.userId)
            recommendations = self.hybridRecommender.get_top_n_recommendations(recommendationsNum)
        
        elif self.method == 'RL':
            userProfile = self.data.calculate_user_profile(userId=self.userId)[1:20]
            _, indexes = self.agent.choose_action(observation=userProfile, actionsNum=recommendationsNum)
            rows = [self.env.data.moviesFeatures.iloc[index] for index in indexes]
            recommendations = pd.DataFrame(rows, columns=self.env.data.moviesFeatures.columns)

        else:
            raise ValueError("Invalid method specified. Choose from 'Collaborative', 'Content-Based','Hybrid' or 'RL'.")
        
        return recommendations
