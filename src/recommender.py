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
        self.env = MovieLensEnv(data=MovieLensData(includeEstimatedRatings=True,includeMyRatings=True))

        # Load collaborative filtering model
        self.svdModel = import_model(modelName='02-07-2025-619users')
        
        # Load hybrid recommender
        self.hybridRecommender = HybridRecommender(data=Recommender.data, collaborativeWeight=0.6, titleWeight=2, ratingsWeights=[1.2,1,0.6,-0.7,-2], genresWeights=[2])
        self.hybridRecommender.choose_user(userId=userId)
        
        # Load RL agent
        actorModelName = "actor_10May0028.pt"
        pretrainedActorModel = torch.load(BASE_DIR / f"models/pretrained/{actorModelName}")
        criticModelName = "critic_10May0028.pt"
        pretrainedCriticModel = torch.load(BASE_DIR / f"models/pretrained/{criticModelName}")
        self.agent1 = Agent(beta = 1, actionsNum=self.env.action_space.n, observationDim=self.env.observation_space.shape[0], pretrainedActor=pretrainedActorModel, pretrainedCritic=pretrainedCriticModel)
        #self.agent2 = Agent(beta = 1, actionsNum=self.env.action_space.n, observationDim=self.env.observation_space.shape[0], pretrainedActor=pretrainedActorModel, pretrainedCritic=pretrainedCriticModel)
        self.agent1.load_models('../models checkpoint/3_61100_0.00075_0.99_30-06-2025_19-44')
        #self.agent1.load_models('../models checkpoint/3_61000_0.0009_0.0001_0.99_03-07-2025_23-41')
        #self.agent2.load_models('../models checkpoint/3_61000_0.00075_0.0001_0.99_03-07-2025_14-55')


    def reset(self, userId, method='content_based'):
        self.userId = userId
        self.method = method


    def get_recommendations(self, recommendationsNum=12):
        if self.method == 'Collaborative':
            recommendations = collaborativeRecommendations(data=Recommender.data, userId=self.userId, model=self.svdModel, recomandationsNum=recommendationsNum)
        
        elif self.method == 'Content-Based':
            recommendations = contentBasedRecommendations(data=Recommender.data, userId=self.userId, recomandationsNum=recommendationsNum, ratingsWeights=[1,0.9,0.7,-0.4,-1])
        
        elif self.method == 'Hybrid':
            self.hybridRecommender.choose_user(userId=self.userId)
            recommendations = self.hybridRecommender.get_top_n_recommendations(recommendationsNum)
        
        elif self.method == 'RL':
            userProfile = Recommender.data.calculate_user_profile(userId=self.userId)[1:20]
            _, probs1 = self.agent1.choose_action(observation=userProfile, temperature=0.5)
            probs1 = torch.exp(probs1)
            _, probs2 = self.agent1.choose_action(observation=userProfile)
            indexes = torch.topk(probs1+probs2-0.2*abs(probs1-probs2), k=recommendationsNum).indices.tolist()
            #indexes = torch.multinomial(probs1, num_samples=recommendationsNum, replacement=False).tolist()
            rows = [self.env.data.moviesFeatures.iloc[index] for index in indexes]
            recommendations = pd.DataFrame(rows, columns=self.env.data.moviesFeatures.columns)

        else:
            raise ValueError("Invalid method specified. Choose from 'Collaborative', 'Content-Based', 'Hybrid' or 'RL'.")
        
        return recommendations
