#%%
# Imports
from data_loader.data_movie_lens import MovieLensData
import gym
import numpy as np
import random



#%%
# Env definition
class MovieLensEnv(gym.Env):
    def __init__(self, data: MovieLensData=MovieLensData(), maxSteps=100):
        super(MovieLensEnv, self).__init__()
        self.data = data
        self.maxSteps = maxSteps
        self.availableUsers = list(self.data.all_users.copy())
        self.reset()
        self.action_space = gym.spaces.Discrete(len(self.data.moviesDf))
        self.observation_space = gym.spaces.Box(low=-5, high=5, shape=self.userEmbedding.shape, dtype=np.float32) # approximative range of values


    # New state and reset stepCount
    def reset(self):
        self.stepCount = 0
        self._reset_user()

        return self.userEmbedding, {"userId": self.userId,"userRatings": self.userRatings}


    # The action is the index of the movie in the moviesDf
    def step(self, action):
        self.stepCount += 1
        movie_id = self.data.moviesDf.iloc[action]['movieId']
        
        rating_row = self.userRatings[self.userRatings['movieId'] == movie_id]
        if rating_row.empty:
            reward = 0.0  # not rated movie
        else:
            reward = self._rating_to_reward(rating_row['rating'].values[0])

        done = self.stepCount >= self.maxSteps
        info = {"movieFeatures": self.data.moviesFeatures[self.data.moviesFeatures['movieId'] == movie_id]}

        return self.userEmbedding, reward, done, info


    def render(self, mode='human'):
        print(f"User ID: {self.userId}")
    

    # New user is chosen randomly
    def _reset_user(self):
        if len(self.availableUsers) == 0:
            print("No more users available. User list refreshed.")
            self.availableUsers = list(self.data.all_users.copy())

        self.userId = random.choice(self.availableUsers)
        self.availableUsers.remove(self.userId)
        
        self.userEmbedding = np.array(self.data.calculate_user_profile(self.userId)[1:].copy())
        self.userRatings = self.data.user_ratings(self.userId).copy()
       

    def _rating_to_reward(self, rating: float):
        if rating >= 4.5:
            return 1.0
        elif rating >= 4:
            return 0.7
        elif rating >= 3:
            return 0.4
        elif rating >= 2:
            return -0.5
        return -1.0
