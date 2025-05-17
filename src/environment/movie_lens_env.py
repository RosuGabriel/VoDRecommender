# Imports
from data_loader.movie_lens_data import MovieLensData
import gym
import numpy as np
import random



# Env definition
class MovieLensEnv(gym.Env):
    def __init__(self, data: MovieLensData=MovieLensData(), maxSteps=100, repeatUsers=False, showInitialDetails=False):
        super(MovieLensEnv, self).__init__()
        self.data = data
        self.repeatUsers = repeatUsers
        self.userEmbeddings = data.get_all_user_profiles_from_csv()
        self.maxSteps = maxSteps
        self.reward_range = (-1.0, 1.0)
        self.availableUsers = list(self.data.all_users.copy())

        if showInitialDetails:
            print(f"Available users: {len(self.availableUsers)}")
            print(f"Movies: {len(self.data.moviesDf)}")
            print(f"Ratings: {len(self.data.ratingsDf)}")
            
        self.reset()
        
        self.action_space = gym.spaces.Discrete(len(self.data.moviesDf))
        self.observation_space = gym.spaces.Box(low=-5, high=5, shape=self.userEmbedding.shape, dtype=np.float32) # approximative range of values


    # New state and reset stepCount
    def reset(self):
        self.stepCount = 0
        self._reset_user()

        return self.userEmbedding, {"userId": self.userId,"userRatings": self.userRatings}


    # The action is the index of the movie in the moviesDf
    def step(self, action, updateFactor=0.2):
        self.stepCount += 1

        movie_id = self.data.moviesDf.iloc[action]['movieId']
        
        rating_row = self.userRatings[self.userRatings['movieId'] == movie_id]
        if rating_row.empty:
            reward = 0.0  # not rated movie
        else:
            reward = self._rating_to_reward(rating_row['rating'].values[0])

        movieFeatures = np.array(self.data.get_movie_features(movie_id)[:19])
        
        self.userEmbedding = self.userEmbedding*(1-updateFactor) + movieFeatures*updateFactor*reward
        
        # print('user',self.userEmbedding)
        # print('movie',movieFeatures)
        
        done = self.stepCount >= self.maxSteps
        info = {"movieFeatures": movieFeatures}

        return self.userEmbedding, reward, done, info


    def render(self, mode='human'):
        print(f"User ID: {self.userId}")
        print(f"User embedding: {self.userEmbedding}")
    

    # New user is chosen randomly
    def _reset_user(self):
        if len(self.availableUsers) == 0:
            print("No more users available. User list refreshed.")
            self.availableUsers = list(self.data.all_users.copy())

        self.userId = random.choice(self.availableUsers)
        
        if not self.repeatUsers:
            self.availableUsers.remove(self.userId)

        self.userEmbedding = self.userEmbeddings[self.userEmbeddings['userId'] == self.userId].values[0][1:20]
        self.userRatings = self.data.user_ratings(self.userId).copy()
       

    # Function to map rating to reward
    def _rating_to_reward(self, rating: float):
        if rating >= 4.5:
            return 1.0
        elif rating >= 4:
            return 0.8
        elif rating >= 3:
            return 0.0
        elif rating >= 2:
            return -0.5
        return -1.0
