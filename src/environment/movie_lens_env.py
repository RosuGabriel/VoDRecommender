# Imports
from data_loader.movie_lens_data import MovieLensData
import gym
import numpy as np
import random
from sklearn.metrics.pairwise import cosine_similarity
import torch



# Env definition
class MovieLensEnv(gym.Env):
    def __init__(self, data: MovieLensData=MovieLensData(), maxSteps=100, repeatUsers=False, useContinuousActions=False, showInitialDetails=False, keepProfiles=False,
                updateFactor=0.2, rarityBonus = 0.2):
        super(MovieLensEnv, self).__init__()
        self.data = data
        self.repeatUsers = repeatUsers
        self.originalUserEmbeddings = data.get_all_user_profiles_from_csv()
        self.maxSteps = maxSteps
        self.reward_range = (-1.0, 1.0)
        self.availableUsers = list(self.data.all_users.copy())
        self.userEmbeddings = {}
        self.keepProfiles = keepProfiles
        self.moviePopularity = data.movies_popularity
        self.updateFactor = updateFactor
        self.rarityBonus = rarityBonus
        self.useContinuousActions = useContinuousActions
        self.moviesEmbeddings = data.moviesFeatures.drop(['movieId', 'title', 'year'], axis=1)
        self.moviesEmbeddings = self.moviesEmbeddings.iloc[:, :19]

        if showInitialDetails:
            print(f"Available users: {len(self.availableUsers)}")
            print(f"Movies: {len(self.data.moviesDf)}")
            print(f"Ratings: {len(self.data.ratingsDf)}")
            
        self.reset()
        
        if useContinuousActions:
            self.action_space = gym.spaces.Box(low=-5, high=5, shape=self.userEmbedding.shape, dtype=np.float32) # movie Embedding
        else:
            self.action_space = gym.spaces.Discrete(len(self.data.moviesDf)) # probability of each movie

        self.observation_space = gym.spaces.Box(low=-5, high=5, shape=self.userEmbedding.shape, dtype=np.float32) # approximative range of values


    # New state and reset stepCount
    def reset(self, userId=None):
        self.stepCount = 0
        self._reset_user(userId)

        return np.array(self.userEmbedding), {"userId": self.userId,"userRatings": self.userRatings}


    # Get reward for action and update user embedding
    def step(self, action):
        self.stepCount += 1

        if self.useContinuousActions:
            movieId = self.get_movieId_from_embedding(action)
        else:
            movieId = self.get_movieId_from_index(action)
        movieFeatures = np.array(self.data.get_movie_features(movieId)[:19])
    
        reward = self.get_reward_for_movie(movieId)
        
        # Update user embedding based on the reward
        if reward:
            self.userEmbedding = self.userEmbedding*(1-self.updateFactor) + movieFeatures * self.updateFactor * reward
        
        self.userEmbeddings[self.userId] = self.userEmbedding
        
        done = self.stepCount >= self.maxSteps
        info = {"movieFeatures": movieFeatures, "movieId": movieId}

        return np.array(self.userEmbedding), reward, done, info


    def get_movieId_from_embedding(self, movieEmbedding):
        similarities = cosine_similarity([movieEmbedding], self.moviesEmbeddings)[0]
        movieIndex = similarities.argmax()
        return self.data.moviesDf.iloc[movieIndex]['movieId']
 

    def get_movieId_from_index(self, movieIndex):
        return self.data.moviesDf.iloc[movieIndex]['movieId']
    

    def get_reward_for_movie(self, movieId):
        rating_row = self.userRatings[self.userRatings['movieId'] == movieId]
        
        if rating_row.empty:
            rating = None
            reward = 0.0  # not rated movie
        else:
            rating = rating_row['rating'].values[0]
            reward = self._rating_to_reward(rating)

        if self.rarityBonus:
            rarity = 1.0 - self.moviePopularity[movieId]
            reward = reward * (1-self.rarityBonus) + rarity * self.rarityBonus

        return reward


    def render(self, mode='human'):
        print(f"User ID: {self.userId}")
        print(f"User embedding: {self.userEmbedding}")
    

    # New user is chosen randomly
    def _reset_user(self, userId=None):
        if len(self.availableUsers) == 0:
            print("No more users available. User list refreshed.")
            self.availableUsers = list(self.data.all_users.copy())

        if userId is not None:
            self.userId = userId
        else:
            self.userId = random.choice(self.availableUsers)
            if not self.repeatUsers:
                self.availableUsers.remove(self.userId)

        if self.keepProfiles and self.userId in self.userEmbeddings:
            self.userEmbedding = self.userEmbeddings[self.userId]
        elif self.keepProfiles:
            self.userEmbeddings[self.userId] = self.originalUserEmbeddings[self.originalUserEmbeddings['userId'] == self.userId].values[0][1:20]    
            self.userEmbedding = self.userEmbeddings[self.userId]
        else:
            self.userEmbedding = self.originalUserEmbeddings[self.originalUserEmbeddings['userId'] == self.userId].values[0][1:20]
        
        self.userRatings = self.data.user_ratings(self.userId).copy()
       

    # Function to map rating to reward
    def _rating_to_reward(self, rating: float):
        return 2 * (rating - 0.5) / 4.5 - 1


    def reward_to_rating(self, reward: float):
        return (reward + 1) * 4.5 / 2 + 0.5