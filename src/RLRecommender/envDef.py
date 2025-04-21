#%%
# Imports
import gym
from gym import spaces
import numpy as np
from ClassicRecommender.data import genres_of_movie, genres_to_vector

#%%
# Config
MAX_MOVIE_ID = 193609 # not the number of movies
NUM_GENRES = 19 # includes the absence of genre
NUM_USERS = 610
MAX_RATING = 5

#%%
# Env definition
class VoDEnvWithHistory(gym.Env):
    rewards = {0:-0.2, 1:-1, 2:-0.8, 3:-0.6, 4:-0.2, 5:0.1, 6:0.2, 7:0.4, 8:0.6, 9:0.8, 10:1} # rating -> reward
    def __init__(self, maxSteps=10, maxHistoryDim=10, currentUserId=0):
        super(VoDEnvWithHistory, self).__init__()
        self.stepCount = 0
        self.maxSteps = maxSteps
        self.maxHistoryDim = maxHistoryDim
        self.currentUserId = currentUserId

        # STATES VECTOR
        # History of watched movies (maxHistoryDim) + userId (1)
        
        # Initial idea
        # Every row = (movieId: int, genres: vector binar de lungime G)
        # self.observation_space = spaces.Dict({
        #     "movie_ids": spaces.MultiDiscrete([MAX_MOVIE_ID + 1] * maxHistoryDim),
        #     "genres": spaces.MultiBinary(NUM_GENRES * maxHistoryDim)
        # })

        # Better idea
        # First row is the user ID
        # Every row = [movieId, g1, ..., gn] (gi is a bit)
        self.observation_space = spaces.Box(
            low=np.array([[0] + [0] * NUM_GENRES] * (maxHistoryDim+1)),
            high=np.array([[MAX_MOVIE_ID] + [1] * NUM_GENRES] * (maxHistoryDim+1)),
            shape=(maxHistoryDim+1, 1 + NUM_GENRES),
            dtype=np.int32
        )
        
        # ACTIONS
        self.action_space = spaces.Tuple((
            spaces.Discrete(MAX_MOVIE_ID),  # watch a movie
            spaces.Discrete(MAX_RATING * 2 + 1) # rate it (0=not rated, 1-10=rated) actually 1-5 with half ratings
        ))  

        # INITIAL STATE
        self.state = np.array([[0]*(NUM_GENRES+1)]*(maxHistoryDim+1), dtype=np.int32)

    def reset(self):
        self.stepCount = 0
        self.state = np.array([[0]*(NUM_GENRES+1)]*(self.maxHistoryDim+1), dtype=np.int32)
        self.state[0][0] = self.currentUserId
        return self.state
    
    def render(self, mode='human'):
        print("User:", self.state[0][0])
        print(f"Step: {self.stepCount}")
        print("Watch history:")
        for i, movie in enumerate(self.state[1:], start=1):
            print(f"Movie {i}: ID {movie[0]}, Genres: {movie[1:]}")
        print("--------------------")

    def step(self, action):
        movieId, rating = action
        
        # Get genres
        genres = genres_of_movie(movieId)
        # Vectorize genres
        genres = genres_to_vector(genres)
        
        movieData = np.zeros(NUM_GENRES + 1, dtype=np.int32)
        movieData[0] = movieId
        movieData[1:] = genres

        # Shift the hiostory of watched movies and add the new movie
        # First row is the user ID
        self.state[1:] = np.roll(self.state[1:], shift=1, axis=0)
        self.state[1] = movieData

        # Reward calculation
        reward = self.rewards[rating]

        # Finish episode if maxSteps is reached
        self.stepCount += 1
        done = self.stepCount >= self.maxSteps

        return self.state, reward, done, {}



class VoDEnv(gym.Env):
    rewards = {0:-0.2, 1:-1, 2:-0.8, 3:-0.6, 4:-0.2, 5:0.1, 6:0.2, 7:0.4, 8:0.6, 9:0.8, 10:1} # rating -> reward
    def __init__(self, maxSteps=10, maxHistoryDim=10, currentUserId=0):
        super(VoDEnv, self).__init__()
        self.stepCount = 0
        self.maxSteps = maxSteps
        self.currentUserId = currentUserId
        self.userRatings = {} # to be added

        # STATES VECTOR
        self.observation_space = spaces.Box(
            low=np.array([[0] + [0] * NUM_GENRES]),
            high=np.array([[MAX_MOVIE_ID] + [1] * NUM_GENRES] * (maxHistoryDim+1)),
            shape=(maxHistoryDim+1, 1 + NUM_GENRES),
            dtype=np.int32
        )
    def reset(self):
        self.stepCount = 0
        return self.state
    
    def render(self, mode='human'):
        print("User:", self.state[0][0])
        print(f"Step: {self.stepCount}")
        print("Watch history:")
        for i, movie in enumerate(self.state[1:], start=1):
            print(f"Movie {i}: ID {movie[0]}, Genres: {movie[1:]}")
        print("--------------------")

    def step(self, action):
        movieId, rating = action
        
        # Get genres
        genres = genres_of_movie(movieId)
        # Vectorize genres
        genres = genres_to_vector(genres)
        
        movieData = np.zeros(NUM_GENRES + 1, dtype=np.int32)
        movieData[0] = movieId
        movieData[1:] = genres

        # Shift the hiostory of watched movies and add the new movie
        # First row is the user ID
        self.state[1:] = np.roll(self.state[1:], shift=1, axis=0)
        self.state[1] = movieData

        # Reward calculation
        reward = self.rewards[rating]

        # Finish episode if maxSteps is reached
        self.stepCount += 1
        done = self.stepCount >= self.maxSteps

        return self.state, reward, done, {}