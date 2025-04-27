#%%
# Imports
import random
from envDef import VoDEnv
from ClassicRecommender.data import moviesIDs, user_rating_of_movie

#%%
# Env testing
env = VoDEnv(maxSteps=100, maxHistoryDim=20, currentUserId=4)
state = env.reset() 

print("Observație inițială:")
print(state)

#%%
# Random agent
def random_agent(userId: int = 0):
    # Choose random movie to recommend
    movieId = random.choice(moviesIDs)
    
    # Get user's rating
    rating = user_rating_of_movie(userId, movieId)
    
    return movieId, rating

#%%
# Simulation
for episode in range(1):
    state = env.reset()
    done = False
    totalReward = 0
    
    while not done:
        # Action from agent
        action = random_agent()
        
        # Step with chosen action
        next_state, reward, done, info = env.step(action)
        
        # Add reward to total reward
        totalReward += reward
        
        # Print results
        print(f"Reward: {reward:.2f}")
        env.render()  # Watch history
        
    print(f"Total reward in episode {episode+1}: {totalReward:.2f}")
    print('------------------------------------------------------')
print('------------------------------------------------------')
