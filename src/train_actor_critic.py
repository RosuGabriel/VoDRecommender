#%%
# Imports
import numpy as np
from actor_critic.agent import Agent
from environment.movie_lens_env import MovieLensEnv
from data_loader.movie_lens_data import MovieLensData
import time
import torch
from utils.paths import BASE_DIR



#%% 
# Parameters
steps = 3 # recommendations per user
episodes = 1000 # number of users getting recommendations
alpha = 0.001 # learning rate   
gamma = 0.99 # discount factor

# Choose a method ac | pac | papc  <=>  p = pretrained | a = actor | c = critic
method = 'papc'


#%% 
# Load data
data = MovieLensData(includeEstimatedRatings=True)


#%%
# Create environment
env = MovieLensEnv(maxSteps=steps, data=data, repeatUsers=False)


#%% 
# Create agent
if method == 'ac':
    # Not pretrained
    agent = Agent(alpha=0.001, gamma=gamma, actionsNum=env.action_space.n)
else:
    actorModelName = "actor_10May0028.pt"
    pretrainedActorModel = torch.load(BASE_DIR / f"tmp/pretrained/{actorModelName}")
    if method == 'pac':
        agent = Agent(alpha=0.001, gamma=gamma, pretrainedActor=pretrainedActorModel, actionsNum=env.action_space.n)
    elif method == 'papc':
        criticModelName = "critic_10May0028.pt"
        pretrainedCriticModel = torch.load(BASE_DIR / f"tmp/pretrained/{criticModelName}")
        agent = Agent(alpha=0.001, gamma=gamma, pretrainedActor=pretrainedActorModel, pretrainedCritic=pretrainedCriticModel, actionsNum=env.action_space.n, data=data)


#%% 
# Training
bestScore = env.reward_range[0]
scoreHistory = []
loadCheckpoint = False

if loadCheckpoint:
    agent.load_models()

startTime = time.time()

for i in range(episodes):
    observation, obsInfo = env.reset()
    observation = observation / np.linalg.norm(observation + 1e-8)

    done = False
    score = 0
    
    while not done:
        action = agent.choose_action(observation)
        
        newObservation, reward, done, info = env.step(action)   
        score += reward
        if not loadCheckpoint:
            agent.learn(observation, reward, newObservation, done)
        observation = newObservation

    scoreHistory.append(score)
    avgScore = np.mean(scoreHistory[-100:])

    if avgScore > bestScore:
        bestScore = avgScore
        if not loadCheckpoint:
            agent.save_models(fileName=f"{method}_{steps}_{episodes}_{alpha}_{gamma}")

    print(f"Episode {i}\tScore: {score:.1f}\tAvg score: {avgScore:.1f}")

endTime = time.time()
print(f"Execution time: {endTime - startTime:.2f} seconds")
print(f"Best score: {bestScore:.2f}")

#%%
# Plot losses and actions
saveName=f"{method}_{steps}_{episodes}_{alpha}_{gamma}"
agent.plot_all(saveName=saveName)


#%%
# Verify model on my users
myData = MovieLensData(includeMyRatings=True)

#%%
index = agent.choose_action(myData.get_user_profile_from_csv(611)[1:20])
data.moviesDf.iloc[index]['title'] # movie list from "data" because index might be different in "myData"
