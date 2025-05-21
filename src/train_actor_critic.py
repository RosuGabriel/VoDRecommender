#%%
# Imports
import numpy as np
from environment.agent import Agent
from environment.movie_lens_env import MovieLensEnv
from data_loader.movie_lens_data import MovieLensData
import time
import torch
from utils.paths import BASE_DIR



#%% 
# Parameters
steps = 3 # recommendations per user
episodes = 611*16 # number of users getting recommendations
alpha = 0.00065 # learning rate
gamma = 0.99 # discount factor
entropyCoef = 0.3 # entropy coefficient
device = torch.device('cpu') # device to use for learning (cuda or cpu)
network = 'papc' # ac | pac | papc  <=>  p = pretrained | a = actor | c = critic
bufferSize = 128 # training batch size -> steps size means training after each episode
batchSize = int(bufferSize * 4) # the total batch size for sampling
minEntropy = 4 # minimum entropy for the policy
repeatUsers = True # repeat users in env or remove once used
temperature = 1.3 # temperature for exploration/exploitation
repeatedActionPenalty = 0.2 # penalty for repeated actions


#%% 
# Load data
data = MovieLensData(includeEstimatedRatings=True)


#%%
# Create environment
env = MovieLensEnv(maxSteps=steps, data=data, repeatUsers=repeatUsers)


#%%
# Create agent
if network == 'ac':
    agent = Agent(alpha=alpha, gamma=gamma, actionsNum=env.action_space.n, observationDim=env.observation_space.shape[0], device=device, batchSize=batchSize)
else:
    actorModelName = "actor_10May0028.pt"
    pretrainedActorModel = torch.load(BASE_DIR / f"models/pretrained/{actorModelName}")
    if network == 'pac':
        agent = Agent(alpha=alpha, gamma=gamma, actionsNum=env.action_space.n, observationDim=env.observation_space.shape[0], device=device, batchSize=batchSize, pretrainedActor=pretrainedActorModel)
    elif network == 'papc':
        criticModelName = "critic_10May0028.pt"
        pretrainedCriticModel = torch.load(BASE_DIR / f"models/pretrained/{criticModelName}")
        agent = Agent(alpha=alpha, gamma=gamma, actionsNum=env.action_space.n, observationDim=env.observation_space.shape[0], device=device, batchSize=batchSize, pretrainedActor=pretrainedActorModel, pretrainedCritic=pretrainedCriticModel)


#%%
# Reset scores
bestScore = env.reward_range[0]
scoreHistory = []
avgScores = []


#%%
# Train function
def train(bestScore=-1, temperature=1.0):
    startTime = time.time()
    episode = 0
   
    for _ in range(episodes):
        observation, obsInfo = env.reset()
        
        done = False
        score = 0
        
        while not done:
            action = agent.choose_action(observation, temperature=temperature)
            
            newObservation, reward, done, info = env.step(action)
            
            # Penalty for repeating actions
            frequency = agent.actionHistory[-steps*4:].count(action)
            reward = max(reward - repeatedActionPenalty*frequency, -1.0)
            
            score += reward
            agent.add_to_batch(observation, action, reward, newObservation, done)
            observation = newObservation
            
        if len(agent.batch) >= bufferSize:
            episode += 1

            agent.learn(entropyCoef=entropyCoef, minEntropy=minEntropy, batchSize=bufferSize)

            score = score / steps

            scoreHistory.append(score)
            avgScore = np.mean(scoreHistory[-100:])
            avgScores.append(avgScore)

            if avgScore > bestScore and episode > 10:
                bestScore = avgScore
                agent.save_models(fileName=f"{network}_{steps}_{episodes}_{alpha}_{gamma}_{time.strftime('%d-%m-%Y_%H-%M')}")
            if not episode % 10:
                print(f"Episode {episode}\tScore: {score:.3f}\tAvg score: {avgScore:.3f}")

    endTime = time.time()
    print(f"Execution time: {endTime - startTime:.2f} seconds")
    print(f"Best score: {bestScore:.2f}")

    return bestScore, avgScores


#%%
# Train and plot
bestScore, avgScores = train(bestScore=bestScore, temperature=temperature)

saveName=f"{network}_{steps}_{episodes}_{alpha}_{gamma}_{time.strftime('%d-%m-%Y_%H-%M')}"
agent.plot_all(saveName=saveName, avgScores=avgScores)


# #%%
# # Modify learning rate
# for paramGroup in agent.optimizer.param_groups:
#     paramGroup['lr'] = 0.007


# #%%
# # Load my users
# myData = MovieLensData(includeMyRatings=True)

# #%%
# # Test my users
# index = agent.choose_action(myData.get_user_profile_from_csv(615)[1:20])
# data.moviesDf.iloc[index]['title'] # movie list from "data" because index might be different in "myData"
