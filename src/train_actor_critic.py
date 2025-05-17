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
episodes = 611*6 # number of users getting recommendations
alpha = 0.0001 # learning rate
gamma = 0.99 # discount factor
entropyCoef = 0.05 # entropy coefficient
device = torch.device('cuda') # device to use for learning (cuda or cpu)
network = 'papc' # ac | pac | papc  <=>  p = pretrained | a = actor | c = critic
bufferSize = 128 # training batch size -> steps size means training after each episode
batchSize = bufferSize * 3


#%% 
# Load data
data = MovieLensData(includeEstimatedRatings=True)


#%%
# Create environment
env = MovieLensEnv(maxSteps=steps, data=data, repeatUsers=False)


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

# agent.load_models()

#%%
# Reset scores
bestScore = env.reward_range[0]
scoreHistory = []
avgScores = []


#%%
# Train functions
def train_agent():
    loadCheckpoint = False

    if loadCheckpoint:
        agent.load_models()

    startTime = time.time()

    for i in range(episodes):
        observation, obsInfo = env.reset()

        done = False
        score = 0
        
        while not done:
            action = agent.choose_action(observation)
            
            newObservation, reward, done, info = env.step(action)
            score += reward
            if not loadCheckpoint:
                agent.learn(observation, reward, newObservation, done, entropyCoef=entropyCoef)
            observation = newObservation

        score = score / steps

        scoreHistory.append(score)
        avgScore = np.mean(scoreHistory[-100:])
        avgScores.append(avgScore)

        if avgScore > bestScore:
            bestScore = avgScore
            if not loadCheckpoint:
                agent.save_models(fileName=f"{network}_{steps}_{episodes}_{alpha}_{gamma}")

        print(f"Episode {i}\tScore: {score:.2f}\tAvg score: {avgScore:.2f}")

    endTime = time.time()
    print(f"Execution time: {endTime - startTime:.2f} seconds")
    print(f"Best score: {bestScore:.2f}")

    return avgScores


def train_agent_with_batches(bestScore=-1):
    startTime = time.time()
    episode = 0

    for _ in range(episodes):
        observation, obsInfo = env.reset()
        
        done = False
        score = 0
        
        while not done:
            action = agent.choose_action(observation)
            
            newObservation, reward, done, info = env.step(action)
            
            # Penalty for repeating actions
            frequency = agent.actionHistory[:-steps*5].count(action)
            reward = max(reward - 0.05*frequency, -1.0)
            
            score += reward
            agent.add_to_batch(observation, action, reward, newObservation, done)
            observation = newObservation

        #if len(agent.batch) >= bufferSize:
        agent.learn_from_batch(entropyCoef=entropyCoef)

        score = score / steps

        scoreHistory.append(score)
        avgScore = np.mean(scoreHistory[-100:])
        avgScores.append(avgScore)

        if avgScore > bestScore and episode > 100:
            bestScore = avgScore
            agent.save_models(fileName=f"{network}_{steps}_{episodes}_{alpha}_{gamma}_{time.strftime('%d-%m-%Y_%H-%M')}")
        episode += 1
        if not episode % 100:
            print(f"Episode {episode}\tScore: {score:.3f}\tAvg score: {avgScore:.3f}")

    endTime = time.time()
    print(f"Execution time: {endTime - startTime:.2f} seconds")
    print(f"Best score: {bestScore:.2f}")

    return bestScore, avgScores


#%%
# Train and plot
if bufferSize < 2:
    avgScores = train_agent()
else:
    bestScore, avgScores = train_agent_with_batches(bestScore=bestScore)

saveName=f"{network}_{steps}_{episodes}_{alpha}_{gamma}_{time.strftime('%d-%m-%Y_%H-%M')}"
agent.plot_all(saveName=saveName, avgScores=avgScores)


# #%%
# # Verify model on my users
# myData = MovieLensData(includeMyRatings=True)

# #%%
# index = agent.choose_action(myData.get_user_profile_from_csv(614)[1:20])
# data.moviesDf.iloc[index]['title'] # movie list from "data" because index might be different in "myData"
