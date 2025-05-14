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
steps = 10 # recommendations per user
episodes = 3 # number of users getting recommendations
alpha = 0.1 # learning rate
gamma = 0.99 # discount factor
entropyCoef = 0.05 # entropy coefficient

# Choose a method ac | pac | papc  <=>  p = pretrained | a = actor | c = critic
models = 'papc'
# Batch training method
batchSize = steps


#%% 
# Load data
data = MovieLensData(includeEstimatedRatings=True)


#%%
# Create environment
env = MovieLensEnv(maxSteps=steps, data=data, repeatUsers=False)


#%% 
# Create agent
if models == 'ac':
    agent = Agent(alpha=alpha, gamma=gamma, actionsNum=env.action_space.n)
else:
    actorModelName = "actor_10May0028.pt"
    pretrainedActorModel = torch.load(BASE_DIR / f"models/pretrained/{actorModelName}")
    if models == 'pac':
        agent = Agent(alpha=alpha, gamma=gamma, actionsNum=env.action_space.n, pretrainedActor=pretrainedActorModel)
    elif models == 'papc':
        criticModelName = "critic_10May0028.pt"
        pretrainedCriticModel = torch.load(BASE_DIR / f"models/pretrained/{criticModelName}")
        agent = Agent(alpha=alpha, gamma=gamma, actionsNum=env.action_space.n, pretrainedActor=pretrainedActorModel, pretrainedCritic=pretrainedCriticModel)


#%%
# Train funtions
def train_agent():
    bestScore = env.reward_range[0]
    scoreHistory = []
    loadCheckpoint = False
    avgScores = []

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
                agent.learn(observation, reward, newObservation, done, entropyCoef=entropyCoef)
            observation = newObservation

        score = score / steps

        scoreHistory.append(score)
        avgScore = np.mean(scoreHistory[-100:])
        avgScores.append(avgScore)

        if avgScore > bestScore:
            bestScore = avgScore
            if not loadCheckpoint:
                agent.save_models(fileName=f"{models}_{steps}_{episodes}_{alpha}_{gamma}")

        print(f"Episode {i}\tScore: {score:.2f}\tAvg score: {avgScore:.2f}")

    endTime = time.time()
    print(f"Execution time: {endTime - startTime:.2f} seconds")
    print(f"Best score: {bestScore:.2f}")

    return avgScores


def train_agent_with_batches():
    bestScore = env.reward_range[0]
    scoreHistory = []
    loadCheckpoint = False
    avgScores = []

    if loadCheckpoint:
        agent.load_models()

    startTime = time.time()

    for i in range(episodes):
        observation, obsInfo = env.reset()
        observation = observation / np.linalg.norm(observation + 1e-8)

        done = False
        score = 0
        
        while len(agent.batch) < batchSize:
            action = agent.choose_action(observation)
            
            newObservation, reward, done, info = env.step(action)   
            score += reward
            if not loadCheckpoint:
                agent.add_to_batch(observation, action, reward, newObservation, done)
            observation = newObservation

        agent.learn_from_batch(entropyCoef=entropyCoef)
    
        score = score / batchSize

        scoreHistory.append(score)
        avgScore = np.mean(scoreHistory[-100:])
        avgScores.append(avgScore)

        if avgScore > bestScore:
            bestScore = avgScore
            if not loadCheckpoint:
                agent.save_models(fileName=f"{models}_{steps}_{episodes}_{alpha}_{gamma}")

        print(f"Episode {i}\tScore: {score:.2f}\tAvg score: {avgScore:.2f}")

    endTime = time.time()
    print(f"Execution time: {endTime - startTime:.2f} seconds")
    print(f"Best score: {bestScore:.2f}")

    return avgScores


#%%
# Train and plot
if batchSize < 2:
    avgScores = train_agent()
else:
    avgScores = train_agent_with_batches()

saveName=f"{models}_{steps}_{episodes}_{alpha}_{gamma}"
agent.plot_all(saveName=saveName, avgScores=avgScores)


# #%%
# # Verify model on my users
# myData = MovieLensData(includeMyRatings=True)

# #%%
# index = agent.choose_action(myData.get_user_profile_from_csv(614)[1:20])
# data.moviesDf.iloc[index]['title'] # movie list from "data" because index might be different in "myData"
