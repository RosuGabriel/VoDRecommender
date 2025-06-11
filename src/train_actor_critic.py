#%%
# Imports
from environment.agent import Agent
from environment.movie_lens_env import MovieLensEnv
from data_loader.movie_lens_data import MovieLensData
import time
import torch
from utils.paths import BASE_DIR
from collections import deque
import numpy as np



#%%
# Parameters
steps = 3 # recommendations per user
episodes = 611*2 # number of users getting recommendations
alpha = 0.0003 # learning rate
gamma = 0.99 # discount factor
entropyCoef = 0.2 # entropy coefficient
device = torch.device('cuda') # device to use for learning (cuda or cpu)
bufferSize = 256 # training batch size
batchSize = 10000 # the total batch size for sampling
maxEntropy = 9 # minimum entropy for the policy
repeatUsers = True # repeat users in env or remove once used
temperature = 1.5 # temperature for exploration/exploitation
repeatedActionPenalty = 0.15 # penalty for repeated actions
saveName=f"{steps}_{episodes}_{alpha}_{gamma}_{time.strftime('%d-%m-%Y_%H-%M')}"
stateUpdateFactor = 0.3
explorationPercentage = int(0.05*episodes) # percentage of episodes to explore at the beginning
rewardScale = 10


#%% 
# Load data
data = MovieLensData(includeEstimatedRatings=True)


#%%
# Create environment
env = MovieLensEnv(maxSteps=steps, data=data, repeatUsers=repeatUsers, keepProfiles=True)


#%%
# Create agent
actorModelName = "actor_10May0028.pt"
pretrainedActorModel = torch.load(BASE_DIR / f"models/pretrained/{actorModelName}")
criticModelName = "critic_10May0028.pt"
pretrainedCriticModel = torch.load(BASE_DIR / f"models/pretrained/{criticModelName}")
agent = Agent(alpha=alpha, gamma=gamma, actionsNum=env.action_space.n, observationDim=env.observation_space.shape[0], device=device, batchSize=batchSize, pretrainedActor=pretrainedActorModel, pretrainedCritic=pretrainedCriticModel)


#%%
# Reset scores
bestScore = env.reward_range[0] * rewardScale
scoreHistory = deque(maxlen=100)
avgScores = []


#%%
# Train function
def train():
    global bestScore, temperature, maxEntropy, entropyCoef
    startTime = time.time()
    episode = 0
    avgScore = -rewardScale
    recentEntropies = deque(maxlen=100)
    entropy = maxEntropy
    #entropyDiscount = 6 / episodes
    
    for _ in range(episodes+bufferSize):
        observation, obsInfo = env.reset()
        
        done = False
        score = 0
        
        while not done:
            action, _ = agent.choose_action(observation, temperature=temperature)
            
            # Gradient clipping (limits big changes)
            torch.nn.utils.clip_grad_norm_(agent.actorCritic.parameters(), max_norm=0.5)
          
            newObservation, reward, done, info = env.step(action, updateFactor=stateUpdateFactor)
            reward *= rewardScale

            # Penalty for repeating actions
            frequency = agent.actionHistory[steps*-30:].count(action)
            reward = max(reward - repeatedActionPenalty*frequency, -rewardScale)

            # Bonus for better actions
            if reward > avgScore:
                reward = reward + (reward - avgScore)
            
            score += reward
            agent.add_to_batch(observation, action, reward, newObservation, done)
            observation = newObservation
                        
        if len(agent.batch) >= bufferSize:
            episode += 1

            if episode > explorationPercentage:
                if avgScore < np.mean(avgScores[-100:]):
                    if  temperature < 1.4:
                        temperature = temperature * 1.05
                    elif entropyCoef < 0.5:
                        entropyCoef = entropyCoef * 1.05
                elif temperature > 1.0:
                    temperature = temperature * 0.99
                else:
                    entropyCoef = max(entropyCoef*0.99, 0.1)

            entropy = agent.learn(entropyCoef=entropyCoef, maxEntropy=maxEntropy, batchSampleSize=bufferSize)
            if entropy is not None:
                recentEntropies.append(entropy)

            score = score / steps

            scoreHistory.append(score)
            avgScore = sum(scoreHistory) / len(scoreHistory) if scoreHistory else -rewardScale
            avgScores.append(avgScore)

            if avgScore > bestScore and episode > explorationPercentage:
                bestScore = avgScore
                agent.save_models(fileName=saveName)
            if not episode % 25:
                print(f"Episode {episode}\tScore: {score:.3f}\tAvg score: {avgScore:.3f}")
                print(f"Entropy {sum(recentEntropies) / len(recentEntropies):.3f}\tEntropyCoef {entropyCoef:.3f}\tTemperature {temperature:.3f}")


    endTime = time.time()
    print(f"Execution time: {endTime - startTime:.2f} seconds")
    print(f"Best score: {bestScore/rewardScale:.2f}")

    return bestScore, avgScores


#%%
# Train
bestScore, avgScores = train()


#%%
# Plot the results
agent.plot_all(saveName=saveName, avgScores=avgScores, startSlice=bufferSize+explorationPercentage)
print(f"Best avg. rating: {env.reward_to_rating(bestScore/rewardScale):.2f} stars")
