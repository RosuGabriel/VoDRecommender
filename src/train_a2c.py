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
# Trainging parameters
steps = 3 # recommendations per user
episodes = 611*40 # number of users getting recommendations
alpha = 0.005 # learning rate
beta = 0.0001 # learning rate for critic, if None then shared optimizer is used
gamma = 0.99 # discount factor
device = torch.device('cuda') # device to use for learning (cuda or cpu)
bufferSize = 256 # training batch size
batchSize = 10000 # the total batch size for sampling
repeatUsers = True # repeat users in env or remove once used
stateUpdateFactor = 0.2
recentExperienceRatio = 0.5
keepProfiles = False
# Entropy parameters
entropyCoef = 0.03 # entropy coefficient
maxEntropy = 10
pivotEntropy = 8
# Temperature parameters
temperature = 1.3 # initial temperature
minTemperature = 1
maxTemperature = 1.5
# Penalties and bonuses
repeatedActionPenalty = 0.1 # penalty for repeated actions
newActionBonus = 0.3 # bonus for new actions
betterActionBonus = 1
rarityBonus = 0.2

saveName=f"{steps}_{episodes}_{alpha}_{gamma}_{time.strftime('%d-%m-%Y_%H-%M')}"

#%% 
# Load data
data = MovieLensData(includeEstimatedRatings=True)


#%%
# Create environment
env = MovieLensEnv(maxSteps=steps, data=data, repeatUsers=repeatUsers, keepProfiles=keepProfiles, updateFactor=stateUpdateFactor, rarityBonus=rarityBonus)


#%%
# Create agent
actorModelName = "actor_10May0028.pt"
pretrainedActorModel = torch.load(BASE_DIR / f"models/pretrained/{actorModelName}")
criticModelName = "critic_10May0028.pt"
pretrainedCriticModel = torch.load(BASE_DIR / f"models/pretrained/{criticModelName}")
agent = Agent(alpha=alpha, beta=beta, gamma=gamma, actionsNum=env.action_space.n, observationDim=env.observation_space.shape[0], device=device, batchSize=batchSize, pretrainedActor=pretrainedActorModel, pretrainedCritic=pretrainedCriticModel)


#%%
# Reset scores
minimumReward = env.reward_range[0]
bestScore = minimumReward
scoreHistory = deque(maxlen=100)
avgScores = []


#%%
# Train function
def train():
    global bestScore, temperature, maxEntropy, entropyCoef
    startTime = time.time()
    episode = 0
    avgScore = minimumReward
    recentEntropies = deque(maxlen=100)
    entropy = maxEntropy
    
    for _ in range(episodes+bufferSize):
        observation, obsInfo = env.reset()
        
        done = False
        score = 0
        
        while not done:
            action, _ = agent.choose_action(observation, temperature=temperature)
            
            newObservation, reward, done, info = env.step(action)
            
            # Penalty for repeating actions
            frequency = agent.actionHistory[steps*-100:].count(action)
            reward = max(reward - repeatedActionPenalty*frequency, minimumReward)

            # Bonus for new actions
            if action not in agent.actionHistory[steps*-100:]:
                reward += abs(reward) * newActionBonus

            # Bonus for better actions
            if reward > avgScore:
                reward += abs(reward) * betterActionBonus
            
            score += reward
            agent.add_to_batch(observation, action, reward, newObservation, done)
            observation = newObservation
                        
        if len(agent.batch) >= bufferSize:
            episode += 1

            # if episode % updateStep == 0:
            #     if temperature <= maxTemperature:
            #         temperature *= 1.05

            entropy = agent.learn(entropyCoef=entropyCoef, maxEntropy=maxEntropy, batchSampleSize=bufferSize, recentExperienceRatio=recentExperienceRatio)
            if entropy is not None:
                recentEntropies.append(entropy)

            score = score / steps

            scoreHistory.append(score)
            avgScore = sum(scoreHistory) / len(scoreHistory) if scoreHistory else minimumReward
            avgScores.append(avgScore)

            if np.mean(avgScores[-100:]) > bestScore:
                bestScore = np.mean(avgScores[-100:])
                agent.save_models(fileName=saveName)
            if not episode % 25:
                print(f"Episode {episode}\tScore: {score:.3f}\tAvg score: {avgScore:.3f}")
                print(f"Entropy {sum(recentEntropies) / len(recentEntropies):.3f}\tEntropyCoef {entropyCoef:.3f}\tTemperature {temperature:.3f}")


    endTime = time.time()
    print(f"Execution time: {endTime - startTime:.2f} seconds")
    print(f"Best score: {bestScore:.2f}")

    return bestScore, avgScores


#%%
# Train
bestScore, avgScores = train()

#%%
# Plot the results
agent.plot_all(saveName=saveName, avgScores=avgScores, startSlice=bufferSize)
print(f"Best avg. rating: {env.reward_to_rating(bestScore):.2f} stars")
