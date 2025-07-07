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
steps = 3 # recommendations per userId
episodes = 610*60 # number of users getting recommendations
alpha = 0.0009 # learning rate
beta  = 0.0001 # learning rate for critic, if None then shared optimizer is used
gamma = 0.99 # discount factor
device = torch.device('cuda') # device to use for learning (cuda or cpu)
bufferSize = 256 # training batch size
batchSize = 10000 # the total batch size for sampling
repeatUsers = True # repeat users in env or remove once used
stateUpdateFactor = 0.2
recentExperienceRatio = 0.5
keepProfiles = True
usePiForCritic = True # use actor's pi for critic or selected action embedding
# Entropy parameters
initialEntropyCoef = 0.05 # entropy coefficient
maxEntropy = 15
# Temperature parameters
initialTemperature = 1.2
useTemperatureDecay = False
# Penalties and bonuses
repeatedActionPenalty = 0.2 # penalty for repeated actions
newActionBonus = 0.2 # bonus for new actions
betterActionBonus = 0.15
rarityBonus = 0.2
entropyThreshold = 7.5

loadAgentName = None
saveName=f"{steps}_{episodes}_{alpha}_{beta}_{gamma}_{time.strftime('%d-%m-%Y_%H-%M')}"


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
agent = Agent(alpha=alpha, beta=beta, gamma=gamma, usePiForCritic=usePiForCritic, actionsNum=env.action_space.n, observationDim=env.observation_space.shape[0], device=device, batchSize=batchSize, pretrainedActor=pretrainedActorModel, pretrainedCritic=pretrainedCriticModel)

if loadAgentName:
    agent.load_models(fileName=loadAgentName)


#%%
# Reset scores
minimumReward = env.reward_range[0]
bestScore = minimumReward
scoreHistory = deque(maxlen=100)
avgScores = []


#%%
# Train function
def train():
    global bestScore, temperature, maxEntropy, recentExperienceRatio, entropyThreshold
    startTime = time.time()
    episode = 0
    avgScore = minimumReward
    entropyCoef = initialEntropyCoef
    entropy = maxEntropy
    temperature = initialTemperature
    if useTemperatureDecay:
        temperatureDecay = initialTemperature - 0.9
    progress = 0.0

    actionEmbedding = None
    newActionEmbedding = None
    
    for _ in range(episodes+bufferSize):
        observation, obsInfo = env.reset()
        
        done = False
        score = 0
        
        while not done:
            actionIndex, _ = agent.choose_action(observation, temperature=temperature)
        
            newObservation, reward, done, info = env.step(actionIndex)
            
            # Penalty for repeating actions
            frequency = agent.actionHistory[steps*-10:].count(actionIndex)
            reward = max(reward - repeatedActionPenalty*frequency, minimumReward)

            frequency = agent.actionHistory[steps*-100:steps*-10].count(actionIndex)
            reward = max(reward - repeatedActionPenalty/10*frequency*(1-progress/2), minimumReward)

            # Use bonus when batch is filled
            if len(agent.batch) > bufferSize and reward > 0:
                # Bonus for new actions
                if actionIndex not in agent.actionHistory[steps*-500:]:
                    reward += reward * newActionBonus
                # Bonus for better actions
                if reward > avgScore:
                    reward += reward * betterActionBonus * progress

            if useTemperatureDecay:
                temperature = initialTemperature - (episode/episodes)*temperatureDecay
            
            score += reward

            if not usePiForCritic:
                movieId = data.moviesDf.iloc[actionIndex]['movieId']
                actionEmbedding = np.array(data.get_movie_features(movieId)[:19])

                newActionIndex, _ = agent.choose_action(newObservation, temperature=temperature)
                newMovieId = data.moviesDf.iloc[newActionIndex]['movieId']
                newActionEmbedding = np.array(data.get_movie_features(newMovieId)[:19])
                
            agent.add_to_batch(observation, actionIndex, actionEmbedding, reward, newObservation, newActionEmbedding, done)
            observation = newObservation
                        
        if len(agent.batch) >= bufferSize:
            episode += 1

            progress = episode / episodes
            
            avgEntropy = np.mean(agent.entropyHistory[-100:])

            entropy = agent.learn(entropyCoef=entropyCoef, maxEntropy=maxEntropy, batchSampleSize=bufferSize, recentExperienceRatio=recentExperienceRatio)

            score = score / steps

            scoreHistory.append(score)
            avgScore = sum(scoreHistory) / len(scoreHistory) if scoreHistory else minimumReward
            avgScores.append(avgScore)

            if avgScore > bestScore and progress > 0.15 and entropy > 3.5:
                bestScore = avgScore
                agent.save_models(fileName=saveName)
            if not episode % 25:
                print(f"Progress {progress*100:.2f}%\tEpisode {episode}\tScore: {score:.3f}\tAvg score: {avgScore:.3f}")
                print(f"Entropy {avgEntropy:.3f}\tEntropyCoef {entropyCoef:.3f}\tRecentExperienceRatio {recentExperienceRatio:.3f}\tTemperature {temperature:.3f}")


    endTime = time.time()
    print(f"Execution time: {endTime - startTime:.2f} seconds")
    print(f"Best score: {bestScore:.2f}")
    print(saveName)
    print(f"Best avg. rating: {env.reward_to_rating(bestScore):.2f} stars")

    return bestScore, avgScores


#%%
# Train
saveName=f"{steps}_{episodes}_{alpha}_{beta}_{gamma}_{time.strftime('%d-%m-%Y_%H-%M')}"
bestScore, avgScores = train()

#%%
# Plot the results
agent.plot_all(saveName=saveName, avgScores=avgScores, startSlice=bufferSize+episodes//10)
