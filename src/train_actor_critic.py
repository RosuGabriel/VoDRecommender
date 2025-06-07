#%%
# Imports
from environment.agent import Agent
from environment.movie_lens_env import MovieLensEnv
from data_loader.movie_lens_data import MovieLensData
import time
import torch
from utils.paths import BASE_DIR
from collections import deque



#%%
# Parameters
steps = 3 # recommendations per user
episodes = 611*40 # number of users getting recommendations
alpha = 0.00005 # learning rate
gamma = 0.99 # discount factor
entropyCoef = 1 # entropy coefficient
device = torch.device('cuda') # device to use for learning (cuda or cpu)
network = 'papc' # ac | pac | papc  <=>  p = pretrained | a = actor | c = critic
bufferSize = 512 # training batch size -> steps size means training after each episode
batchSize = int(bufferSize * 4) # the total batch size for sampling
maxEntropy = 10 # minimum entropy for the policy
repeatUsers = True # repeat users in env or remove once used
temperature = 1 # temperature for exploration/exploitation
repeatedActionPenalty = 0.15 # penalty for repeated actions
saveName=f"{network}_{steps}_{episodes}_{alpha}_{gamma}_{time.strftime('%d-%m-%Y_%H-%M')}"

updateStep = 400
recentEntropiesDim = 100


#%% 
# Load data
data = MovieLensData(includeEstimatedRatings=True)


#%%
# Create environment
env = MovieLensEnv(maxSteps=steps, data=data, repeatUsers=repeatUsers, keepProfiles=True)


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
scoreHistory = deque(maxlen=100)
avgScores = []


#%%
# Train function
def train():
    global bestScore, temperature, maxEntropy, entropyCoef, recentEntropiesDim, updateStep
    startTime = time.time()
    episode = 0
    avgScore = -1
    recentEntropies = deque(maxlen=recentEntropiesDim)

    for _ in range(episodes+bufferSize):
        observation, obsInfo = env.reset()
        
        done = False
        score = 0
        
        while not done:
            action, _ = agent.choose_action(observation, temperature=temperature)
            
            # Gradient clipping (limits big changes)
            torch.nn.utils.clip_grad_norm_(agent.actorCritic.parameters(), 0.5)
            newObservation, reward, done, info = env.step(action, updateFactor=0.1)
            
            # Penalty for repeating actions
            frequency = agent.actionHistory[steps*-30:].count(action)
            reward = max(reward - repeatedActionPenalty*frequency, -1)

            # Bonus for better actions
            if reward > avgScore:
                reward = min(reward + (reward - avgScore) * 1.5, 1.0)
            
            score += reward
            agent.add_to_batch(observation, action, reward, newObservation, done)
            observation = newObservation
            
        if len(agent.batch) >= bufferSize:
            episode += 1

            # First 5% of episodes just explore
            if episode > episodes*0.05:
            # If performance decreases increase entropy coefficient
                if avgScore < sum(avgScores[-100:])/len(avgScores[-100:]) and entropyCoef < 0.91:
                    entropyCoef *= 1.05
                else:
                    entropyCoef = max(entropyCoef * 0.95, 0.1)  # decay entropy coefficient

                # If score is improving, decrease maximum entropy
                if avgScore > sum(avgScores[-1000:])/len(avgScores[-1000:]):
                    maxEntropy = max(maxEntropy * 0.99, 2)  # decay minimum entropy
                    
                # If entropy stays high, increase maximum entropy
                if sum(recentEntropies) / len(recentEntropies) > maxEntropy * 1.05:
                    maxEntropy = maxEntropy * 1.05
                
            entropy = agent.learn(entropyCoef=entropyCoef, maxEntropy=maxEntropy, batchSize=bufferSize)
            if entropy is not None:
                recentEntropies.append(entropy)

            score = score / steps

            scoreHistory.append(score)
            avgScore = sum(scoreHistory) / len(scoreHistory) if scoreHistory else -1
            avgScores.append(avgScore)

            if avgScore > bestScore and episode > episodes*0.1:
                bestScore = avgScore
                agent.save_models(fileName=saveName)
            if not episode % 25:
                print(f"Episode {episode}\tScore: {score:.3f}\tAvg score: {avgScore:.3f}")
                print(f"Entropy {sum(recentEntropies) / len(recentEntropies):.3f}\tMaxEntropy {maxEntropy:.3f}\tEntropyCoef {entropyCoef:.3f}")


    endTime = time.time()
    print(f"Execution time: {endTime - startTime:.2f} seconds")
    print(f"Best score: {bestScore:.2f}")

    return bestScore, avgScores


#%%
# Train
bestScore, avgScores = train()


#%%
# Plot the results
agent.plot_all(saveName=saveName, avgScores=avgScores)
print(f"Best avg. reward: {env.reward_to_rating(bestScore):.2f} stars")