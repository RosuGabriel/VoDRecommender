# Imports
import torch
import torch.optim as optim
from torch.distributions import Categorical
from actor_critic.ActorCriticNet import ActorCriticNet
from actor_critic.PActorCriticNet import PActorCriticNet
from actor_critic.PActorPCriticNet import PActorPCriticNet
import matplotlib.pyplot as plt
import numpy as np
import random
from collections import deque



class Agent:
    def __init__(self, alpha=0.0003, gamma=0.99, actionsNum=2, observationDim=19, device=None, pretrainedActor=None, pretrainedCritic=None, batchSize=32):
        self.gamma = gamma
        self.actionsNum = actionsNum
        self.action = None
        self.batch = deque(maxlen=batchSize)
        self.device = device or torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        self.actorLossHistory = []
        self.criticLossHistory = []
        self.totalLossHistory = []
        self.deltaHistory = []
        self.actionHistory = []
        self.rewardHistory = []
        self.avgRatingHistory = []
        self.stateValueHistory = []
        self.entropyHistory = []

        if pretrainedActor and pretrainedCritic:
            self.actorCritic = PActorPCriticNet(pretrainedActor, pretrainedCritic, newInDim=observationDim, newActionsNum=actionsNum).to(self.device)
        elif pretrainedActor:
            self.actorCritic = PActorCriticNet(pretrainedActor, newInDim=observationDim, newActionsNum=actionsNum).to(self.device)
        else:
            self.actorCritic = ActorCriticNet(actionsNum=actionsNum).to(self.device)
        
        self.optimizer = optim.Adam(self.actorCritic.parameters(), lr=alpha)


    def choose_action(self, observation, temperature: float=1.0, actionsNum: int=1):
        state = torch.tensor(observation, dtype=torch.float32).to(self.device)
        
        _, probs = self.actorCritic(state)

        logits = torch.log(probs + 1e-8)  # avoiding log(0)
        
        # Temperature > 1.0 more exploration | < 1.0 more exploitation
        scaledLogits = logits / temperature

        dist = Categorical(logits=scaledLogits)
        action = dist.sample()

        self.action = action
        
        self.actionHistory.append(action.item())

        if actionsNum > 1:
            otherActions = torch.multinomial(probs, num_samples=actionsNum, replacement=False).tolist()

        return action.item(), otherActions if actionsNum > 1 else None
    

    def add_to_batch(self, currentState, action, reward, newState, done):
        self.batch.append((currentState, action, reward, newState, done))

    
    def learn(self, entropyCoef=0.01, batchSize=32, maxEntropy=0.01):
        if len(self.batch) == 0:
            return
        
        sampledBatch = random.sample(self.batch, min(batchSize, len(self.batch)))
        
        currentStates, actions, rewards, newStates, dones = zip(*sampledBatch)

        currentStates = np.array(currentStates) 
        actions = np.array(actions)
        rewards = np.array(rewards)
        newStates = np.array(newStates)
        dones = np.array(dones)

        currentStates = torch.from_numpy(currentStates).float().to(self.device)
        actions = torch.from_numpy(actions).int().to(self.device)
        rewards = torch.from_numpy(rewards).float().to(self.device)
        newStates = torch.from_numpy(newStates).float().to(self.device)
        dones = torch.from_numpy(dones).int().to(self.device)

        rewards = (rewards - rewards.mean()) / (rewards.std() + 1e-8)
        
        self.optimizer.zero_grad()

        # Forward
        currentStateValues, probs = self.actorCritic(currentStates)
        newStateValues, _ = self.actorCritic(newStates)

        currentStateValues = currentStateValues.squeeze()
        newStateValues = newStateValues.squeeze().detach()

        # Calculate log probabilities and entropy
        dist = Categorical(probs)
        logProbs = dist.log_prob(actions)
        entropy = dist.entropy().mean()

        # Calculate targets
        targets = rewards + self.gamma * newStateValues * (1 - dones)
        delta = torch.clamp(targets - currentStateValues, -1, 1)

        # Losses
        actorLoss = -logProbs * delta.detach()
        criticLoss = delta ** 2

        # Backward
        entropyBonus = torch.clamp(entropy, max=maxEntropy)
        totalLoss = (actorLoss + criticLoss).mean() - entropyCoef * entropyBonus
        totalLoss.backward()
        self.optimizer.step()

        self.actorLossHistory.append(actorLoss.mean().item())
        self.criticLossHistory.append(criticLoss.mean().item())
        self.totalLossHistory.append(totalLoss.item())
        self.deltaHistory.append(delta.mean().item())
        self.stateValueHistory.append(currentStateValues.mean().item())
        self.entropyHistory.append(entropy.item())

        return entropy.item()

    
    def save_models(self, fileName: str=None):
        print('... saving models ...')
        if not fileName:
            torch.save(self.actorCritic.state_dict(), self.actorCritic.checkpointFile)
        else:
            torch.save(self.actorCritic.state_dict(), self.actorCritic.checkpointDir / (fileName + '.pt'))


    def load_models(self, fileName: str=None):
        print('... loading models ...')
        if not fileName:
            self.actorCritic.load_state_dict(torch.load(self.actorCritic.checkpointFile, map_location=self.device, weights_only=True))
        else:
            self.actorCritic.load_state_dict(torch.load(self.actorCritic.checkpointDir / (fileName + '.pt'), map_location=self.device, weights_only=True))

        
    def plot_all(self, avgScores=[], saveName=None, startSlice=0, endSlice=None):
        plotData = [
        ("Actor Loss", self.actorLossHistory, 'orange', None, '-'),
        ("Critic Loss", self.criticLossHistory, 'steelblue', None, '-'),
        ("Total Loss", self.totalLossHistory, 'mediumpurple', None, '-'),
        ("Delta", self.deltaHistory, 'saddlebrown', None, '-'),
        ("Avgerage Reward", avgScores, 'olivedrab', None, '-'),
        ("State Values", self.stateValueHistory, 'darkblue', None, '-'),
        ("Actions", self.actionHistory, 'crimson', '.', ''),
        ("Entropy", self.entropyHistory, 'darkorange', None, '-'),
        ]

        n = len(plotData)
        cols = 2
        rows = (n + 1) // cols

        fig, axes = plt.subplots(rows, cols, figsize=(14, 3 * rows))
        axes = axes.flatten()

        for i, (title, values, color, marker, linestyle) in enumerate(plotData):
            step = int(str(len(values))[:-3]) if len(values) > 1000 else 1
            axes[i].plot(values[startSlice:endSlice:step], label=title, color=color, marker=marker, linestyle=linestyle)
            axes[i].set_title(f"{title}")
            axes[i].set_xlabel("Batches") if title != "Actions" else axes[i].set_xlabel("Steps")
            axes[i].set_xlim()
            axes[i].set_ylabel("Value")
            axes[i].grid(True)

        # Delete empty subplots
        for j in range(i + 1, len(axes)):
            fig.delaxes(axes[j])

        plt.tight_layout()

        if saveName:
            plt.savefig(self.actorCritic.checkpointDir / f"{saveName}.png")

        plt.show()
        