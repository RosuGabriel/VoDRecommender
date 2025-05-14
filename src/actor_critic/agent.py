import torch
import torch.optim as optim
from torch.distributions import Categorical
from actor_critic.ActorCriticNet import ActorCriticNet
from actor_critic.PActorCriticNet import PActorCriticNet
from actor_critic.PActorPCriticNet import PActorPCriticNet
import matplotlib.pyplot as plt
import numpy as np



class Agent:
    def __init__(self, alpha=0.0003, gamma=0.99, actionsNum=2, device=None, pretrainedActor=None, pretrainedCritic=None):
        self.gamma = gamma
        self.actionsNum = actionsNum
        self.action = None
        self.prevAction = None
        self.batch = []
        self.device = device or torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        self.actorLosses = []
        self.criticLosses = []
        self.totalLosses = []
        self.deltas = []
        self.actions = []
        self.batchActions = []
        self.rewards = []
        self.avgRatings = []
        self.stateValues = []

        if pretrainedActor and pretrainedCritic:
            self.actorCritic = PActorPCriticNet(pretrainedActor, pretrainedCritic, newInDim=19, newActionsNum=actionsNum).to(self.device)
        elif pretrainedActor:
            self.actorCritic = PActorCriticNet(pretrainedActor, newInDim=19, newActionsNum=actionsNum).to(self.device)
        else:
            self.actorCritic = ActorCriticNet(actionsNum=actionsNum).to(self.device)
        
        self.optimizer = optim.Adam(self.actorCritic.parameters(), lr=alpha)


    def choose_action(self, observation):
        state = torch.tensor(observation, dtype=torch.float32).to(self.device)
        
        _, probs = self.actorCritic(state)

        dist = Categorical(probs)
        action = dist.sample()

        self.action = action
        
        self.actions.append(action.item())

        return action.item()
    

    def learn(self, currentState, reward, newState, done, entropyCoef = 0.01):
        currentState = torch.tensor(currentState, dtype=torch.float32).to(self.device)
        newState = torch.tensor(newState, dtype=torch.float32).to(self.device)
        reward = torch.tensor(reward, dtype=torch.float32).to(self.device)

        self.optimizer.zero_grad()

        currentStateValue, probs = self.actorCritic(currentState)
        newStateValue, _ = self.actorCritic(newState)

        currentStateValue = currentStateValue.squeeze()
        newStateValue = newStateValue.squeeze().detach()

        dist = Categorical(probs)
        logProb = dist.log_prob(self.action)
        entropy = dist.entropy()

        if self.prevAction == self.action:
            reward += torch.tensor(-0.5, dtype=torch.float32).to(self.device)
        
        self.prevAction = self.action

        delta = reward + self.gamma * newStateValue * (1 - int(done)) - currentStateValue

        actorLoss = -logProb * delta - entropyCoef * entropy
        criticLoss = delta ** 2
        totalLoss = actorLoss + criticLoss

        totalLoss.backward()
        self.optimizer.step()

        self.actorLosses.append(actorLoss.item())
        self.criticLosses.append(criticLoss.item())
        self.totalLosses.append(totalLoss.item())
        self.deltas.append(delta.item())
        self.rewards.append(reward.item())
        self.stateValues.append(currentStateValue.item())

    
    def add_to_batch(self, currentState, action, reward, newState, done):
        if self.prevAction == action:
            reward += -0.5
        self.batch.append((currentState, action, reward, newState, done))
        self.prevAction = action

    
    def learn_from_batch(self, entropyCoef=0.01):
        if len(self.batch) == 0:
            return
        
        currentStates, actions, rewards, newStates, dones = zip(*self.batch)

        currentStates = torch.tensor(currentStates, dtype=torch.float32).to(self.device)
        actions = torch.tensor(actions, dtype=torch.long).to(self.device)
        rewards = torch.tensor(rewards, dtype=torch.float32).to(self.device)
        newStates = torch.tensor(newStates, dtype=torch.float32).to(self.device)
        dones = torch.tensor(dones, dtype=torch.float32).to(self.device)
        
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
        delta = targets - currentStateValues

        # Losses
        actorLoss = -logProbs * delta.detach() - entropyCoef * entropy
        criticLoss = delta ** 2

        # Backward
        totalLoss = (actorLoss + criticLoss).mean()
        totalLoss.backward()
        self.optimizer.step()

        self.batch = []

        self.actorLosses.append(actorLoss.mean().item())
        self.criticLosses.append(criticLoss.mean().item())
        self.totalLosses.append(totalLoss.item())
        self.deltas.append(delta.mean().item())
        self.rewards.append(rewards.mean().item())
        self.stateValues.append(currentStateValues.mean().item())
        self.batchActions.append(actions.float().mean().item())

    
    def save_models(self, fileName: str=None):
        print('... saving models ...')
        if not fileName:
            torch.save(self.actorCritic.state_dict(), self.actorCritic.checkpointFile)
        else:
            torch.save(self.actorCritic.state_dict(), self.actorCritic.checkpointDir / (fileName + '.pt'))


    def load_models(self, fileName: str=None):
        print('... loading models ...')
        if not fileName:
            self.actorCritic.load_state_dict(torch.load(self.actorCritic.checkpointFile, map_location=self.device))
        else:
            self.actorCritic.load_state_dict(torch.load(self.actorCritic.checkpointDir / (fileName + '.pt'), map_location=self.device))

        
    def plot_all(self, avgScores=[], saveName=None, startSlice=0, endSlice=None):
        plotData = [
        ("Actor Loss", self.actorLosses, 'orange', None, '-'),
        ("Critic Loss", self.criticLosses, 'steelblue', None, '-'),
        ("Total Loss", self.totalLosses, 'mediumpurple', None, '-'),
        ("Delta", self.deltas, 'saddlebrown', None, '-'),
        ("Avgerage Score", avgScores, 'olivedrab', None, '-'),
        ("State Values", self.stateValues, 'darkblue', None, '-'),
        ("Rewards", self.rewards, 'slategray', '.', ''),
        ("Actions", self.actions, 'crimson', '.', '')
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
            axes[i].set_xlabel("Episodes") if title != "Actions" else axes[i].set_xlabel("Steps")
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
        