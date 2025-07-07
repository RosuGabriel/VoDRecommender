# Imports
import torch
import torch.optim as optim
from torch.distributions import Categorical
from models.ActorCriticNet import ActorCriticNet
import matplotlib.pyplot as plt
import numpy as np
import random



class Agent:
    def __init__(self, alpha=0.0001, beta=None, gamma=0.99, actionsNum=2, observationDim=19, device=None, pretrainedActor=None,
                 pretrainedCritic=None, batchSize=32, usePiForCritic=False):
        self.gamma = gamma
        self.actionsNum = actionsNum
        self.action = None
        self.batchSize = batchSize
        self.batch = []
        self.device = device or torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.separateOptimizers = False
        self.usePiForCritic = usePiForCritic

        self.actorLossHistory = []
        self.totalLossHistory = []
        self.criticLossHistory = []
        self.deltaHistory = []
        self.actionHistory = []
        self.rewardHistory = []
        self.avgRatingHistory = []
        self.stateValueHistory = []
        self.entropyHistory = []
        
        # Actor-Critic network
        self.actorCritic = ActorCriticNet(pretrainedActor, pretrainedCritic, newInDim=observationDim, newActionsNum=actionsNum).to(self.device)
     
        # Optimizer type
        if beta is None:
            # Shared optimizer
            self.optimizer = torch.optim.Adam(self.actorCritic.parameters(), lr=alpha)
        else:
            self.separateOptimizers = True

            # Actor parameters
            self.actorParameters = []
            if self.actorCritic.a_input_adapter:
                self.actorParameters += list(self.actorCritic.a_input_adapter.parameters())

            self.actorParameters += list(self.actorCritic.actor_fc1.parameters())
            self.actorParameters += list(self.actorCritic.actor_fc2.parameters())
            self.actorParameters += list(self.actorCritic.actor_output.parameters())

            # Critic parameters
            self.criticParameters = []
            if self.actorCritic.c_input_adapter:
                self.criticParameters += list(self.actorCritic.c_input_adapter.parameters())

            self.criticParameters += list(self.actorCritic.action_adapter.parameters())
            self.criticParameters += list(self.actorCritic.critic.parameters())

            # Optimizer for each component
            self.actorOptimizer = optim.Adam(self.actorParameters, lr=alpha)
            self.criticOptimizer = optim.Adam(self.criticParameters, lr=beta)


    def choose_action(self, observation, temperature: float=1.0):
        state = torch.tensor(observation, dtype=torch.float32).to(self.device)
        
        _, probs = self.actorCritic(state, justActor=True)

        logits = torch.log(probs + 1e-8)  # avoiding log(0)
        
        scaledLogits = logits / temperature
        dist = Categorical(logits=scaledLogits)

        action = dist.sample()

        self.action = action
        
        self.actionHistory.append(action.item())

        return action.item(), probs if temperature == 1.0 else scaledLogits


    def add_to_batch(self, currentState, actionIndex, actionEmbedding, reward, newState, newActionEmbedding, done):
        if len(self.batch) >= self.batchSize:
            idx = random.randint(0, int(len(self.batch)/2))
            self.batch.pop(idx)
        self.batch.append((currentState, actionIndex, actionEmbedding, reward, newState, newActionEmbedding, done))


    def sample_batch(self, batchSampleSize, recentExperienceRatio):
        batchSampleSize = min(batchSampleSize, len(self.batch))
        recentSize = int(batchSampleSize * recentExperienceRatio)
        oldSize = batchSampleSize - recentSize

        # Take recent experience
        recentBatch = self.batch[-recentSize:] if recentSize > 0 else []

        # Take randomly old experience
        oldCandidates = self.batch[:-recentSize] if recentSize > 0 else self.batch
        oldBatch = random.sample(oldCandidates, oldSize) if oldSize > 0 else []

        sampledBatch = oldBatch + recentBatch
        random.shuffle(sampledBatch)
        currentStates, actionIndexes, actionEmbeddings, rewards, newStates, newActionEmbeddings, dones = zip(*sampledBatch)
        return currentStates, actionIndexes, actionEmbeddings, rewards, newStates, newActionEmbeddings, dones

    
    def learn(self, entropyCoef=0.01, batchSampleSize=32, maxEntropy=10, recentExperienceRatio=0.25):
        if len(self.batch) == 0:
            return

        currentStates, actionIndexes, actionEmbeddings, rewards, newStates, newActionEmbeddings, dones = self.sample_batch(batchSampleSize=batchSampleSize, recentExperienceRatio=recentExperienceRatio)

        currentStates = np.array(currentStates) 
        actionIndexes = np.array(actionIndexes)
        rewards = np.array(rewards)
        newStates = np.array(newStates)
        dones = np.array(dones)

        currentStates = torch.from_numpy(currentStates).float().to(self.device)
        actionIndexes = torch.from_numpy(actionIndexes).float().to(self.device)
        rewards = torch.from_numpy(rewards).float().to(self.device)
        newStates = torch.from_numpy(newStates).float().to(self.device)
        dones = torch.from_numpy(dones).float().to(self.device)

        if not self.usePiForCritic:
            actionEmbeddings = np.array(actionEmbeddings)
            newActionEmbeddings = np.array(newActionEmbeddings)

            actionEmbeddings = torch.from_numpy(actionEmbeddings).float().to(self.device)
            newActionEmbeddings = torch.from_numpy(newActionEmbeddings).float().to(self.device)

        # State values
        if not self.usePiForCritic:
            currentStateValues, probs = self.actorCritic(currentStates, actionEmbeddings)
            newStateValues, _ = self.actorCritic(newStates, newActionEmbeddings)
        else:
            currentStateValues, probs = self.actorCritic(currentStates)
            newStateValues, _ = self.actorCritic(newStates)

        currentStateValues = currentStateValues.squeeze()
        newStateValues = newStateValues.squeeze().detach()

        # Calculate log probabilities and entropy
        dist = Categorical(probs)
        logProbs = dist.log_prob(actionIndexes)
        entropy = dist.entropy().mean()

        # Calculate targets and delta
        targets = rewards + self.gamma * newStateValues * (1 - dones)
        delta = targets - currentStateValues
        
        # Losses
        criticLoss = (delta ** 2).mean()
        actorLoss = (-logProbs * delta.detach()).mean()

        # Entropy bonus on actor loss
        entropyBonus = torch.clamp(entropy, max=maxEntropy)
        actorLoss = actorLoss - entropyCoef * entropyBonus

        totalLoss = criticLoss + actorLoss

        # Backward
        if self.separateOptimizers:
            self.criticOptimizer.zero_grad()
            criticLoss.backward(retain_graph=True)
            torch.nn.utils.clip_grad_norm_(self.criticParameters, max_norm=0.5)
            self.criticOptimizer.step()

            self.actorOptimizer.zero_grad()
            actorLoss.backward()
            torch.nn.utils.clip_grad_norm_(self.actorParameters, max_norm=0.5)
            self.actorOptimizer.step()
        else:
            self.optimizer.zero_grad()
            totalLoss.backward()
            torch.nn.utils.clip_grad_norm_(self.actorCritic.parameters(), max_norm=0.5)
            self.optimizer.step()

        # Add to history for plot
        self.actorLossHistory.append(actorLoss.item())
        self.totalLossHistory.append(totalLoss.item())
        self.criticLossHistory.append(criticLoss.item())
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
        ("Total Loss", self.totalLossHistory, 'darkviolet', None, '-'),
        ("Actor Loss", self.actorLossHistory, 'mediumpurple', None, '-'),
        ("Critic Loss", self.criticLossHistory, 'steelblue', None, '-'),
        ("Delta", self.deltaHistory, 'saddlebrown', None, '-'),
        ("Avgerage Reward", avgScores, 'olivedrab', None, '-'),
        ("State Values", self.stateValueHistory, 'darkblue', None, '-'),
        ("Entropy", self.entropyHistory, 'darkorange', None, '-'),
        ("Actions", self.actionHistory, 'crimson', '.', ''),
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
        