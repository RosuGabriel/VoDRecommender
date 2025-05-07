import torch
import torch.optim as optim
from torch.distributions import Categorical
from actor_critic.network import ActorCriticNetwork
from actor_critic.pretrainedNet import ModifiedActorCritic
from actor_critic.pretrainedNet2 import ModifiedActorCritic2
from data_loader.movie_lens_data import MovieLensData
import matplotlib.pyplot as plt



class Agent:
    def __init__(self, alpha=0.0003, gamma=0.99, actionsNum=2, device=None, pretrainedActor=None, pretrainedCritic=None, data: MovieLensData=None):
        self.gamma = gamma
        self.actionsNum = actionsNum
        self.action = None
        self.data = data
        self.device = device or torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        self.actor_losses = []
        self.critic_losses = []
        self.total_losses = []
        self.deltas = []
        self.actions = []
        self.rewards = []
        self.avgRatings = []
        self.stateValues = []

        if pretrainedActor and pretrainedCritic:
            self.actorCritic = ModifiedActorCritic2(pretrainedActor, pretrainedCritic, newInDim=19, newActionsNum=actionsNum).to(self.device)
        elif pretrainedActor:
            self.actorCritic = ModifiedActorCritic(pretrainedActor, newInDim=19, newActionsNum=actionsNum).to(self.device)
        else:
            self.actorCritic = ActorCriticNetwork(actionsNum=actionsNum).to(self.device)
        
        self.optimizer = optim.Adam(self.actorCritic.parameters(), lr=alpha)


    def choose_action(self, observation):
        state = torch.tensor(observation, dtype=torch.float32).to(self.device)
        
        _, probs = self.actorCritic(state)

        dist = Categorical(probs)
        action = dist.sample()
        self.action = action
        
        self.actions.append(action.item())

        return action.item()
    

    def learn(self, currentState, reward, newState, done):
        currentState = torch.tensor(currentState, dtype=torch.float32).to(self.device)
        newState = torch.tensor(newState, dtype=torch.float32).to(self.device)
        reward = torch.tensor(reward, dtype=torch.float32).to(self.device)

        self.optimizer.zero_grad()

        actionEmbedding = None
        if self.data:
            movieId = self.data.moviesDf.iloc[self.action.item()]['movieId']
            actionEmbedding = self.data.get_movie_features(movieId)[:19]
            actionEmbedding = torch.tensor(actionEmbedding, dtype=torch.float32).to(self.device)

        if actionEmbedding is not None:
            currentStateValue, probs = self.actorCritic(currentState, action=actionEmbedding)
            newStateValue, _ = self.actorCritic(newState, action=actionEmbedding)
        else:
            currentStateValue, probs = self.actorCritic(currentState)
            newStateValue, _ = self.actorCritic(newState)

        currentStateValue = currentStateValue.squeeze()
        newStateValue = newStateValue.squeeze()

        dist = Categorical(probs)
        logProb = dist.log_prob(self.action)

        delta = reward + self.gamma * newStateValue * (1 - int(done)) - currentStateValue

        actorLoss = -logProb * delta
        criticLoss = delta ** 2
        totalLoss = actorLoss + criticLoss

        totalLoss.backward()
        self.optimizer.step()

        self.actor_losses.append(actorLoss.item())
        self.critic_losses.append(criticLoss.item())
        self.total_losses.append(totalLoss.item())
        self.deltas.append(delta.item())
        self.rewards.append(reward.item())
        self.stateValues.append(currentStateValue.item())

    
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

        
    def plot_all(self, saveName=None):
        data = [
        ("Actor Loss", self.actor_losses, 'orange', None, '-'),
        ("Critic Loss", self.critic_losses, 'blue', None, '-'),
        ("Total Loss", self.total_losses, 'green', None, '-'),
        ("Delta", self.deltas, 'red', None, '-'),
        ("Rewards", self.rewards, 'brown', '.', ''),
        ("State Values", self.stateValues, 'cyan', None, '-'),
        ("Actions", self.actions, 'purple', '.', '')
        ]

        n = len(data)
        cols = 2
        rows = (n + 1) // cols

        fig, axes = plt.subplots(rows, cols, figsize=(14, 3 * rows))
        axes = axes.flatten()

        stride = int(str(len(self.actor_losses))[:-3]) if len(self.actor_losses) > 1000 else 1
        
        for i, (title, values, color, marker, linestyle) in enumerate(data):
            axes[i].plot(values[::stride], label=title, color=color, marker=marker, linestyle=linestyle)
            axes[i].set_title(f"{title} evolution")
            axes[i].set_xlabel("Steps")
            axes[i].set_ylabel("Value")
            axes[i].grid(True)

        # Delete empty subplots
        for j in range(i + 1, len(axes)):
            fig.delaxes(axes[j])

        plt.tight_layout()

        if saveName:
            plt.savefig(self.actorCritic.checkpointDir / f"{saveName}.png")

        plt.show()

        