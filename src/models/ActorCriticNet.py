import torch.nn as nn
from models.Actor import Actor
from models.Critic import Critic
from utils.paths import BASE_DIR



class ActorCriticNet(nn.Module):
    def __init__(self, stateDim: int, actionDim: int, hiddenSize: int,
                 name='prtr_actor_critic', checkpointDir=BASE_DIR / "tmp/models checkpoint"):
        super(ActorCriticNet, self).__init__()
        self.name = name
        self.checkpointDir = checkpointDir
        self.checkpointFile = self.checkpointDir / (name + '.pt')

        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(p=0.5)

        # Instantiate networks
        self.actor = Actor(input_dim=stateDim, output_dim=actionDim, hidden_size=hiddenSize)
        self.critic = Critic(input_dim=stateDim, output_dim=actionDim, hidden_size=hiddenSize)

        self.softmax = nn.Softmax(dim=-1)


    def forward(self, state, action=None, justActor=False):
        # Actor forward
        pi_logits = self.actor(state)
        pi = self.softmax(pi_logits)

        if justActor:
            return None, pi

        v = None
        if action is None:
            action = pi
            if len(action.shape) == 1:
                action = action.unsqueeze(0)
        else:
            if len(action.shape) == 1:
                action = action.unsqueeze(0)

        if len(state.shape) == 1:
            state = state.unsqueeze(0)

        # Value from critic
        v = self.critic(state, action)

        return v, pi
