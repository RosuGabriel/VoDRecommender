import torch.nn as nn
from utils.paths import BASE_DIR
import torch
from torch.distributions import Normal



class PretrainedContinuousActorCriticNet(nn.Module):
    def __init__(self, pretrainedActor, pretrainedCritic, newInDim: int, newActionsNum: int,
                 name='prtr_actor_critic', checkpointDir=BASE_DIR / "tmp/models checkpoint"):
        super(PretrainedContinuousActorCriticNet, self).__init__()
        self.name = name
        self.checkpointDir = checkpointDir
        self.checkpointFile = self.checkpointDir / (name + '.pt')
        self.actor_output = None

        self.relu = nn.ReLU()
        self.tanh = nn.Tanh()
        self.softplus = nn.Softplus()

        # Input adapters for actor and critic
        self.a_input_adapter = None
        if pretrainedActor.action_layer_1.in_features != newInDim:
            self.a_input_adapter = nn.Linear(newInDim, pretrainedActor.action_layer_1.in_features)

        self.c_input_adapter = None
        if pretrainedCritic.critic_layer_1.in_features != newInDim:
            self.c_input_adapter = nn.Linear(newInDim, pretrainedCritic.critic_layer_1.in_features)    

        # Actor layers
        self.actor_fc1 = pretrainedActor.action_layer_1
        self.actor_fc2 = pretrainedActor.action_layer_2

        self.mean1 = nn.Linear(pretrainedActor.action_layer_2.out_features, pretrainedActor.action_layer_2.out_features//2)
        self.std1 = nn.Linear(pretrainedActor.action_layer_2.out_features, pretrainedActor.action_layer_2.out_features//2)

        self.mean2 = nn.Linear(pretrainedActor.action_layer_2.out_features//2, newActionsNum)
        self.std2 = nn.Linear(pretrainedActor.action_layer_2.out_features//2, newActionsNum)

        self.action_adapter = nn.Linear(newInDim, pretrainedCritic.critic_layer_1.out_features)
        self.pi_adapter = nn.Linear(newActionsNum, pretrainedCritic.critic_layer_1.out_features)

        # Critic network
        self.critic = pretrainedCritic


    def forward(self, state, action=None, justAction=False):
        # Adapt actor input
        if self.a_input_adapter:
            x = self.a_input_adapter(state)
            x = self.relu(x)
        else:
            x = state
        
        # Actor forward
        x = self.relu(self.actor_fc1(x))
        x = self.relu(self.actor_fc2(x))

        # Get mean and std for action distribution
        mu = self.relu(self.mean1(x))
        mu = self.tanh(self.mean2(mu)) * 5

        std = self.relu(self.std1(x))
        std = self.softplus(self.std2(std)) + 1e-6

        # Adapt critic input
        if self.c_input_adapter:
            state = self.relu(self.c_input_adapter(state))
        
        v = None

        # Sample action from distribution
        if action is None:
            dist = Normal(mu, std)
            action = dist.rsample()

        if justAction:
            return action 
        
        if len(action.shape) == 1:
            actionForCritic = action.unsqueeze(0)
        actionForCritic = self.action_adapter(action)

        if len(state.shape) == 1:
            state = state.unsqueeze(0)

        # Value from critic
        v = self.critic(state, actionForCritic)

        return v, action, mu, std
