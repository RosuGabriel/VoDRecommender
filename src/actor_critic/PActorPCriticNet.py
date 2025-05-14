import torch.nn as nn
from utils.paths import BASE_DIR



class PActorPCriticNet(nn.Module):
    def __init__(self, pretrainedActor, pretrainedCritic, newInDim: int, newActionsNum: int,
                 name='prtr_actor_critic', checkpointDir=BASE_DIR / "tmp/actor critic"):
        super(PActorPCriticNet, self).__init__()
        self.name = name
        self.checkpointDir = checkpointDir
        self.checkpointFile = self.checkpointDir / (name + '.pt')

        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(p=0.5)

        # If original in is not same as new
        self.a_input_adapter = None
        if pretrainedActor.action_layer_1.in_features != newInDim:
            self.a_input_adapter = nn.Linear(newInDim, pretrainedActor.action_layer_1.in_features)

        self.c_input_adapter = None
        if pretrainedCritic.critic_layer_1.in_features != newInDim:
            self.c_input_adapter = nn.Linear(newInDim, pretrainedCritic.critic_layer_1.in_features)    

        # Actor layers
        self.actor_fc1 = pretrainedActor.action_layer_1
        self.actor_fc2 = pretrainedActor.action_layer_2

        self.actor_output = nn.Linear(pretrainedActor.action_layer_2.out_features, newActionsNum)

        self.action_adapter = nn.Linear(newActionsNum, pretrainedCritic.critic_layer_1.out_features)

        self.softmax = nn.Softmax(dim=-1)

        # Critic network
        self.critic = pretrainedCritic  # input state + action


    def forward(self, state, action=None):
        # Adapt actor input
        if self.a_input_adapter:
            x = self.relu(self.a_input_adapter(state))
        else:
            x = state

        # Actor forward
        x = self.relu(self.actor_fc1(x))
        x = self.dropout(x)
        x = self.relu(self.actor_fc2(x))
        x = self.dropout(x)

        pi_logits = self.actor_output(x)
        pi = self.softmax(pi_logits)

        # Adapt critic input
        if self.c_input_adapter:
            state = self.relu(self.c_input_adapter(state))
        
        v = None
        if action is None:
            action = pi

        if len(state.shape) == 1:
            state = state.unsqueeze(0)
        if len(action.shape) == 1:
            action = action.unsqueeze(0)

        action = self.action_adapter(action)

        # Value from critic
        v = self.critic(state, action)

        return v, pi
