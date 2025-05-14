import torch.nn as nn
from utils.paths import BASE_DIR



class PActorCriticNet(nn.Module):
    def __init__(self, pretrainedActor, newInDim=19, newActionsNum=2,
                 name='prtr_actor_critic', checkpointDir=BASE_DIR / "tmp/actor critic"):
        super(PActorCriticNet, self).__init__()
        self.name = name
        self.checkpointDir = checkpointDir
        self.checkpointFile = self.checkpointDir / (name + '.pt')

        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(p=0.5)

        self.input_adapter = None
        if pretrainedActor.action_layer_1.in_features != newInDim:
            self.input_adapter = nn.Linear(newInDim, pretrainedActor.action_layer_1.in_features)

        # Actor layers
        self.actor_fc1 = pretrainedActor.action_layer_1
        self.actor_fc2 = pretrainedActor.action_layer_2

        # New pi layer
        self.pi = nn.Linear(pretrainedActor.action_layer_2.out_features, newActionsNum)
        self.softmax = nn.Softmax(dim=-1)

        # Add a critic
        self.v = nn.Linear(pretrainedActor.action_layer_2.out_features, 1)


    def forward(self, state):
        if self.input_adapter:
            x = self.relu(self.input_adapter(state))
        else:
            x = state

        x = self.relu(self.actor_fc1(x))
        x = self.dropout(x)
        x = self.relu(self.actor_fc2(x))
        x = self.dropout(x)

        v = self.v(x)
        pi = self.softmax(self.pi(x))
        return v, pi