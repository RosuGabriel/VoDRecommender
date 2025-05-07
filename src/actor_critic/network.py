import torch.nn as nn
from utils.paths import BASE_DIR



class ActorCriticNetwork(nn.Module):
    def __init__(self, actionsNum, inDim=19, fc1Dims=1024, fc2Dims=512,
                 name='actor_critic', checkpointDir=BASE_DIR / "tmp/actor critic"):
        super(ActorCriticNetwork, self).__init__()
        self.model_name = name
        self.checkpointDir = checkpointDir
        self.checkpointFile = self.checkpointDir / (name + '.pt')
        
        self.fc1Dims = fc1Dims
        self.fc2Dims = fc2Dims
        self.actionsNum = actionsNum

        self.fc1 = nn.Linear(inDim, fc1Dims)
        self.fc2 = nn.Linear(fc1Dims, fc2Dims)
        
        self.relu = nn.ReLU()
        self.v = nn.Linear(fc2Dims, 1)

        self.pi = nn.Linear(fc2Dims, actionsNum)
        self.softmax = nn.Softmax(dim=-1)


    def forward(self, state):
        x = self.relu(self.fc1(state))
        x = self.relu(self.fc2(x))
        
        v = self.v(x)
        pi = self.softmax(self.pi(x))

        return v, pi
