import torch.nn as nn



class Actor(nn.module):
    def __init__(self, dim):
        super().__init__()
        self.fc = nn.Sequential(
            nn.Linear(dim, 128),
            nn.ReLU(),
            nn.Linear(128, 1),
            nn.Tanh()
        )

    def forward(self, users, movies):
        pass


class Critic(nn.module):
    def __init__(self, userDim):
        super().__init__()
        self.fc = nn.Sequential(
            nn.Linear(userDim, 32),
            nn.ReLU(),
            nn.Linear(32, 1)
        )

    def forward(self, users):
        pass