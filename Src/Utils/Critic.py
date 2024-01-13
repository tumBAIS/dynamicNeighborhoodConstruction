import torch
import torch.nn as nn
from Src.Utils.Utils import NeuralNet

# This file implements all neural network functions required to construct the critic
class Base_Critic(NeuralNet):
    def __init__(self, state_dim, config):
        super(Base_Critic, self).__init__()
        self.config = config
        self.state_dim = state_dim

    def init(self):
        print("Critic: ", [(name, param.shape) for name, param in self.named_parameters()])
        self.optim = self.config.optim(self.parameters(), lr=self.config.critic_lr)

class Critic(Base_Critic):
    def __init__(self, state_dim, config):
        super(Critic, self).__init__(state_dim, config)

        self.fc1 = nn.Linear(state_dim, 1)
        self.init()

    def forward(self, x):
        x = self.fc1(x)
        return x

class Qval(Base_Critic):
    def __init__(self, state_dim, action_dim, config):
        super(Qval, self).__init__(state_dim, config)
        self.fc1 = nn.Linear(state_dim, self.config.hiddenLayerSize)
        self.fc2 = nn.Linear(self.config.hiddenLayerSize+action_dim, self.config.hiddenLayerSize)
        self.fc3 = nn.Linear(self.config.hiddenLayerSize, 1)
        self.relu = nn.ReLU()
        self.init()

    def forward(self, x,y):
        out = self.fc1(x)
        out = self.relu(out)
        combined = torch.cat([out, y], len(y.shape) - 1)
        out = self.fc2(combined)
        out = self.relu(out)
        out = self.fc3(out)
        # out = self.relu(out)
        return out

class Qval_sequential(Base_Critic):
    def __init__(self, state_dim, action_dim, config):
        super(Qval_sequential, self).__init__(state_dim, config)
        self.nn_model = nn.Sequential(
            nn.Linear(state_dim+action_dim, self.config.hiddenLayerSize, bias=True),
            nn.ReLU(),
            nn.Linear(self.config.hiddenLayerSize, self.config.hiddenLayerSize, bias=True),
            nn.ReLU(),
            nn.Linear(self.config.hiddenLayerSize, 1, bias=True)
            # nn.ReLU()
        )
        self.hiddenLayerSize =  self.config.hiddenLayerSize
        self.init()
    def forward(self,x,y):
        nn_input = torch.cat([x, y], 1)
        return self.nn_model.forward(nn_input)

