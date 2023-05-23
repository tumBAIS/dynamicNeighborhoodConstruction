import numpy as np
import torch
from torch import tensor, float32
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
from Src.Utils.Utils import NeuralNet
import itertools


def get_Basis(config):
        if config.fourier_order > 0:
            return Fourier_Basis(config=config)
        else:
            return Basis(config=config)


class Basis(NeuralNet):
    def __init__(self,  config):
        super(Basis, self).__init__()

        self.config = config

        # Variables for normalizing state features
        self.state_low = tensor(config.env.observation_space.low, dtype=float32, requires_grad=False)
        self.state_high = tensor(config.env.observation_space.high, dtype=float32, requires_grad=False)
        self.state_diff = self.state_high - self.state_low
        self.state_dim = len(self.state_low)
        self.feature_dim = self.state_dim
        self.flag = (self.state_diff > 1e3).any().item()  # Flag to Normalize or not
        self.forward = self.preprocess

        print("State Low: {} :: State High: {}".format(self.state_low, self.state_high))

    def init(self):
        print("State features: ", [(m, p.shape) for m, p in self.named_parameters()])
        self.optim = self.config.optim(self.parameters(), lr=self.config.state_lr)

    def preprocess(self, state):
        if self.flag:
            return state
        else:
            return (state - self.state_low)/self.state_diff


class Fourier_Basis(Basis):
    def __init__(self,  config):
        super(Fourier_Basis, self).__init__(config)

        dim = self.state_dim
        order = self.config.fourier_order

        if self.config.fourier_coupled:
            if (order+1)**dim > 1000:
                raise ValueError("Reduce Fourier order please... ")

            coeff = np.arange(0, order+1)
            weights = torch.from_numpy(np.array(list(itertools.product(coeff, repeat=dim))).T)  # size = n**d
            self.get_basis = self.coupled
            self.feature_dim = weights.shape[-1]
        else:
            weights = torch.from_numpy(np.arange(1, order + 1))
            self.get_basis = self.uncoupled
            self.feature_dim = weights.shape[-1] * dim

        self.forward = self.forward_wo_state_trafo
        self.basis_weights = weights.type(torch.FloatTensor).requires_grad_(False)
        self.dummy_param = torch.nn.Parameter(torch.rand(1).type(torch.FloatTensor))
        self.init()

    def coupled(self, x):
        # Creates a cosine only basis having order^(dim) terms
        basis = torch.matmul(x, self.basis_weights)
        basis = torch.cos(basis * np.pi)
        return basis

    def uncoupled(self, x):
        x = x.unsqueeze(2)  # convert shape from r*c to r*c*1
        basis = x * self.basis_weights  # Broadcast multiplication r*c*1 x 1*d => r*c*d
        basis = torch.cos(basis * np.pi)
        return basis.view(x.shape[0], -1)  # convert shape from r*c*d => r*(c*d)

    def forward_wo_state_trafo(self, state):
        x = self.preprocess(state)
        return self.get_basis(x)
    def forward_w_state_trafo(self, state):
        if state.numpy().shape[0] > 1:
            state = torch.tensor(self.tf_idf_matrix[state.numpy()[:,0].astype(int), :], dtype=torch.float32)
        else:
            state = torch.tensor(self.tf_idf_matrix[int(state.numpy()[0][0]),:],dtype=torch.float32)
            state = state[None]
        x = self.preprocess(state)
        return x

class NN_Basis(Basis):
    def __init__(self, config):
        super(NN_Basis, self).__init__(config)

        self.feature_dim = self.config.feature_dim[-1]
        layers = []
        dims = [self.state_dim]
        dims.extend(self.config.feature_dim)

    def forward(self, state):
        return



