import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal
from Src.Utils.Utils import NeuralNet, pairwise_distances


# This file implements all neural network functions required to construct the actor
class Actor(NeuralNet):
    def __init__(self, state_dim, config, action_dim=None):
        super(Actor, self).__init__()

        self.config = config
        self.state_dim = state_dim
        if action_dim is None:
            self.action_dim = config.env.action_space.shape[0]
        else:
            self.action_dim = action_dim

    def init(self):
        temp, param_list = [], []
        for name, param in self.named_parameters():
            temp.append((name, param.shape))
            if 'var' in name:
                param_list.append(
                    {'params': param, 'lr': self.config.actor_lr / 100})
            else:
                param_list.append({'params': param})
        self.optim = self.config.optim(param_list, lr=self.config.actor_lr)

        print("Actor: ", temp)

class Categorical(Actor):
    def __init__(self, state_dim, config,action_space, action_dim=None):
        super(Categorical, self).__init__(state_dim, config)

        if action_dim is not None:
            self.action_dim = action_dim

        self.fc1 = nn.Linear(self.state_dim, self.action_dim)
        self.action_space = action_space
        # self.action_array = np.arange(self.action_dim)
        self.init()


    def forward(self, state):
        x = self.fc1(state)
        return x

    def get_action(self, state, training):
        x = self.forward(state)
        dist = F.softmax(x, -1)
        probs = dist.cpu().view(-1).data.numpy()
        if training:
            action_id = np.random.choice(self.action_dim, p=probs)
            action = torch.tensor([self.action_space[action_id,:]])
        else:
            action_id = np.argmax(probs)
            action = torch.tensor([self.action_space[action_id, :]])
        return action, dist

    def get_action_w_prob(self, state, explore=0):
        x = self.forward(state)
        dist = F.softmax(x, -1)

        probs = dist.cpu().view(-1).data.numpy()
        action = np.random.choice(self.action_dim, p=probs)

        return action, dist, dist.data[0][action]

    def get_prob(self, state, action):
        x = self.forward(state)
        dist = F.softmax(x, -1)
        return dist.gather(1, action), dist

    def get_prob_from_dist(self, dist, action):
        return dist.gather(1, action)

    def get_log_prob(self, state, action):
        x = self.forward(state)
        log_dist = F.log_softmax(x + 1e-5, -1)
        tmp_AT = torch.tensor(self.action_space,dtype=torch.float32)
        action = np.where(np.all(tmp_AT.numpy() == action.numpy()[0], axis=1))[0][0]
        return  log_dist.gather(1,torch.tensor([[action]])), torch.exp(log_dist)

    def get_log_prob_from_dist(self, dist, action):
        return torch.log(dist.gather(dim=1, index=action) + 1e-5)

    def get_entropy_from_dist(self, dist):
        return - torch.sum(dist * torch.log(dist + 1e-5), dim=-1)

class Categorical_deep(Actor):
    def __init__(self, state_dim, config,action_space, action_dim=None):
        super(Categorical_deep, self).__init__(state_dim, config)

        if action_dim is not None:
            self.action_dim = action_dim

        hidden = self.config.hiddenActorLayerSize
        self.fc1 = nn.Linear(state_dim, hidden)
        self.fc2 = nn.Linear(hidden, hidden)
        self.fc3 = nn.Linear(hidden, self.action_dim)
        self.relu = nn.ReLU()
        self.action_space = action_space
        self.init()


    def forward(self, state):
        out = self.fc1(state)
        out = self.relu(out)
        out = self.fc2(out)
        out = self.relu(out)
        out = self.fc3(out)
        return out

    def get_action(self, state, training):
        x = self.forward(state)
        dist = F.softmax(x, -1)
        probs = dist.cpu().view(-1).data.numpy()
        if training:
            action_id = np.random.choice(self.action_dim, p=probs)
            action = torch.tensor([self.action_space[action_id,:]])
        else:
            action_id = np.argmax(probs)
            action = torch.tensor([self.action_space[action_id, :]])
        return action, dist

    def get_action_w_prob(self, state, explore=0):
        x = self.forward(state)
        dist = F.softmax(x, -1)

        probs = dist.cpu().view(-1).data.numpy()
        action = np.random.choice(self.action_dim, p=probs)

        return action, dist, dist.data[0][action]

    def get_prob(self, state, action):
        x = self.forward(state)
        dist = F.softmax(x, -1)
        return dist.gather(1, action), dist

    def get_prob_from_dist(self, dist, action):
        return dist.gather(1, action)

    def get_log_prob(self, state, action):
        x = self.forward(state)
        log_dist = F.log_softmax(x + 1e-5, -1)
        tmp_AT = torch.tensor(self.action_space,dtype=torch.float32)
        action = np.where(np.all(tmp_AT.numpy() == action.numpy()[0], axis=1))[0][0]
        return  log_dist.gather(1,torch.tensor([[action]])), torch.exp(log_dist)

    def get_log_prob_from_dist(self, dist, action):
        return torch.log(dist.gather(dim=1, index=action) + 1e-5)

    def get_entropy_from_dist(self, dist):
        return - torch.sum(dist * torch.log(dist + 1e-5), dim=-1)

class Gaussian(Actor):
    def __init__(self, state_dim, action_dim, config):
        super(Gaussian, self).__init__(state_dim, config)
        # override super class variable
        self.action_dim = action_dim
        self.fc_mean = nn.Linear(state_dim, self.action_dim)

        if config.actor_output_layer == 'sigmoid':
            self.output_layer = torch.sigmoid
        else:
            self.output_layer = torch.tanh

        if self.config.gauss_variance > 0:
            self.forward = self.forward_wo_var
        else:
            self.fc_var = nn.Linear(state_dim, self.action_dim)
            self.forward = self.forward_with_var
        self.init()
    def forward_wo_var(self, state):
        mean = self.output_layer(self.fc_mean(state))*self.config.actor_scaling_factor_mean
        var = torch.ones_like(mean, requires_grad=False) * self.config.gauss_variance
        return mean, var

    def forward_with_var(self, state):
        mean = self.output_layer(self.fc_mean(state))*self.config.actor_scaling_factor_mean
        var  = (F.sigmoid(self.fc_var(state)) + 1e-2)*self.config.actor_scaling_factor_mean
        return mean, var

    def get_action(self, state, training):
        mean, var = self.forward(state)
        dist = Normal(mean, var)
        if training:
            action = dist.sample()
        else:
            action = mean.detach()

        return action, dist

    def get_log_prob(self, state, action):
        mean, var = self.forward(state)
        dist = Normal(mean, var)
        return dist.log_prob(action), dist

    def get_log_prob_from_dist(self, dist, action):
        return dist.log_prob(action)

    def get_prob_from_dist(self, dist, action, scalar=True):
        if scalar:
            prod = torch.exp(torch.sum(dist.log_prob(action), -1, keepdim=True))
        else:
            prod = torch.exp(dist.log_prob(action))
        return prod
    def get_entropy_from_dist(self, dist):
        return dist.entropy()


class Gaussian_deep(Actor):
    def __init__(self, state_dim, action_dim, config):
        super(Gaussian_deep, self).__init__(state_dim, config)

        self.action_dim = action_dim

        self.fc1 = nn.Linear(state_dim, self.config.hiddenActorLayerSize)
        self.fc2 = nn.Linear(self.config.hiddenActorLayerSize, self.config.hiddenActorLayerSize)
        self.fc3 = nn.Linear(self.config.hiddenActorLayerSize, self.action_dim)
        self.relu = nn.ReLU()

        if config.actor_output_layer == 'sigmoid':
            self.output_layer = torch.sigmoid
        else:
            self.output_layer = torch.tanh

        if self.config.gauss_variance > 0:
            self.forward = self.forward_wo_var
        else:
            self.fc4 = nn.Linear(state_dim, config.hiddenActorLayerSize)
            self.fc5 = nn.Linear(config.hiddenActorLayerSize, config.hiddenActorLayerSize)
            self.fc6 = nn.Linear(config.hiddenActorLayerSize, self.action_dim)
            self.forward = self.forward_with_var
        self.init()

    def forward_wo_var(self, state):
        mean = self.fc1(state)
        mean = self.relu(mean)
        mean = self.fc2(mean)
        mean = self.relu(mean)
        mean = self.fc3(mean)

        mean = self.output_layer(mean) * self.config.actor_scaling_factor_mean

        var = torch.ones_like(mean, requires_grad=False) * self.config.gauss_variance
        return mean, var

    def forward_with_var(self, state):

        mean = self.fc1(state)
        mean = self.relu(mean)
        mean = self.fc2(mean)
        mean = self.relu(mean)
        mean = self.fc3(mean)

        mean = self.output_layer(mean) * self.config.actor_scaling_factor_mean

        var = self.fc4(state)
        var = self.relu(var)
        var = self.fc5(var)
        var = self.relu(var)
        var = self.fc6(var)
        var = (F.sigmoid(var) + 1e-2)*self.config.actor_scaling_factor_mean

        return mean, var

    def get_action(self, state, training):
        mean, var = self.forward(state)
        dist = Normal(mean, var)
        if training:
            action = dist.sample()
        else:
            action = mean

        return action, dist

    def get_log_prob(self, state, action):
        mean, var = self.forward(state)
        dist = Normal(mean, var)
        return dist.log_prob(action), dist

    def get_log_prob_from_dist(self, dist, action):
        return dist.log_prob(action)

    def get_prob_from_dist(self, dist, action, scalar=True):
        if scalar:
            prod = torch.exp(torch.sum(dist.log_prob(action), -1, keepdim=True))
        else:
            prod = torch.exp(dist.log_prob(action))
        return prod
    def get_entropy_from_dist(self, dist):
        return dist.entropy()

