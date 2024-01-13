import numpy as np
import torch
from torch import float32
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
from Src.Utils.Utils import NeuralNet, pairwise_distances
from Src.Utils import Basis


# This function implements the action representation according to Yash Chandak, Georgios Theocharous, James Kostas, Scott Jordan, and Philip Thomas. Learning action
# representations for reinforcement learning. In International conference on machine learning, pages 941–950.
# PMLR, 2019.

class Action_representation(NeuralNet):
    def __init__(self,
                 state_dim,
                 action_dim,
                 config):
        super(Action_representation, self).__init__()

        self.state_dim = state_dim
        self.action_dim = action_dim
        self.config = config
        self.norm_const = np.log(self.action_dim)

        # Action embeddings to project the predicted action into original dimensions
        if config.true_embeddings:
            embeddings = config.env.get_embeddings()
            self.reduced_action_dim = np.shape(embeddings)[1]
            maxi, mini = np.max(embeddings), np.min(embeddings)
            embeddings = ((embeddings - mini)/(maxi-mini))*2 - 1  # Normalize to (-1, 1)

            self.embeddings = Variable(torch.from_numpy(embeddings).type(float32), requires_grad=False)
        else:
            self.reduced_action_dim = config.reduced_action_dim
            if self.config.load_embed:
                try:
                    init_tensor = torch.load(self.config.paths['embedding'])['embeddings']
                except KeyError:
                    init_tensor = torch.load(self.config.paths['embedding'])['embeddings.weight']
                assert init_tensor.shape == (self.action_dim, self.reduced_action_dim)
                print("embeddings successfully loaded from: ", self.config.paths['embedding'])
            else:
                init_tensor = torch.rand(self.action_dim, self.reduced_action_dim)*2 - 1   # Don't initialize near the extremes.

            self.embeddings = torch.nn.Parameter(init_tensor.type(float32), requires_grad=True)

        # One layer neural net to get action representation
        self.fc1 = nn.Linear(self.state_dim*2, self.reduced_action_dim)

        print("Action representation: ", [(name, param.shape) for name, param in self.named_parameters()])
        self.optim = config.optim(self.parameters(), lr=self.config.embed_lr)

    def get_match_scores(self, action):
        # compute similarity probability based on L2 norm
        embeddings = self.embeddings
        if not self.config.true_embeddings:
            embeddings = F.tanh(embeddings)

        # compute similarity probability based on L2 norm
        similarity = - pairwise_distances(action, embeddings)  # Negate euclidean to convert diff into similarity score

        return similarity

    def get_best_match(self, action,state,critic,weights_changed=True,training=True):
        similarity = self.get_match_scores(action)
        val, pos = torch.max(similarity, dim=1)
        return pos.cpu().item() #data.numpy()[0]

    def get_embedding(self, action):
        # Get the corresponding target embedding
        action_emb = self.embeddings[action]
        if not self.config.true_embeddings:
            action_emb = F.tanh(action_emb)
        return action_emb

    def forward(self, state1, state2):
        # concatenate the state features and predict the action required to go from state1 to state2
        state_cat = torch.cat([state1, state2], dim=1)
        x = F.tanh(self.fc1(state_cat))
        return x

    def unsupervised_loss(self, s1, a, s2, normalized=True):
        x = self.forward(s1, s2)
        similarity = self.get_match_scores(x)  # Negative euclidean
        if normalized:
            loss = F.cross_entropy(similarity, a, size_average=True)/self.norm_const \
                   + self.config.emb_reg * torch.pow(self.embeddings, 2).mean()/self.reduced_action_dim
        else:
            loss = F.cross_entropy(similarity, a, size_average=True) \
                   + self.config.emb_reg * torch.pow(self.embeddings, 2).mean()
        return loss



class Action_representation_deep(NeuralNet):
    def __init__(self,state_dim, action_dim, config):
        super(Action_representation_deep, self).__init__()

        self.state_dim = state_dim
        self.action_dim = action_dim
        self.config = config
        self.norm_const = np.log(self.action_dim)

        # Action embeddings to project the predicted action into original dimensions
        if config.true_embeddings:
            embeddings = config.env.get_embeddings()
            self.reduced_action_dim = np.shape(embeddings)[1]
            maxi, mini = np.max(embeddings), np.min(embeddings)
            embeddings = ((embeddings - mini)/(maxi-mini))*2 - 1  # Normalize to (-1, 1)

            self.embeddings = Variable(torch.from_numpy(embeddings).type(float32), requires_grad=False)
        else:
            self.reduced_action_dim = config.reduced_action_dim
            if self.config.load_embed:
                try:
                    init_tensor = torch.load(self.config.paths['embedding'])['embeddings']
                except KeyError:
                    init_tensor = torch.load(self.config.paths['embedding'])['embeddings.weight']
                assert init_tensor.shape == (self.action_dim, self.reduced_action_dim)
                print("embeddings successfully loaded from: ", self.config.paths['embedding'])
            else:
                init_tensor = torch.rand(self.action_dim, self.reduced_action_dim)*2 - 1   # Don't initialize near the extremes.

            self.embeddings = torch.nn.Parameter(init_tensor.type(float32), requires_grad=True)

        # Deep representation
        self.fc1 = nn.Linear(self.state_dim*2,128)
        self.fc2 = nn.Linear(128,64)
        self.fc3 = nn.Linear(64,self.reduced_action_dim)
        self.relu = nn.ReLU()


        print("Action representation: ", [(name, param.shape) for name, param in self.named_parameters()])
        self.optim = config.optim(self.parameters(), lr=self.config.embed_lr)

    def get_match_scores(self, action):
        # compute similarity probability based on L2 norm
        embeddings = self.embeddings
        if not self.config.true_embeddings:
            embeddings = F.tanh(embeddings)

        # compute similarity probability based on L2 norm
        similarity = - pairwise_distances(action, embeddings)  # Negate euclidean to convert diff into similarity score

        # compute similarity probability based on dot product
        # similarity = torch.mm(action, torch.transpose(embeddings, 0, 1))  # Dot product

        return similarity

    def get_best_match(self, action,state,critic,weights_changed=True,training=True):
        similarity = self.get_match_scores(action)
        val, pos = torch.max(similarity, dim=1)
        return pos.cpu().item() #data.numpy()[0]

    def get_embedding(self, action):
        # Get the corresponding target embedding
        action_emb = self.embeddings[action]
        if not self.config.true_embeddings:
            action_emb = F.tanh(action_emb)
        return action_emb

    def forward(self, state1, state2):
        # concatenate the state features and predict the action required to go from state1 to state2
        # state1 = super(Action_representation_deep, self).forward(state1)
        # state2 = super(Action_representation_deep, self).forward(state2)

        state_cat = torch.cat([state1, state2], dim=1)
        x = self.fc1(state_cat)
        x = self.relu(x)
        x = self.fc2(x)
        x = self.relu(x)
        x = self.fc3(x)
        x = F.tanh(x)
        # x = self.fc1(state_cat)
        # x = F.tanh(self.fc2(x))
        return x

    def unsupervised_loss(self, s1, a, s2,normalized=True):
        x = self.forward(s1, s2)
        similarity = self.get_match_scores(x)  # Negative euclidean
        # loss = F.cross_entropy(similarity, a, size_average=True)/self.norm_const \
        #        + self.config.emb_reg * torch.pow(self.embeddings, 2).mean()/self.reduced_action_dim
        if normalized:
            loss = F.cross_entropy(similarity, a, size_average=True)/self.norm_const \
                   + self.config.emb_reg * torch.pow(self.embeddings, 2).mean()/self.reduced_action_dim
        else:
            loss = F.cross_entropy(similarity, a, size_average=True) \
                   + self.config.emb_reg * torch.pow(self.embeddings, 2).mean()
        return loss

# If
class No_Action_representation(NeuralNet):
    def __init__(self,
                 state_dim,
                 action_dim,
                 config,
                 action_space):
        super(No_Action_representation, self).__init__()

        self.state_dim = state_dim
        self.action_dim = action_dim
        self.action_space = action_space
        self.config = config
        self.norm_const = np.log(self.action_dim)
        self.reduced_action_dim = action_dim

    def get_best_match(self, action,state,critic,weights_changed=True,training=True):
        action = np.where(np.all(self.action_space==action.numpy()[0],axis=1))[0][0]
        return action



