import time

import numpy as np
import torch
import pyflann
import gurobipy as gp
from gurobipy import GRB
from Src.Utils import MathProg

# This .py file implements the mapping function according to Gabriel Dulac-Arnold, Richard Evans, Hado van Hasselt, Peter Sunehag, Timothy Lillicrap, Jonathan Hunt,
# Timothy Mann, Theophane Weber, Thomas Degris, and Ben Coppin. Deep reinforcement learning in large
# discrete action spaces. arXiv preprint arXiv:1512.07679, 2015.

class Knn():
    def __init__(self,
                 state_features,
                 action_dim,
                 config,
                 action_space_matrix,
                 critic):

        self.state_dim = state_features.state_dim
        self.action_dim = action_dim
        self.reduced_action_dim = action_dim
        self.config = config
        self.action_space_matrix = action_space_matrix
        self.flann = pyflann.FLANN()
        self.index = self.flann.build_index(self.action_space_matrix, algorithm='kdtree')
        self.k = config.knns
        self.feature_dims = state_features.feature_dim
        self.action_space_matrix = torch.tensor(self.action_space_matrix,dtype=torch.float32)


    def get_best_match(self, proto_action,state,critic,weights_changed,training):
        ## Obtain k nearest neighbours

        action_ids, _ = self.flann.nn_index(np.array(proto_action.tolist()[0]), self.k)
        actions = self.action_space_matrix[action_ids[0],:]
        if self.k==1:
            actions = actions.view(1, -1)

        # Obtain q-values of k nearest neighbours
        state_replications = torch.tile(state,(self.k,1))

        qvalues = critic.forward(state_replications,actions)


        ## Pick neighbour with highest q-value
        id_max_q_value = torch.argmax(qvalues).tolist()
        if len(action_ids[0].shape) == 0:
            action = int(action_ids[id_max_q_value].tolist())
        else:
            action = int(action_ids[0][id_max_q_value].tolist())

        return action