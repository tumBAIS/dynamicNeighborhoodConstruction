import numpy as np
import torch
import pyflann

# This file implements Dynamic Neighborhood Construction (DNC), the main contribution of the paper "Handling Large Discrete Action Spaces via Dynamic Neighborhood Construction"
class DNC():
    def __init__(self,
                 state_features,
                 action_dim,
                 config,
                 action_space_matrix):

        # Common attributes
        self.state_dim = state_features.state_dim
        self.action_dim = action_dim
        self.reduced_action_dim = action_dim
        self.config = config
        self.feature_dims = state_features.feature_dim


        # Attributes required for the base action generation
        self.clipmin = self.config.env.action_space.low[0]
        self.clipmax = self.config.env.action_space.high[0]
        self.scaler = lambda a, amin, amax, smax, smin: (a - amin) / (amax-amin) * (smax - smin) + smin
        self.outputLayerLimit = config.a_clip
        

        # Attributes required for the neighborhood generation
        self.maximum_greedy_search_steps = config.maximum_greedy_search_steps
        self.clipped_decimals = config.clipped_decimals
        self.perturb_scaler = config.perturb_scaler
        self.perturbation_range = config.perturbation_range
        self.perturbation_array = []
        tmp = np.zeros(config.n_actions)
        for j in range(config.perturbation_range):
            for i in range(config.n_actions):
                tmp[i] = 1 # j+1
                self.perturbation_array.append(tmp)
                tmp = np.zeros(config.n_actions)
        self.perturbation_array = torch.tensor(self.perturb_scaler*np.array(self.perturbation_array),dtype=torch.float32)
        self.perturbed_array = torch.tensor(np.zeros((2 * self.perturbation_range * self.config.n_actions, self.config.n_actions)),dtype=torch.float32)
        self.max_array_perturbed_array = torch.tensor(np.ones((self.config.n_actions*self.perturbation_range, self.config.n_actions))*self.clipmax,dtype=torch.float32)
        self.min_array_perturbed_array = torch.tensor(np.ones((self.config.n_actions*self.perturbation_range, self.config.n_actions))*self.clipmin,dtype=torch.float32)
        self.action_space_matrix = torch.tensor(action_space_matrix)

        # Attributes required within Simulated Annealing
        self.initialAcceptance = config.initialAcceptance
        if self.maximum_greedy_search_steps == 0:
            self.initialK = 0
        else:
            self.initialK = int(0.1 * self.perturbation_array.shape[0])
        self.cooling = config.cooling  # this cooling parameter determines the percentage of cooling for the hamming, num_neighbors, and the K values
        self.coolingK = max(int(self.cooling * self.initialK), 1)
        self.acceptanceCooling = config.acceptanceCooling


        # Only if matrix lookup needed, e.g., for debugging purposes
        if len(self.action_space_matrix)>0 and config.env_name!='Recommender_py':
            self.flann = pyflann.FLANN()
            self.index = self.flann.build_index(self.action_space_matrix, algorithm='kdtree') #linear = brute force
            self.recomm = 0
        elif len(self.action_space_matrix)==0 and config.env_name=='Recommender_py':
            self.flann = pyflann.FLANN()
            self.tf_idf_matrix = self.config.env.tf_idf_matrix
            self.index = self.flann.build_index(self.config.env.tf_idf_matrix, algorithm='kdtree') #linear = brute force
            self.recomm = 1
            self.size_feature_vector = self.config.env.tf_idf_matrix.shape[1]
            self.no_recommended_items = self.config.env.no_recommended_items
        else:
            print('No action_space_matrix, so mapping returns literal action!')
            self.recomm = 0

        # Search procedure corresponds to simulated annealing
        self.search = self.simulatedAnnealing


    # Calculation of Q-values
    def get_qvalues(self,state, neighbors,critic):
        state_replications = torch.tile(state, (len(neighbors), 1))
        neighbors = torch.tensor(neighbors)
        return critic.forward(state_replications, neighbors), neighbors


    # Generation of neighborhood
    def deterministic_neighborhood(self,original_array):
        if self.perturbation_range == 0:
            perturbed_array = original_array
        else:
            perturbed_array_high = torch.minimum(original_array + self.perturbation_array,self.max_array_perturbed_array)
            perturbed_array_low = torch.maximum(original_array - self.perturbation_array,self.min_array_perturbed_array)
            self.perturbed_array[:self.perturbation_range * self.config.n_actions,:] = perturbed_array_high
            self.perturbed_array[self.perturbation_range * self.config.n_actions:, :] = perturbed_array_low
            perturbed_array = torch.unique(self.perturbed_array,dim=0)
        return perturbed_array

    # Generation of base action
    def get_baseaction_minmax(self,proto_action,state,critic):
        clipped_proto_action = torch.clip(proto_action,min=-self.outputLayerLimit,max=self.outputLayerLimit)
        base = torch.round(self.scaler(clipped_proto_action, amin=-self.outputLayerLimit, amax=self.outputLayerLimit, smin=self.clipmin,smax=self.clipmax),decimals=self.clipped_decimals)
        q,a = self.get_qvalues(state,base,critic)
        return base,q,a
    
    # Main DNC Loop: 1) Base action, 2) SA (within which base action generation repeated)
    def get_neighbors(self, state, proto_action, critic):
       k = self.initialK
       acceptanceRatio = self.initialAcceptance
       k_best_actions = torch.empty((0))
       baseAction,best_q, best_action = self.get_baseaction_minmax(proto_action,state,critic)
       
       #do simulated annelaing or greedy search
       best_q,best_action = self.search(k,acceptanceRatio,baseAction,state,critic,k_best_actions,best_q,best_action)
         
       return best_q,best_action
     
   # Simulated Annealing
    def simulatedAnnealing(self, k, acceptanceRatio, baseAction, state, critic, k_best_actions,
                           best_q, best_action):
        while k > 0:
            neighborhood = self.deterministic_neighborhood(baseAction)
            neighborhood = np.append(neighborhood, baseAction, axis=0)  # also evaluate base action
            qvalues, actions = self.get_qvalues(state, neighborhood, critic)

            # select the top K neighbors
            best_qs, best_idxs = torch.topk(qvalues, min(k, len(qvalues)), dim=0, sorted=True)
            k_best_actions = torch.cat((k_best_actions, actions[best_idxs.flatten()]))

            if best_qs[0] > qvalues[-1]:  # accept element in neighborhood if better or same to base action
                if best_qs[0] > best_q:
                    best_q, best_action = best_qs[0], actions[best_idxs[0]]  # NEIGHBOR IS ALSO GLOBALLY BEST
                baseAction = actions[best_idxs[0]]
            elif np.random.random() < np.exp(-(qvalues[-1] - best_qs[0]).detach().numpy() / acceptanceRatio):  # else, accept element in neighborhood with some probability
                acceptanceRatio -= self.acceptanceCooling
                baseAction = actions[best_idxs[0]]  # select the best action (but not globally) in the last neighborhood
            else:  # move to different neighborhood
                rand = np.random.randint(0, len(k_best_actions))
                baseAction = k_best_actions[
                    rand]  # select a random action among the top K actions in different neighborhoods
                k_best_actions = np.delete(k_best_actions, rand, axis=0)

            # Cooling scheme
            k -= self.coolingK
        return best_q, best_action
       
    # Function that is called by the RL agent
    def get_best_match(self, proto_action, state, critic):
        bestQ, action = self.get_neighbors(state, proto_action,critic)

        # If using the recommender environment, look-up wether the movies that DNC recommends exist
        if self.recomm == 1:
            j=0
            feasible_action = []
            for i in range(0,self.no_recommended_items):
                action_tmp = action[0,j:j+self.size_feature_vector]
                action_id, _ = self.flann.nn_index(np.array(action_tmp).astype(np.float64), 1)
                feasible_action.extend(self.tf_idf_matrix[action_id].tolist()[0])
                j+=self.size_feature_vector
            return torch.tensor(feasible_action).view(1,-1)
        elif len(self.action_space_matrix)>0:
            action_id, _ = self.flann.nn_index(np.array(action).astype(np.float64), 1)
            return int(action_id)
        return action

    