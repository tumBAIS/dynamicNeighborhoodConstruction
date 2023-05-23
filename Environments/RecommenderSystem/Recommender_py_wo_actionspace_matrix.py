from __future__ import print_function

import sys

import numpy as np
import scipy.special
import torch
from Src.Utils.Utils import Space

class Recommender_py_wo_actionspace_matrix(object):
    def __init__(self,
                 only_labels = True,
                 n_actions=3,
                 max_steps=100,
                 path_to_files='Environments'
                 ):

        self.actionLiteral = True

        self.max_steps = max_steps

        self.tf_idf_matrix = np.load(path_to_files+'/RecommenderSystem/tf_idf_matrix.npy')
        self.rewards_per_item = np.load(path_to_files+'/RecommenderSystem/rewards_vector.npy')

        self.tf_idf_matrix = np.around(self.tf_idf_matrix,decimals=4)

        self.no_items = self.tf_idf_matrix.shape[0]
        self.n_actions = n_actions
        high = np.max(self.tf_idf_matrix)
        low = np.min(self.tf_idf_matrix)
        self.observation_space = Space(low=low*np.ones(self.tf_idf_matrix.shape[1], dtype=np.float32), high=high*np.ones(self.tf_idf_matrix.shape[1], dtype=np.float32),dtype=np.float32)
        self.state_space = self.tf_idf_matrix

        if n_actions == self.tf_idf_matrix.shape[1]:
            self.no_recommended_items = int(self.n_actions / self.tf_idf_matrix.shape[1])
            self.action_space = Space(size=self.no_items,low = [low], high = [high])
            self.total_no_actions = self.no_items
            self.action_space_matrix_ids = []
            self.action_space_matrix = []
        else:
            self.no_recommended_items = int(self.n_actions / self.tf_idf_matrix.shape[1])
            self.action_space = Space(size=self.no_items,low = [low], high = [high])
            self.n_actions_help = int(self.n_actions / self.tf_idf_matrix.shape[1])
            self.total_no_actions = scipy.special.binom(self.no_items,self.n_actions_help).astype(int)
            self.action_space_matrix_ids = []
            self.action_space_matrix = []
        self.reset()

    def seed(self, seed):
        self.seed = seed


    def reset(self,training=True):
        """
        Sets the environment to default conditions
        :return: None
        """
        self.steps_taken = 0
        self.objects = {}

        id = np.random.randint(0,self.no_items)
        self.curr_state = self.state_space[id]

        return self.curr_state



    def step(self,action,training=True):
        #
        if torch.is_tensor(action):
            if len(action.shape) == 2:
                action = action.view(-1).numpy()
            else:
                action = action.numpy()
        elif len(action.shape)>1:
            action = action[0]

        self.steps_taken += 1
        item_probabilities = self.get_item_probabilities(action)
        item_probabilities = np.maximum(item_probabilities,np.zeros(len(item_probabilities)))
        user_choices = np.arange(0,self.no_recommended_items)
        user_choices = user_choices.tolist()
        user_choices.append(-1)
        selected_item = np.random.choice(user_choices, 1, replace=False, p=item_probabilities)
        if selected_item[0] == -1:
            selected_item = self.sample_item()
        else:
            action_tmp = np.reshape(action,(self.no_recommended_items,self.tf_idf_matrix.shape[1]))
            selected_item = action_tmp[selected_item[0],:]

        done = self.is_terminal(selected_item,action)
        self.curr_state = selected_item
        return self.curr_state , self.reward, done, {'No INFO implemented yet'}

    def sample_item(self):
        item_probabilities = np.ones(self.no_items)/self.no_items
        selected_item = self.curr_state
        while np.array_equal(selected_item,self.curr_state):
           ids  = np.random.choice(np.arange(0,self.no_items),1,replace=False, p=item_probabilities)
           selected_item = self.tf_idf_matrix[ids[0]]
        return selected_item


    def sigmoid(self,x):
        return 1/(1+np.exp(-x*5))

    # based on cosine similarity
    def get_similarity(self,state,action):
        if np.array_equal(action,state):
            return -1
        else:
            return np.dot(action, state) / (sum(action ** 2) * sum(state ** 2))
    def get_item_probabilities(self,action):

        non_existing_item = False
        act_sim = []
        if self.no_recommended_items > 1:
            j = 0
            for i in range(0,self.no_recommended_items):
                similarity = self.get_similarity(self.curr_state,action[j:j+self.tf_idf_matrix.shape[1]])
                j+=self.tf_idf_matrix.shape[1]
                y = self.sigmoid(similarity)
                act_sim.append(y)
            x = np.average(np.array(act_sim))
            act_sim =x * np.array(act_sim)/sum(np.array(act_sim))
            no_item_prob = 1-x
        else:
            similarity = self.get_similarity(self.curr_state, action)
            x = self.sigmoid(similarity)
            act_sim = np.array([x])
            no_item_prob = 1 - act_sim[0]

        act_sim = act_sim.tolist()
        act_sim.extend([no_item_prob])
        item_probabilities = np.array(act_sim)

        return item_probabilities

    def is_terminal(self,selected_item,action):
        if self.no_recommended_items > 1:
            action_tmp = np.reshape(action, (self.no_recommended_items, self.tf_idf_matrix.shape[1]))
            done = (selected_item in action_tmp) * np.random.choice([0, 1], 1, replace=False, p=[0.9, 0.1]) + (
                        selected_item not in action_tmp) * np.random.choice([0, 1], 1, replace=False, p=[0.8, 0.2])
        else:
            done = np.array_equal(action,selected_item)*np.random.choice([0,1],1,replace=False, p=[0.9,0.1]) + (not np.array_equal(action,selected_item)) * np.random.choice([0, 1], 1, replace=False, p=[0.8, 0.2])

        selected_item = np.around(np.array(selected_item, dtype=np.float64), decimals=4)
        id = np.where(np.all(selected_item == self.tf_idf_matrix, axis=1))
        self.reward = self.rewards_per_item[id[0][0]]
        if self.steps_taken >= self.max_steps or done[0] == 1:
            return 1
        else:
            return 0


if __name__=="__main__":
    # Random Agent
    rewards_list = []
    steps_list = []
    steps = []
    n_actions = 2
    env = Recommender_py_wo_actionspace_matrix(only_labels=True,n_actions=n_actions,max_steps=100)
    for i in range(1000):
        rewards = 0
        steps = 0
        done = False
        env.reset()
        while not done:
            action = np.zeros(env.tf_idf_matrix.shape[1]*2)
            id1 = 0
            id2 = 0
            while id1 == id2:
                id1 = np.random.randint(0,env.no_items)
                id2 = np.random.randint(0, env.no_items)
            action[0:env.tf_idf_matrix.shape[1]] = env.tf_idf_matrix[id1,:]
            action[env.tf_idf_matrix.shape[1]:env.tf_idf_matrix.shape[0]*2] = env.tf_idf_matrix[id2, :]
            next_state, r, done, _ = env.step(action)
            rewards += r
            steps+=1
        rewards_list.append(rewards)
        steps_list.append(steps)

    print("Average random rewards: ", np.mean(rewards_list))
    print("Standard deviation:",np.std(rewards_list))
    print("Average random steps: ", np.mean(steps_list))