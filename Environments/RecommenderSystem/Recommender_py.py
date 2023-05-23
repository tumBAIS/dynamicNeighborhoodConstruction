from __future__ import print_function
import sys
import numpy as np
import scipy.special
from Src.Utils.Utils import Space

class Recommender_py(object):
    def __init__(self,
                 n_actions=3,
                 max_steps=100,
                 path_to_files='Environments'
                 ):

        self.actionLiteral = False

        self.max_steps = max_steps

        self.tf_idf_matrix = np.load(path_to_files+'/RecommenderSystem/tf_idf_matrix.npy')
        self.rewards_per_item = np.load(path_to_files+'/RecommenderSystem/rewards_vector.npy')

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
            self.action_space_matrix_ids = np.arange(0,self.no_items).astype(int)
            self.action_space_matrix = np.zeros((self.no_items,self.tf_idf_matrix.shape[1]))
            self.action_space_matrix[:, :] = self.tf_idf_matrix[:,:]
        else:
            self.no_recommended_items = int(self.n_actions / self.tf_idf_matrix.shape[1])
            self.action_space = Space(size=self.no_items,low = [low], high = [high])
            self.n_actions_help = int(self.n_actions / self.tf_idf_matrix.shape[1])
            self.total_no_actions = scipy.special.binom(self.no_items,self.n_actions_help).astype(int)
            self.action_space_matrix_ids = np.zeros((self.total_no_actions,self.n_actions_help))
            self.action_space_matrix = np.zeros((self.total_no_actions,self.n_actions))
            k = 0
            for i in range(0, self.no_items):
                first_column = np.ones(self.no_items - i-1) * i
                second_column = np.arange(min(i + 1, self.no_items), self.no_items)
                self.action_space_matrix_ids[k:k + self.no_items - i-1, :] = np.array([first_column, second_column]).transpose()
                self.action_space_matrix[k:k + self.no_items - i - 1, :self.tf_idf_matrix.shape[1]] = self.tf_idf_matrix[first_column.astype(int)]
                self.action_space_matrix[k:k + self.no_items - i - 1, self.tf_idf_matrix.shape[1]:2*self.tf_idf_matrix.shape[1]] = self.tf_idf_matrix[second_column.astype(int)]
                k += self.no_items - i-1
            self.action_space_matrix_ids = self.action_space_matrix_ids.astype(int)

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

        self.curr_state = np.random.randint(0,self.no_items)

        return self.state_space[self.curr_state]



    def step(self,action,training=True):

        self.steps_taken += 1
        action = self.action_space_matrix_ids[action]

        item_probabilities = self.get_item_probabilities(action)
        item_probabilities = np.maximum(item_probabilities,np.zeros(len(item_probabilities)))
        if self.no_recommended_items > 1:
            user_choices = action.tolist()
        else:
            user_choices = [action]
        user_choices.extend([-1])

        selected_item = np.random.choice(user_choices, 1, replace=False, p=item_probabilities)
        if selected_item[0] == -1:
            selected_item = self.sample_item()
        done = self.is_terminal(selected_item[0],action)
        self.curr_state = selected_item[0]
        return self.state_space[self.curr_state],self.reward,done,{'No INFO implemented yet'}

    def make_state(self,selected_item):
        return np.array([selected_item[0],self.curr_state[0]])
    def sample_item(self):
        item_probabilities = np.ones(self.no_items)/self.no_items
        selected_item = self.curr_state
        while selected_item == self.curr_state:
            selected_item = np.random.choice(np.arange(0,self.no_items),1,replace=False, p=item_probabilities)
        return selected_item


    def sigmoid(self,x):
        return 1/(1+np.exp(-x*5))

    def get_similarity(self,action):
        if np.array_equal(self.tf_idf_matrix[self.curr_state], action):
            return -1
        else:
            return np.dot(action, self.tf_idf_matrix[self.curr_state]) / (sum(action ** 2) * sum(self.tf_idf_matrix[self.curr_state] ** 2))
    def get_item_probabilities(self,action):

        act_sim = []
        if self.no_recommended_items > 1:
            for i in range(0,self.no_recommended_items):
                y = self.sigmoid(self.get_similarity(self.tf_idf_matrix[action[i]]))
                act_sim.append(y)
            x = np.average(np.array(act_sim))
            act_sim =x * np.array(act_sim)/sum(np.array(act_sim))
            no_item_prob = 1-x
        else:
            x = self.sigmoid(self.get_similarity(self.tf_idf_matrix[action]))
            act_sim = np.array([x])
            no_item_prob = 1 - act_sim[0]

        act_sim = act_sim.tolist()
        act_sim.extend([no_item_prob])
        item_probabilities = np.array(act_sim)

        return item_probabilities

    def is_terminal(self,selected_item,action):

        if self.no_recommended_items > 1:
            done = (selected_item in action) * np.random.choice([0, 1], 1, replace=False, p=[0.9, 0.1]) + (
                        selected_item not in action) * np.random.choice([0, 1], 1, replace=False, p=[0.8, 0.2])
        else:
            done = (action == selected_item)*np.random.choice([0,1],1,replace=False, p=[0.9,0.1]) + (action != selected_item) * np.random.choice([0, 1], 1, replace=False, p=[0.8, 0.2])

        if self.steps_taken >= self.max_steps or done[0] == 1:
            self.reward = self.rewards_per_item[selected_item]
            return 1
        else:
            self.reward = self.rewards_per_item[selected_item]
            return 0


if __name__=="__main__":
    # Random Agent
    rewards_list = []
    steps_list = []
    steps = []
    env = Recommender_py(n_actions=46,max_steps=100)
    for i in range(1000):
        rewards = 0
        steps = 0
        done = False
        env.reset()
        while not done:
            action = np.array([np.random.randint(0,env.total_no_actions)])
            next_state, r, done, _ = env.step(action[0])
            rewards += r
            steps+=1
        rewards_list.append(rewards)
        steps_list.append(steps)

    print("Average random rewards: ", np.mean(rewards_list))
    print("Standard deviation:",np.std(rewards_list))
    print("Average random steps: ", np.mean(steps_list))