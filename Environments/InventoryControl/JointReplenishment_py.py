from __future__ import print_function
import numpy as np
from Src.Utils.Utils import Space, binaryEncoding

class JointReplenishment_py(object):
    def __init__(self,
                 smin=0,
                 smax=66,
                 n_items=1,
                 debug=False,
                 max_steps=100,
                 commonOrderCosts = 75,
                 mappingType ='knn_mapping'
                 ):

        
        self.n_actions = n_items
        self.max_steps=max_steps
        self.commonOrder = commonOrderCosts
        if mappingType == 'knn_mapping' or mappingType == 'learned_mapping' or mappingType == 'no_mapping':
            self.actionLiteral = False
        else:
            self.actionLiteral = True
        
        if self.n_actions > 5 and self.actionLiteral==False:
            print('Action space too large for 1 FLANN dataset, please set actionLiteral to True')
       
        self.s_min = smin # minimum order quanitity
        self.s_max = smax # maximu order quantity
        self.lambda_even = 20#20#demand parameter #optimal s,S = (22,28) for K=0
        self.lambda_uneven = 10 #optimal s,S = (11,16) for K=0
        
        self.cur_inv = np.full(self.n_actions, 2, dtype=np.int32)
        self.b = np.zeros(self.n_actions, dtype=np.float32)
        self.h = np.zeros(self.n_actions, dtype=np.float32)
        self.k = np.zeros(self.n_actions, dtype=np.float32)
        self.d_lambda = np.zeros(self.n_actions, dtype=np.float32)
        
        
        self.debug = debug
      
        self.action_space = Space(size=(self.s_max+1)**self.n_actions,low=np.zeros(self.n_actions, dtype=np.float32), high=np.full(self.n_actions, self.s_max, dtype=np.float32)) # ,low=smin,high=smax
        self.observation_space = Space(low=(-2*self.lambda_even)*np.ones(self.n_actions, dtype=np.float32), high=np.full(self.n_actions, self.s_max, dtype=np.float32), dtype=np.float32)
        self.obs_space_span = self.observation_space.high[0]-self.observation_space.low[0]
        if self.actionLiteral == False:
            self.levels, self.action_space_matrix = self.get_action_levels(self.n_actions)
        else:
            self.levels = []
            self.action_space_matrix = []
        
        self.set_rewards()
        self.reset()

    def seed(self, seed):
        self.seed = seed

    def get_embeddings(self):
        print('not implemented')
        return self.levels.copy()

    def set_rewards(self):
        # All rewards
        self.h_even = -1 #holding costs per item quantity
        self.h_uneven = -1 
        self.b_even = -19 #backorder costs per item quantity
        self.b_uneven = -19
        self.k_even = -10 #ordering costs per order
        self.k_uneven = -10 
        self.K = -self.commonOrder  #fixed order costs (if K=0, we know the optimal policy)
        
        for i in range(self.n_actions):
            if np.mod(i,2)==0:
                self.b[i] = self.b_even
                self.k[i] = self.k_even
                self.h[i] = self.h_even
                self.d_lambda[i] = self.lambda_even
            else:
                self.b[i] = self.b_uneven
                self.k[i] = self.k_uneven
                self.h[i] = self.h_uneven
                self.d_lambda[i] = self.lambda_uneven
        
       

    def reset(self,training=True):
        """
        Sets the environment to default conditions
        """
        self.steps_taken = 0

        self.cur_inv = np.full(self.n_actions, 25)

        self.curr_state = self.make_state()

        return self.curr_state


    def step(self, action,training=True):
        self.steps_taken += 1
        reward = 0
        # Check if previous state was end of MDP, if it was, then we are in absorbing state currently.
        # Terminal state has a Self-loop and a 0 reward
        term = self.is_terminal()
        if term:
            return self.curr_state, 0, term, {'No INFO implemented yet'}

        #ordering of new items
        if self.actionLiteral:
            orderUpTo = np.array(action.numpy()[0]).astype(dtype=np.int32)#only used for evaluation of static policy/actionLiteral
        else:
            orderUpTo = self.levels[action].astype(dtype=np.int32)  # Table look up for the impact/effect of the selected action
        orders = (orderUpTo - self.cur_inv).clip(min=0,dtype=np.int32)
        reward += np.sum(orders*self.k)
         
        if np.any(orders):
            reward += self.K
            
        #add new order to inventory (we assume zero lead time)
        self.cur_inv = np.add(self.cur_inv, orders)
        
        #sample demand and subtract from inventory
        demand = np.random.poisson(self.d_lambda,self.n_actions)
        self.cur_inv = np.subtract(self.cur_inv,demand)
        
        #pay holding costs over positive inventory
        reward += np.sum(self.cur_inv.clip(min=0) * self.h)
        
        #pay backorder costs over negative inventory
        reward += np.sum(self.cur_inv.clip(max=0) * -self.b)
       
        # self.update_state()
        self.curr_state = self.make_state()


        return self.curr_state.copy(), reward, self.is_terminal(), {'No INFO implemented yet'}


    # Normalize such that it works wel with subsequent steps
    def make_state(self):
        state = self.cur_inv/self.obs_space_span # -self.avg_no_items)/self.avg_no_items
        return state


    def is_terminal(self):
        if self.steps_taken >= self.max_steps:
            return 1
        else:
            return 0
        
    def get_action_levels(self, n_actions):
         shape = ((self.s_max+1)**n_actions, n_actions)
         levels = np.zeros(shape)
         for idx in range(shape[0]):
             action = binaryEncoding(idx, n_actions,self.s_max+1)
             levels[idx] = action

         return levels,levels
     
        
    def simPolicy(self,s,S,replications=10):
        rewards_list = []
        for t in range(replications):
            rewards = 0
            done = False
            env.reset()
            while not done:
                action = []
                for i in range(len(env.cur_inv)):
                    if env.cur_inv[i] <= s[i%2]:
                        action.extend([ S[i%2]] )
                    else:
                        action.append( 0 )
                
                next_state, r, done, _ = env.step(action=action)
                rewards += r
            rewards_list.append(rewards)
        return np.mean(rewards_list)
         

if __name__=="__main__":
    # Random Agent
    rewards_list = []
    env = JointReplenishment_py(debug=True, n_items=4, commonOrderCosts=0, max_steps=100, actionLiteral = True)
    
    s = [22,11]
    S = [28,16]
    r = env.simPolicy(s,S,replications=10000)
    print("Average static policy rewards: ", r)
    
    for i in range(50):
        rewards = 0
        done = False
        env.reset()
        while not done:
            # env.render()
            action = np.random.randint((env.s_max+1)**env.n_actions)
            next_state, r, done, _ = env.step(action)
            rewards += r
        rewards_list.append(rewards)

    print("Average random rewards: ", np.mean(rewards_list), np.sum(rewards_list))