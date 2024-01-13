from __future__ import print_function
import numpy as np
import torch
from Src.Utils.Utils import Space

class JobShop_py(object):
    def __init__(self,
                 n_machines = 5,
                 n_jobs = 50,
                 debug=False,
                 max_steps=100,
                 mappingType ='DNC'
                 ):

        
        self.n_actions = n_machines#action vector size
        self.max_steps=max_steps
       
        self.num_machines = n_machines  # Number of machines
        self.max_jobs_per_machine = n_jobs  # Maximum capacity per machine
        self.max_energy_usage = 100  # Max energy that can be used by a machine
        
        # Initialize state
        self.machines = {'energy_usage': np.random.uniform(1.0, 1.2, self.num_machines),
                         'wear_level': np.zeros(self.num_machines)}#machines are heterogeneous, so consumption per job differs from the start

        
        if mappingType == 'knn_mapping' or mappingType == 'learned_mapping' or mappingType == 'no_mapping':
            raise Exception("Sorry, this mapping type is not supported") 
        else:
            self.actionLiteral = True
        
        self.debug = debug
      
        self.action_space = Space(size=(n_jobs)**self.n_actions,low=np.zeros(self.n_actions, dtype=np.float32), high=np.full(self.n_actions, n_jobs, dtype=np.float32)) # ,low=smin,high=smax
        self.observation_space = Space(low=np.ones(2*self.n_actions, dtype=np.float32), high=np.full(2*self.n_actions, 5.0, dtype=np.float32), dtype=np.float32)

        self.action_space_matrix = []
        
        self.reset()

    def seed(self, seed):
        self.seed = seed

    def get_embeddings(self):
        print('not implemented')
        return self.levels.copy()


    def step(self, action,training=True):
        reward = 0
        done = False
        info = {}

        if isinstance(action, torch.Tensor):
            action = action.cpu().numpy()[0]
            
        # Update machine states based on action
        job_counts = []  # To store the number of jobs assigned to each machine
        for i, jobs in enumerate(action):


            wear_increase = np.random.uniform(0.05, 0.4) if int(jobs) > 0.75 * self.max_jobs_per_machine else 0#wear is stochastic
            self.machines['wear_level'][i] += wear_increase

            if int(jobs) < 0.5 * self.max_jobs_per_machine:
                wear_decrease = np.random.uniform(0.05, 0.4)  # Random wear decrease
                self.machines['wear_level'][i] = max(self.machines['wear_level'][i] - wear_decrease,0)  # Decrease wear but ensure it's not negative


            self.machines['energy_usage'][i] = self.calculate_energy_usage(i, jobs)

            # Calculate reward (or penalty)
            reward -= self.machines['energy_usage'][i]
            reward += jobs * 3  # Reward proportional to the number of jobs (e.g., 10 points per job)

            job_counts.append(jobs)
    
        # Calculate variance penalty, this couples the machines and makes sequential decisions over machines sub-optimal
        variance_penalty = np.std(job_counts)
        reward -= variance_penalty  # Subtract variance penalty from reward

        self.current_step += 1
        if self.current_step == self.max_steps-1:
            done=True

        return self._next_observation(), reward, done, info

    def calculate_energy_usage(self, machine_id, jobs):
        base_energy = self.machines['energy_usage'][machine_id]  # Base energy consumption rate
        wear_factor = self.machines['wear_level'][machine_id]
        energy_usage = base_energy * (1 + wear_factor) * jobs
        return min(energy_usage, self.max_energy_usage)  # Cap the energy usage

    def reset(self,training=False):
        self.machines = {'energy_usage': np.random.uniform(1.0, 2, self.num_machines),
                         'wear_level': np.zeros(self.num_machines)}
        # self.machines['energy_usage'] = np.ones(self.num_machines)
        self.current_step = 0
        #self.jobs_remaining = self.num_jobs
        return self._next_observation()

    def _next_observation(self):
        # Combine energy usage and wear level for each machine
        obs = np.concatenate((self.machines['energy_usage'], self.machines['wear_level']))
        return obs

    def render(self, mode='human'):
        print(f"Machine states: {self.machines}")

 
if __name__=="__main__":
    # Random Agent
    env = JobShop_py()
    state = env.reset()
    done = False
    r = 0
    while not done:
        action = np.random.randint(1,50,5)  # Replace with your RL algorithm's action
        next_state, reward, done, _ = env.step(action)
        r+=reward
        env.render() 
    print(reward)
