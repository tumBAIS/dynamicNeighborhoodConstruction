import numpy as np

# Parent to all RL algorithms -- in our instance, QAC
class Agent:

    def __init__(self, config):
        self.state_low, self.state_high = config.env.observation_space.low, config.env.observation_space.high
        self.state_diff = self.state_high - self.state_low

        try:
            if config.env.action_space.dtype == np.float32:
                self.action_low, self.action_high = config.env.action_space.low, config.env.action_space.high
                self.action_diff = self.action_high - self.action_low
        except:
            print('-------------- Warning: Possible action type mismatch ---------------')

        self.state_dim = config.env.observation_space.shape[0]

        if len(config.env.action_space.shape) > 0 and config.mapping == 'learned_mapping':
            self.action_dim = config.env.action_space.shape[0]
        elif config.mapping == 'knn_mapping' or config.mapping == 'no_mapping' or config.mapping == 'minmax_mapping' or config.mapping == 'dnc_mapping':
            self.action_dim = config.env.n_actions
        else:
            self.action_dim = config.env.action_space.n

        self.config = config

        # Abstract class variables
        self.modules = None

    def clear_gradients(self):
        for _, module in self.modules:
            module.optim.zero_grad()

    def save(self):
        if self.config.save_model:
            for name, module in self.modules:
                module.save(self.config.paths['checkpoint'] + name+'.pt')

    def step(self, loss, clip_norm=False):
        self.clear_gradients()
        loss.backward()
        for _, module in self.modules:
            module.step(clip_norm)

    def reset(self):
        for _, module in self.modules:
            module.reset()
