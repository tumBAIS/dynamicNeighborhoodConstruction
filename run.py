import numpy as np
import Src.Utils.Utils as Utils
from Src.parser import Parser
from Src.config import Config
from time import time



class Solver:
    def __init__(self, config):
        # Initialize the required variables

        self.config = config
        self.env = self.config.env
        self.state_dim = np.shape(self.env.reset())[0]

        if len(self.env.action_space.shape) > 0:
            self.action_dim = self.env.action_space.shape[0]
        else:
            self.action_dim = self.env.action_space.n
        print("Actions space: {} :: State space: {}".format(self.action_dim, self.state_dim))

        self.model = config.algo(config=config)

    def eval(self, episodes=100):
        # Evaluate the model
        rewards = []
        steps = []
        step_time = []
        for episode in range(episodes):
            trajectory = []
            state = np.float32(self.env.reset(training=False))
            total_r, step = 0, 0
            done = False
            while not done:
                t1 = time()
                action, dist = self.model.get_action(state,training=True)
                new_state, reward, done, info = self.env.step(action,training=False)
                state = new_state
                trajectory.append((action, reward))
                total_r += reward
                step += 1
                step_time.append(time()-t1)
            rewards.append(total_r)
            steps.append(step)
        return rewards, step_time,steps

    # Main training loop
    def train(self):
        # Learn the model on the environment
        returns = []
        test_returns = []
        test_std = []
        run_time = []
        total_loss_actor_history = []
        total_loss_critic_history = []
        total_loss_actor = 0
        total_loss_critic = 0

        avg_step_time = []
        checkpoint = self.config.save_after
        rewards, rm, start_ep = [], 0, 0

        steps = 0
        t0 = time()
        t_init = time()
        for episode in range(start_ep, self.config.max_episodes):

            episode_loss_actor = []
            episode_loss_critic = []

            # Reset both environment and model before a new episode
            state = self.env.reset()
            self.model.reset()

            step, total_r = 0, 0
            done = False



            while not done:
                action, a_hat = self.model.get_action(state, training=True)
                new_state, reward, done, info = self.env.step(action=action)
                loss_actor,loss_critic=self.model.update(state, action, a_hat, reward, new_state, done)
                episode_loss_actor.append(loss_actor)
                episode_loss_critic.append(loss_critic)
                state = new_state
                total_r += reward
                step += 1
                if step > self.config.max_steps:
                    break

            # Track actor and critic loss
            total_loss_actor = total_loss_actor*0.99+0.01*np.average(episode_loss_actor)
            total_loss_actor_history.append(total_loss_actor)
            total_loss_critic=total_loss_critic*0.99+0.01*np.average(episode_loss_critic)
            total_loss_critic_history.append(total_loss_critic)
            steps += step
            rm = 0.99*rm + 0.01*total_r

            if episode%checkpoint == 0 or episode == self.config.max_episodes-1:
                print('Episode '+str(episode)+' / current actor loss: ' + str(total_loss_actor))
                print('Episode '+str(episode)+' / current critic loss: ' + str(total_loss_critic))
                np.save(self.config.paths['results'] + "actor_loss", total_loss_actor_history)
                np.save(self.config.paths['results'] + "critic_loss", total_loss_critic_history)
                returns.append(rm)
                run_time.append((time()-t_init))
                print('time required for '+str(checkpoint)+' :' +str(time()-t0))
                if self.config.env_name == 'Recommender_py':
                    test_reward,step_time,_=self.eval(500)
                else:
                    test_reward, step_time, _ = self.eval(10)
                avg_test_reward = np.average(test_reward)
                std_test_reward = np.std(test_reward)
                avg_step_time.append(np.average(step_time))
                test_std.append(std_test_reward)
                test_returns.append(avg_test_reward)
                Utils.save_plots(returns, run_time,config=self.config)
                Utils.save_plots_test_runs(test_returns,test_std,avg_step_time,config=self.config)
                t0 = time()
                steps = 0

# @profile
def main(train=True):
    t = time()
    args = Parser().get_parser().parse_args()

    config = Config(args)
    solver = Solver(config=config)

    if train:
        solver.train()
    print(time()-t)

if __name__== "__main__":
        main(train=True)

