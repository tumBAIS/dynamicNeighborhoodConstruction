from __future__ import print_function

import time
import sys

import numpy as np
import matplotlib.pyplot  as plt
from matplotlib.patches import Rectangle, Circle, Arrow
from matplotlib.ticker import NullLocator
from Src.Utils.Utils import Space, binaryEncoding

class Gridworld_py_teleport_tiles(object):
    def __init__(self,
                 mappingType='knn_mapping',
                 n_actions=4,
                 debug=True,
                 max_step_length=0.2,
                 max_steps=30,
                 with_wall=True,
                 collision_reward=0,
                 constraint='hard',
                 ):

        self.debug = debug
        self.constraint = constraint
        self.n_actions = n_actions
        self.action_space = Space(size=2**n_actions)
        self.observation_space = Space(low=np.zeros(2, dtype=np.float32), high=np.ones(2, dtype=np.float32), dtype=np.float32)
        self.disp_flag = False

        if mappingType == 'knn_mapping' or mappingType == 'learned_mapping' or mappingType == 'no_mapping':
            self.actionLiteral = False
        else:
            self.actionLiteral = True

        self.movement = self.get_movements(self.n_actions)

        if self.n_actions == 28 and self.actionLiteral:
            self.max_dist = 8.93217285994847
            self.action_space_matrix = np.zeros(0)
        else:
            self.motions,self.action_space_matrix = self.get_action_motions(self.n_actions)

        self.wall = with_wall
        self.wall_width = 0.05
        self.step_unit = self.wall_width - 0.005
        self.repeat = int(max_step_length / self.step_unit)

        self.max_steps = int(max_steps / max_step_length)
        self.step_reward = -0.05
        self.collision_reward = collision_reward  # -0.05
        self.movement_reward = 0  # 1
        self.randomness = 0.1

        self.get_static_teleport_tiles()

        self.n_lidar = 0
        self.angles = np.linspace(0, 2 * np.pi, self.n_lidar + 1)[:-1]  # Get 10 lidar directions,(11th and 0th are same)
        self.lidar_angles = list(zip(np.cos(self.angles), np.sin(self.angles)))

        # if debug:
        self.heatmap_scale = 99
        self.heatmap = np.zeros((self.heatmap_scale + 1, self.heatmap_scale + 1))

        self.reset()

    def seed(self, seed):
        self.seed = seed

    def get_embeddings(self):
        return self.motions.copy()

    def set_rewards(self):
        # All rewards
        self.G1_reward = 100 #100
        self.G2_reward = 0

    def reset(self,training=True):
        """
        Sets the environment to default conditions
        :return: None
        """
        self.set_rewards()
        self.steps_taken = 0
        self.reward_states = self.get_reward_states()
        self.dynamic_obs = self.get_dynamic_obstacles()
        self.objects = {}

        self.curr_pos = np.array([0.99, 0.01]) # np.array([0.1,0.35])  #

        if training:
            while True:
                self.curr_pos[0] = np.random.uniform(0,1)
                self.curr_pos[1] = np.random.uniform(0, 1)
                if self.valid_pos((self.curr_pos[0],self.curr_pos[1])) and not self.is_terminal():
                    break
        self.curr_state = self.make_state()
        return self.curr_state

    def get_movements(self, n_actions):
        """
        Divides 360 degrees into n_actions and
        assigns how much it should make the agent move in both x,y directions

        usage:  delta_x, delta_y = np.dot(action, movement)
        :param n_actions:
        :return: x,y direction movements for each of the n_actions
        """
        x = np.linspace(0, 2*np.pi, n_actions+1)
        y = np.linspace(0, 2*np.pi, n_actions+1)
        motion_x = np.around(np.cos(x)[:-1], decimals=3)
        motion_y = np.around(np.sin(y)[:-1], decimals=3)
        movement = np.vstack((motion_x, motion_y)).T

        if self.debug: print("Movements:: ", movement)
        return movement

    def get_action_motions(self, n_actions):
        shape = (2**n_actions, 2)
        motions = np.zeros(shape)

        if self.actionLiteral:
            action_space_matrix = np.zeros(0)
        else:
            action_space_matrix = np.zeros((2 ** n_actions, n_actions))
        max_dist = 0
        for idx in range(shape[0]):
            action = binaryEncoding(idx, n_actions)
            if not self.actionLiteral:
                action_space_matrix[idx] = action
            motions[idx] = np.dot(action, self.movement)
            tmp = np.linalg.norm(motions[idx], ord=2)
            if tmp > max_dist:
                max_dist = tmp
        # Normalize to make maximium distance covered at a step be 1
        motions /= max_dist
        self.max_dist = max_dist
        return motions,action_space_matrix

    def new_pos_valid(self,new_pos,delta,reward,motion):
        # Check if hit obstacle
        if self.check_hit_teleport_tiles(new_pos):
            # Negative reward in wall area with constraint
            dist = np.linalg.norm(delta)
            reward += self.movement_reward * dist - 10 # small reward for moving
            if dist >= self.wall_width:
                print("ERROR: Step size bigger than wall width", new_pos, self.curr_pos, dist, delta, motion,
                      self.step_unit)
            self.curr_pos = np.array([0.99,0.01])
            reward += self.get_goal_rewards(self.curr_pos)
            # reward += self.open_gate_condition(self.curr_pos)
            return reward,True

        # If not hit obstacle and not hit region limit, return True, else return False
        elif self.valid_pos(new_pos,check_wall=False):
            dist = np.linalg.norm(delta)
            reward += self.movement_reward * dist  # small reward for moving
            if dist >= self.wall_width:
                print("ERROR: Step size bigger than wall width", new_pos, self.curr_pos, dist, delta, motion,
                      self.step_unit)
            self.curr_pos = new_pos
            reward += self.get_goal_rewards(self.curr_pos)
            # reward += self.open_gate_condition(self.curr_pos)
            return reward,True
        else:
            reward += self.collision_reward
            return reward,False

    def step(self, action,training=True):
        self.steps_taken += 1
        reward = 0
        # Check if previous state was end of MDP, if it was, then we are in absorbing state currently.
        # Terminal state has a Self-loop and a 0 reward
        term = self.is_terminal()
        if term:
            return self.curr_state, 0, term, {'No INFO implemented yet'}

        if self.actionLiteral:
            motion = np.dot(action,self.movement)[0]/self.max_dist
        else:
            motion = self.motions[action]  # Table look up for the impact/effect of the selected action
        reward += self.step_reward

        for i in range(self.repeat):
            if np.random.rand() < self.randomness:
                # Add noise some percentage of the time
                noise = np.random.rand(2)/1.415  # normalize by max L2 of noise
                delta = noise * self.step_unit  # Add noise some percentage of the time
            else:
                delta = motion * self.step_unit

            new_pos = self.curr_pos + delta  # Take a unit step in the direction of chosen action
            reward,break_loop = self.new_pos_valid(new_pos,delta,reward,motion)
            if not break_loop:
                break

            # To avoid overshooting the goal
            if self.is_terminal():
                break

            self.curr_state = self.make_state()

        if not training:
            # Track the positions being explored by the agent
            x_h, y_h = self.curr_pos*self.heatmap_scale
            self.heatmap[min(int(y_h), 99), min(int(x_h), 99)] += 1
        return self.curr_state.copy(), reward, self.is_terminal(), {'No INFO implemented yet'}


    def make_state(self):
        x, y = self.curr_pos
        state = [x, y]

        # Append lidar values
        for cosine, sine in self.lidar_angles:
            r, r_prv = 0, 0
            pos = (x+r*cosine, y+r*sine)
            while self.valid_pos(pos) and r < 0.5:
                r_prv = r
                r += self.step_unit
                pos = (x+r*cosine, y+r*sine)
            state.append(r_prv)

        return state

    def get_goal_rewards(self, pos):
        for key, val in self.reward_states.items():
            region, reward = val
            if reward and self.in_region(pos, region):
                self.reward_states[key] = (region, 0)  # remove reward once taken
                if self.debug: print("Got reward {} in {} steps!! ".format(reward, self.steps_taken))
                return reward
        return 0

    def get_reward_states(self):
        self.G1 = (0, 0.45, 0.05, 0.5)
        self.G2 = (0.70, 0.85, 0.75, 0.90)
        return {'G1': (self.G1, self.G1_reward),
                'G2': (self.G2, self.G2_reward)}

    def get_dynamic_obstacles(self):
        """
        :return: dict of objects, where key = obstacle shape, val = on/off
        """
        return {}

        # self.Gate = (0.15,0.25,0.35,0.3)
        # return {'Gate': (self.Gate, self.Gate_reward)}

    def get_static_teleport_tiles(self):
        self.T1 = (0.4, 0.4, 0.6, 0.6)


    def check_hit_teleport_tiles(self,pos):
        flag = False
        # Check collision with static obstacles
        if self.in_region(pos,self.T1):
            flag = True
        return flag

    def valid_pos(self, pos,check_wall=True):
        flag = True
        # Check boundary conditions
        if not self.in_region(pos, [0,0,1,1]):
            flag = False

        # Check collision with dynamic obstacles
        for key, val in self.dynamic_obs.items():
            region, cond = val
            if cond and self.in_region(pos, region):
                flag = False
                break
        return flag

    def is_terminal(self):
        if self.in_region(self.curr_pos, self.G1):
            return 1
        elif self.steps_taken >= self.max_steps:
            return 1
        else:
            return 0

    def in_region(self, pos, region):
        x0, y0 = pos
        x1, y1, x2, y2 = region
        if x0 >= x1 and x0 <= x2 and y0 >= y1 and y0 <= y2:
            return True
        else:
            return False