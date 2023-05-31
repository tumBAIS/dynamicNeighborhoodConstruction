from __future__ import print_function
import numpy as np
import torch
from torch import tensor, float32
from torch.autograd import Variable
import torch.nn as nn
import shutil
import random
from collections import deque
import itertools
import matplotlib.pyplot as plt
from os import path, mkdir, listdir, fsync, name
import importlib
from time import time
import sys
from torch.utils.data import Dataset

np.random.seed(0)
torch.manual_seed(0)
dtype = torch.FloatTensor

class Logger(object):
    fwrite_frequency = 1800  # 30 min * 60 sec
    temp = 0

    def __init__(self, log_path, method): # restore
        self.terminal = sys.stdout
        self.file = 'file' in method
        self.term = 'term' in method
        self.log_path = log_path
        self.log = open(path.join(log_path, "logfile.log"), "w")


    def write(self, message):
        if self.term:
            self.terminal.write(message)

        if self.file:
            self.log.write(message)

            # Save the file frequently
            if (time() - self.temp) > self.fwrite_frequency:
                self.flush()
                self.temp = time()

    def flush(self):
        #this flush method is needed for python 3 compatibility.
        #this handles the flush command by doing nothing.
        #you might want to specify some extra behavior here.

        # Save the contents of the file without closing
        # https://stackoverflow.com/questions/19756329/can-i-save-a-text-file-in-python-without-closing-it
        # WARNING: Time consuming process, Makes the code slow if too many writes
        self.log.flush()
        fsync(self.log.fileno())

def save_plots(rewards,run_time, config):
    np.save(config.paths['results'] + "rewards", rewards)
    np.save(config.paths['results'] + "time", run_time)
    if config.debug:
        if 'Grid' in config.env_name or 'room' in config.env_name:
            # Save the heatmap
            plt.figure()
            plt.title("Exploration Heatmap")
            plt.xlabel("100x position in x coordinate")
            plt.ylabel("100x position in y coordinate")
            plt.imshow(config.env.heatmap, cmap='hot', interpolation='nearest', origin='lower')
            plt.savefig(config.paths['results'] + 'heatmap.png')
            np.save(config.paths['results'] + "heatmap", config.env.heatmap)
            config.env.heatmap.fill(0)  # reset the heat map
            plt.close()

        plt.figure()
        plt.ylabel("Total return")
        plt.xlabel("Episode")
        plt.title("Performance")
        plt.plot(rewards)
        plt.savefig(config.paths['results'] + "performance.png")
        plt.close()

def save_plots_test_runs(test_returns,test_std,step_time,config):
    np.save(config.paths['results'] + "test_returns_mean", test_returns)
    np.save(config.paths['results'] + "test_returns_std", test_std)
    np.save(config.paths['results'] + "step_time", step_time)
    x = config.save_after * np.arange(0, len(test_returns))
    plt.figure()
    plt.ylabel("Total return")
    plt.xlabel("Episode")
    plt.title("Performance")
    plt.plot(x,test_returns,'k',color='#CC4F1B')
    plt.fill_between(x,np.array(test_returns)+np.array(test_std),np.array(test_returns)-np.array(test_std),alpha=0.5, edgecolor='#CC4F1B', facecolor='#FF9848')
    plt.savefig(config.paths['results'] + "performance_test_runs.png")
    plt.close()


class NeuralNet(nn.Module):
    def __init__(self):
        super(NeuralNet, self).__init__()
        self.ctr = 0
        self.nan_check_fequency = 10000

    # def custom_weight_init(self):
    #     # Initialize the weight values
    #     for m in self.modules():
    #         weight_init(m)

    def update(self, loss, retain_graph=False, clip_norm=False):
        self.optim.zero_grad()  # Reset the gradients
        loss.backward(retain_graph=retain_graph)
        self.step(clip_norm)

    def step(self, clip_norm):
        if clip_norm:
            torch.nn.utils.clip_grad_norm_(self.parameters(), clip_norm)
        self.optim.step()
        self.check_nan()

    def save(self, filename):
        torch.save(self.state_dict(), filename)

    def load(self, filename):
        self.load_state_dict(torch.load(filename))

    def check_nan(self):
        # Check for nan periodically
        self.ctr += 1
        if self.ctr == self.nan_check_fequency:
            self.ctr = 0
            # Note: nan != nan  #https://github.com/pytorch/pytorch/issues/4767
            for name, param in self.named_parameters():
                if (param != param).any():
                    raise ValueError(name + ": Weights have become nan... Exiting.")

    def reset(self):
        return


def binaryEncoding(num, size,level=1):
    binary = np.zeros(size)
    i = -1
    while num > 0:
        binary[i] = num % (level+1)
        num = num//(level+1)
        i -= 1
    return binary

# def stablesoftmax(x):
#     """Compute the softmax of vector x in a numerically stable way."""
#     shiftx = x - np.max(x)
#     exps = np.exp(shiftx)
#     return exps / np.sum(exps)



def pairwise_distances(x, y):
    '''
    Input: x is a Nxd matrix
           y is a Mxd matirx
    Output: dist is a NxM matrix where dist[i,j] is the square norm between x[i,:] and y[j,:]
    i.e. dist[i,j] = ||x[i,:]-y[j,:]||^2

    Advantage: Less memory requirement O(M*d + N*d + M*N) instead of O(N*M*d)
    Computationally more expensive? Maybe, Not sure.
    adapted from: https://discuss.pytorch.org/t/efficient-distance-matrix-computation/9065/2
    '''
    x_norm = (x ** 2).sum(1).view(-1, 1)
    y_norm = (y ** 2).sum(1).view(1, -1)
    y_t = torch.transpose(y, 0, 1)

    # a^2 + b^2 - 2ab
    dist = x_norm + y_norm - 2.0 * torch.mm(x, y_t)
    # dist[dist != dist] = 0 # replace nan values with 0
    return dist


class Space:
    def __init__(self, low=[0], high=[1], dtype=np.uint8, size=-1):
        if size == -1:
            self.shape = np.shape(low)
        else:
            self.shape = (size, )
        self.low = np.array(low)
        self.high = np.array(high)
        self.dtype = dtype
        self.n = len(self.low)


def save_training_checkpoint(state, is_best, episode_count):
    """
    Saves the models, with all training parameters intact
    :param state:
    :param is_best:
    :param filename:
    :return:
    """
    filename = str(episode_count) + 'checkpoint.path.rar'
    torch.save(state, filename)
    if is_best:
        shutil.copyfile(filename, 'model_best.pth.tar')


def search(dir, name, exact=False):
    all_files = listdir(dir)
    for file in all_files:
        if exact and name == file:
            return path.join(dir, name)
        if not exact and name in file:
            return path.join(dir, name)
    else:
        # recursive scan
        for file in all_files:
            if file == 'Experiments':
                continue
            _path = path.join(dir, file)
            if path.isdir(_path):
                location = search(_path, name, exact)
                if location:
                    return location

def dynamic_load(dir, name, load_class=False):
    try:
        abs_path = search(dir, name).split('/')[1:]

        if len(abs_path) == 0:
            abs_path = search(dir, name).split('\\')[1:]
        pos = abs_path.index('dynamicNeighborhoodConstruction')

        module_path = '.'.join([str(item) for item in abs_path[pos + 1:]])
        print("Module path: ", module_path, name)
        if load_class:
            obj = getattr(importlib.import_module(module_path), name)
        else:
            obj = importlib.import_module(module_path)
        print("Dynamically loaded from: ", obj)
        return obj
    except:
        raise ValueError("Failed to dynamically load the class: " + name )

def check_n_create(dir_path, overwrite=False):
    try:
        if not path.exists(dir_path):
            mkdir(dir_path)
        else:
            if overwrite:
               shutil.rmtree(dir_path)
               mkdir(dir_path)
    except FileExistsError:
        print("\n ##### Warning File Exists... perhaps multi-threading error? \n")

def create_directory_tree(dir_path):
    if name == 'nt':#windows
        sepa = '\\'
    else:
        sepa = '/'
    dir_path = str.split(dir_path, sep=sepa)[1:-1]  #Ignore the blank characters in the start and end of string
    for i in range(len(dir_path)):
        check_n_create(path.join(sepa, *(dir_path[:i + 1])))

def remove_directory(dir_path):
    shutil.rmtree(dir_path, ignore_errors=True)


def clip_norm(params, max_norm=1):
    # return params
    norm_param = []
    for param in params:
        norm = np.linalg.norm(param, 2)
        if norm > max_norm:
            norm_param.append(param/norm * max_norm)
        else:
            norm_param.append(param)
    return norm_param



class MemoryBuffer:
    """
    Pre-allocated memory interface for storing and using Off-policy trajectories

    Note: slight abuse of notation.
          sometimes Code treats 'dist' as extra variable and uses it to store other things, like: prob, etc.
    """
    def __init__(self, max_len, state_dim, action_dim, atype, config, dist_dim=1, stype=float32):

        self.s1 = torch.zeros((max_len, state_dim), dtype=stype, requires_grad=False)
        self.a1 = torch.zeros((max_len, action_dim), dtype=atype, requires_grad=False)
        self.dist = torch.zeros((max_len, dist_dim), dtype=float32, requires_grad=False)
        self.r1 = torch.zeros((max_len, 1), dtype=float32, requires_grad=False)
        self.s2 = torch.zeros((max_len, state_dim), dtype=stype, requires_grad=False)
        self.done = torch.zeros((max_len, 1), dtype=float32, requires_grad=False)

        self.length = 0
        self.max_len = max_len
        self.atype = atype
        self.stype = stype
        self.config = config

    @property
    def size(self):
        return self.length

    def reset(self):
        self.length = 0

    def _get(self, ids):
        return self.s1[ids], self.a1[ids], self.dist[ids], self.r1[ids], self.s2[ids], self.done[ids]

    def batch_sample(self, batch_size, randomize=True):
        if randomize:
            indices = np.random.permutation(self.length)
        else:
            indices = np.arange(self.length)

        for ids in [indices[i:i + batch_size] for i in range(0, self.length, batch_size)]:
            yield self._get(ids)

    def sample(self, batch_size):
        count = min(batch_size, self.length)
        return self._get(np.random.choice(self.length, count))

    def add(self, s1, a1, dist, r1, s2, done):
        pos = self.length
        if self.length < self.max_len:
            self.length = self.length + 1
        else:
            pos = np.random.randint(self.max_len)

        self.s1[pos] = torch.tensor(s1, dtype=self.stype)
        self.a1[pos] = torch.tensor(a1, dtype=self.atype)
        self.dist[pos] = torch.tensor(dist)
        self.r1[pos] = torch.tensor(r1)
        self.s2[pos] = torch.tensor(s2, dtype=self.stype)
        self.done[pos] = torch.tensor(done)

class Trajectory:
    """
    Pre-allocated memory interface for storing and using on-policy trajectories

    Note: slight abuse of notation.
          sometimes Code treats 'dist' as extra variable and uses it to store other things, like: prob, etc.
    """
    def __init__(self, max_len, state_dim, action_dim, atype, config, dist_dim=1, stype=float32):

        self.s1 = torch.zeros((max_len, state_dim), dtype=stype, requires_grad=False)
        self.a1 = torch.zeros((max_len, action_dim), dtype=atype, requires_grad=False)
        self.r1 = torch.zeros((max_len, 1), dtype=float32, requires_grad=False)
        self.s2 = torch.zeros((max_len, state_dim), dtype=stype, requires_grad=False)
        self.done = torch.zeros((max_len, 1), dtype=float32, requires_grad=False)
        self.dist = torch.zeros((max_len, dist_dim), dtype=float32, requires_grad=False)

        self.ctr = 0
        self.max_len = max_len
        self.atype = atype
        self.stype= stype
        self.config = config

    def add(self, s1, a1, dist, r1, s2, done):
        if self.ctr == self.max_len:
            raise OverflowError

        self.s1[self.ctr] = torch.tensor(s1, dtype=self.stype)
        self.a1[self.ctr] = torch.tensor(a1, dtype=self.atype)
        self.dist[self.ctr] = torch.tensor(dist)
        self.r1[self.ctr] = torch.tensor(r1)
        self.s2[self.ctr] = torch.tensor(s2, dtype=self.stype)
        self.done[self.ctr] = torch.tensor(done)

        self.ctr += 1

    def reset(self):
        self.ctr = 0

    @property
    def size(self):
        return self.ctr

    def _get(self, ids):
        return self.s1[ids], self.a1[ids], self.dist[ids], self.r1[ids], self.s2[ids], self.done[ids]

    def get_current_transitions(self):
        pos = self.ctr
        return self.s1[:pos], self.a1[:pos], self.dist[:pos], self.r1[:pos], self.s2[:pos], self.done[:pos]

    def get_all(self):
        return self.s1, self.a1, self.dist, self.r1, self.s2, self.done

    def get_latest(self):
        return self._get([-1])

    def batch_sample(self, batch_size, nth_return):
        # Compute the estimated n-step gamma return
        R = nth_return
        for idx in range(self.ctr-1, -1, -1):
            R = self.r1[idx] + self.config.gamma * R
            self.r1[idx] = R

        # Genreate random sub-samples from the trajectory
        perm_indices = np.random.permutation(self.ctr)
        for ids in [perm_indices[i:i + batch_size] for i in range(0, self.ctr, batch_size)]:
            yield self._get(ids)

