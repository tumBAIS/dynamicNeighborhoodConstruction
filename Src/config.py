import sys
from yaml import dump
from os import path,name
import Src.Utils.Utils as Utils
import numpy as np
import torch
from collections import OrderedDict

class Config(object):
    def __init__(self, args):

        # Path setup
        self.paths = OrderedDict()
        self.paths['root'] = path.abspath(path.join(path.dirname(__file__), '..'))

        # Reproducibility
        seed = args.seed
        np.random.seed(seed)
        torch.manual_seed(seed)

        # Copy all the variables from args to config
        self.__dict__.update(vars(args))

        # Save results after every certain number of episodes
        self.save_after = args.max_episodes // args.save_count if args.max_episodes > args.save_count else args.max_episodes

        # Add path to models
        folder_suffix = args.experiment + args.folder_suffix
        self.paths['Experiments'] = path.join(self.paths['root'], 'Experiments')
        self.paths['experiment'] = path.join(self.paths['Experiments'], args.env_name, args.algo_name, folder_suffix)

        path_prefix = [self.paths['experiment'], str(args.seed)]
        if name == 'nt':
            suffix = '\\'
        else:
            suffix = '/'
        self.paths['logs'] = path.join(*path_prefix, 'Logs'+suffix)
        self.paths['checkpoint'] = path.join(*path_prefix, 'Checkpoints'+suffix)
        self.paths['results'] = path.join(*path_prefix, 'Results'+suffix)

        # Create directories
        for (key, val) in self.paths.items():
            if key not in ['root', 'datasets', 'data']:
                Utils.create_directory_tree(val)

        # Save the all the configuration settings
        dump(args.__dict__, open(path.join(self.paths['experiment'], 'args.yaml'), 'w'), default_flow_style=False,
             explicit_start=True)

        # Output logging
        sys.stdout = Utils.Logger(self.paths['logs'], args.log_output)

        # Get the domain and algorithm
        self.env, self.gym_env, self.cont_actions = self.get_domain(args.env_name, args=args, debug=args.debug,
                                                               path=path.join(self.paths['root'], 'Environments'))
        self.env.seed(seed)


        # Hiddenlayer size
        self.hiddenLayerSize = args.hiddenLayerSize

        # Get the embedding paths, important for learned action representations
        self.get_embedding_paths(args)

        # Set Model
        self.algo = Utils.dynamic_load(path.join(self.paths['root'], 'Src', 'RL_Algorithms'), args.algo_name, load_class=True)

        # optimizer
        if args.optim == 'adam':
            self.optim = torch.optim.Adam
        elif args.optim == 'rmsprop':
            self.optim = torch.optim.RMSprop
        elif args.optim == 'sgd':
            self.optim = torch.optim.SGD
        else:
            raise ValueError('Undefined type of optmizer')


        print("=====Configurations=====\n", args)


    # Required for LAR
    def get_embedding_paths(self, args):
        if hasattr(args, 'true_embeddings'):
            if self.env_name[:9] == 'Gridworld' or self.env_name[:18]=='JointReplenishment':
                prefix = '' if self.fourier_order < 1 else 'Fourier_'
                self.paths['embedding'] = Utils.search(self.paths['root'], prefix + 'Grid' + str(args.n_actions),exact=True)
            if self.true_embeddings:
                self.reduced_action_dim = self.env.get_embeddings().shape[1]


    # Load the different domains
    def get_domain(self, tag, args, path, debug=True):
        if tag[:9] == 'Gridworld':
            obj = Utils.dynamic_load(path, tag, load_class=True)
            env = obj(mappingType=self.mapping,n_actions=args.n_actions, debug=debug,with_wall=args.wall,collision_reward=args.collision_rewards,constraint=args.constraint)
            return env, False, env.action_space.dtype == np.float32
        elif tag[:11] == 'Recommender':
            if self.mapping == 'minmax_mapping' or self.mapping == 'dnc_mapping':
                tag = 'Recommender_py_wo_actionspace_matrix'
                obj = Utils.dynamic_load(path, tag, load_class=True)
                env = obj(n_actions=args.n_actions,max_steps = args.max_steps,path_to_files=path)
            else:
                obj = Utils.dynamic_load(path, tag, load_class=True)
                env = obj(n_actions=args.n_actions,max_steps = args.max_steps,path_to_files=path)
            return env, False, env.action_space.dtype == np.float32
        elif tag[:18] == "JointReplenishment":
            obj = Utils.dynamic_load(path, tag, load_class=True)
            env = obj(smin=self.smin,smax=self.smax,n_items=args.n_actions,max_steps=args.max_steps,commonOrderCosts=args.commonOrderCosts,mappingType=self.mapping)
            return env, False, env.action_space.dtype == np.float32

if __name__ == '__main__':
    pass
