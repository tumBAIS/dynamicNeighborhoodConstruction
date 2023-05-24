import argparse
from datetime import datetime


class Parser(object):
    def __init__(self):
        parser = argparse.ArgumentParser()

        # Seed for reproducibility
        parser.add_argument("--seed", default=1, help="seed for variance testing",type=int)


        # General parameters
        parser.add_argument("--save_count", default=10, help="Number of checkpoints for saving results and model", type=int)
        parser.add_argument("--optim", default='sgd', help="Optimizer type", choices=['adam', 'sgd', 'rmsprop'])
        parser.add_argument("--log_output", default='term_file', help="Log all the print outputs",choices=['term_file', 'term', 'file'])
        parser.add_argument("--debug", default=True, type=self.str2bool, help="Debug mode on/off")
        parser.add_argument("--save_model", default=True, type=self.str2bool, help="flag to save model checkpoints")

        # For documentation purposes
        now = datetime.now()
        timestamp = str(now.month) + '|' + str(now.day) + '|' + str(now.hour) + ':' + str(now.minute) + ':' + str(now.second)
        parser.add_argument("--timestamp", default=timestamp, help="Timestamp to prefix experiment dumps")
        parser.add_argument("--folder_suffix", default='default', help="folder name suffix")
        parser.add_argument("--experiment", default='run', help="Name of the experiment")

        # Which mapping to use
        parser.add_argument("--mapping",default="dnc_mapping",help = "Type of mapping function", choices= ["no_mapping","knn_mapping","dnc_mapping","learned_mapping"]) # minmax achieved by using dnc_mapping and setting --cooling = 0 below

        self.environment_parameters(parser)  # Environment parameters

        # Settings for different mapping functions
        self.LAR_parameters(parser)
        self.KNN_parameters(parser)
        self.DNC_parameters(parser)

        # General settings for QAC algorithm
        self.QAC_parameters(parser)

        self.parser = parser

    def str2bool(self, text):
        if text == 'True':
            arg = True
        elif text == 'False':
            arg = False
        else:
            raise argparse.ArgumentTypeError('Boolean value expected.')
        return arg

    def get_parser(self):
        return self.parser


    def environment_parameters(self, parser):
        parser.add_argument("--algo_name", default='QAC_C2DMapping', help="RL algorithm")
        parser.add_argument("--env_name", default='Gridworld_py', help="Environment to run the code", choices=["Gridworld_py","Recommender_py","JointReplenishment_py"])
        parser.add_argument("--n_actions", default=12, help="size of the action vector", type=int)
        parser.add_argument("--max_episodes", default=int(100), help="maximum number of episodes", type=int)
        parser.add_argument("--max_steps", default=150, help="maximum steps per episode", type=int)

        # Gridworld specific arguments
        parser.add_argument("--wall", default=True, help="With wall", type=self.str2bool)
        parser.add_argument("--collision_rewards", default=0, help="degree of collision rewards", type=float)
        parser.add_argument("--constraint", default='hard', help="type of constraint")

        # Joint replenishment specific
        parser.add_argument("--smin", default=0, help="minimum order level", type=int)
        parser.add_argument("--smax", default=66, help="max order level", type=int)
        parser.add_argument("--commonOrderCosts", default=75, help="common order costs for the inventory env", type=int)
        parser.add_argument("--actionLiteral", default=1, help="mapping return literal action instead of label", type=int)


    def LAR_parameters(self, parser):
        parser.add_argument("--true_embeddings", default=False, help="Use ground truth embeddings or not?", type=self.str2bool)
        parser.add_argument("--only_phase_one", default=False, help="Only phase1 training", type=self.str2bool)
        parser.add_argument("--emb_lambda", default=0.9, help="Lagrangian for learning embedding on the fly", type=float)
        parser.add_argument("--embed_lr", default=1e-4, help="Learning rate of action embeddings", type=float)
        parser.add_argument("--emb_reg", default=1e-5, help="L2 regularization for embeddings", type=float)
        parser.add_argument("--reduced_action_dim",default=2, help="dimensions of action embeddings", type=int)
        parser.add_argument("--load_embed", default=False, type=self.str2bool, help="Retrain flag")
        parser.add_argument("--deepActionRep", default=True, help="use deep NN for learning action reps or not",type=self.str2bool)
        parser.add_argument("--sup_batch_size", default=64, help="(64)Supervised learning Batch size", type=int)
        parser.add_argument("--initial_phase_epochs", default=4, help="maximum number of episodes (500)", type=int)  ##### CHANGED FROM BASE SETTING (500)

    def KNN_parameters(self,parser):
        # KNN specific
        parser.add_argument("--knns", default=2, type=int,help="No. of k nearest neighbours for knn mapping")  # knn-1,knn-2,knn-0.5%

    # DNC and MinMax specific params, MinMax obtained by setting
    def DNC_parameters(self,parser):

        parser.add_argument("--no_neighbours",default=0,type=int,help="use dnc without neighbors for debugging")

        # Neighborhood exploration method, only a_clip and
        parser.add_argument("--a_clip",default=1, help="clipping for minmax", type=float)
        parser.add_argument("--clipped_decimals", default=0, type=int,help="How to clip protoactions for dnc and minmax")
        parser.add_argument("--maximum_greedy_search_steps", default=1, type=int,help="amplitude of perturbation")
        parser.add_argument("--perturb_scaler", default=1, type=float,help="decimals with which we perturb")
        parser.add_argument("--perturbation_range", default=5, type=int, help="decimals with which we perturb")

        # Sim Annealing
        parser.add_argument("--cooling",default=0.25,type=float,help="the speed of cooling for neighborhood construct. mapping")
        parser.add_argument("--initialAcceptance", default=0.9, type=float,help="the speed of cooling for neighborhood construct. mapping")
        parser.add_argument("--acceptanceCooling", default=0.225, type=float,help="the speed of cooling for neighborhood construct. mapping")

    def QAC_parameters(self, parser):
        parser.add_argument("--gamma", default=0.999, help="Discounting factor", type=float)
        parser.add_argument("--actor_lr", default=1e-2, help="(1e-2) Learning rate of actor", type=float)
        parser.add_argument("--critic_lr", default=1e-2, help="(1e-2) Learning rate of critic/baseline", type=float)
        parser.add_argument("--state_lr", default=1e-3, help="Learning rate of state features", type=float)
        parser.add_argument("--gauss_variance", default=0.5, help="Variance for gaussian policy", type=float) # 1 original setting

        parser.add_argument("--hiddenLayerSize", default=32, help="size of hiddenlayer", type=int)
        parser.add_argument("--hiddenActorLayerSize", default=128, help="size of hiddenlayer", type=int)
        parser.add_argument("--deepActor", default=False, help="if we want a deep actor", type=self.str2bool)

        parser.add_argument("--actor_scaling_factor_mean", default=1, help="scale output of actor mean by x", type=float)

        parser.add_argument("--fourier_coupled", default=True, help="Coupled or uncoupled fourier basis", type=self.str2bool)
        parser.add_argument("--fourier_order", default=3, help="Order of fourier basis, " + "(if > 0, it overrides neural nets)", type=int)

        parser.add_argument("--buffer_size", default=int(100), help="Size of memory buffer (6e5)", type=int)
        parser.add_argument("--batch_size", default=1, help="Batch size", type=int)
