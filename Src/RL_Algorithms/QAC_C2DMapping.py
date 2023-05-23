import numpy as np
import torch
from torch import tensor, float32, long
import torch.nn.functional as F
from Src.Utils.Utils import MemoryBuffer, Trajectory
from Src.RL_Algorithms.Agent import Agent
from Src.Utils import Basis, Actor, Critic
from Src.MappingFunctions import ActionRepresentation, Knn, Dnc


# This function integrates the a Q-actor critic (QAC) algorithm with the different Continuous-to-Discrete (C2D) mapping functions, i.e., VAC, knn, DNC, MinMax, LAR, and
# contains the updates of actor and critic
class QAC_C2DMapping(Agent):
    def __init__(self, config):
        super(QAC_C2DMapping, self).__init__(config)

        # Obtain state features
        self.state_features = Basis.get_Basis(config=config)

        # Initialize mapping function
        self.action_space_matrix = config.env.action_space_matrix
        if config.mapping == 'learned_mapping':
            if not config.deepActionRep:
                self.mapping_fct = ActionRepresentation.Action_representation(state_dim=self.state_features.feature_dim,
                                                                             action_dim=self.action_dim, config=config)
            else:
                self.mapping_fct = ActionRepresentation.Action_representation_deep(state_dim=self.state_features.feature_dim,
                                                                             action_dim=self.action_dim, config=config)
        elif config.mapping == 'knn_mapping':
            self.mapping_fct = Knn.Knn(state_features=self.state_features,action_dim=self.action_dim,
                                      config=config,action_space_matrix=self.action_space_matrix)
        elif config.mapping == 'minmax_mapping':
            self.mapping_fct = Minmax.Minmax(action_dim=self.action_dim,
                                      config=config,action_space_matrix=self.action_space_matrix)
        elif config.mapping == 'no_mapping':
            self.mapping_fct = ActionRepresentation.No_Action_representation(state_dim=self.state_features.feature_dim,
                                                                         action_dim=self.action_dim, config=config,action_space=self.action_space_matrix)
        elif config.mapping == 'dnc_mapping':
            self.mapping_fct = Dnc.DNC(state_features=self.state_features,action_dim=self.action_dim,
                                      config=config,action_space_matrix=self.action_space_matrix)
        else:
            raise Exception("Invalid mapping specified")

        # Initial training phase required for LAR
        if config.mapping == 'learned_mapping':
            self.initial_phase = not config.true_embeddings and not config.load_embed
        elif config.mapping == 'knn_mapping' or config.mapping == 'minmax_mapping' or config.mapping == 'no_mapping' or config.mapping == 'dnc_mapping':
            self.initial_phase = False
            self.config.emb_lambda = 0 # No embedding updates on the fly

        # Initialize storage containers
        if self.config.env.actionLiteral:
            adim = self.mapping_fct.reduced_action_dim
            self.action_space_matrix_size = 0
        else:
            self.action_space_matrix_size = self.action_space_matrix.shape[0]
            adim = 1

        # Initialize critic
        self.critic = Critic.Qval(state_dim=self.state_features.feature_dim,action_dim=self.config.env.n_actions, config=config)

        # Initialize actor: If VAC, use categorical, if not use Gaussian policy
        if config.mapping != 'no_mapping':
            if self.config.env_name == 'Gridworld_py' or config.deepActor == 0:
                self.actor = Actor.Gaussian(action_dim=self.mapping_fct.reduced_action_dim,state_dim=self.state_features.feature_dim, config=config)
            else:
                self.actor = Actor.Gaussian_deep(action_dim=self.mapping_fct.reduced_action_dim,state_dim=self.state_features.feature_dim, config=config)
        else:
            self.actor = Actor.Categorical(action_dim=self.action_space_matrix_size,state_dim=self.state_features.feature_dim, config=config,action_space=self.action_space_matrix)

        # Mapping specific requirements
        if (config.mapping == 'dnc_mapping' or config.mapping == 'minmax_mapping') and config.env_name=='Recommender_py':
            atype=float32
        else:
            atype=long

        # Add a memory for training LAR and a container for training critic and actor based on single trajectories
        self.memory =   MemoryBuffer(max_len=self.config.buffer_size, state_dim=self.state_dim,
                                     action_dim=adim, atype=atype, config=config,
                                     dist_dim=self.mapping_fct.reduced_action_dim)  # off-policy
        self.trajectory = Trajectory(max_len=self.config.batch_size, state_dim=self.state_dim,
                                     action_dim=adim, atype=atype, config=config,
                                     dist_dim=self.mapping_fct.reduced_action_dim)  # on-policy

        # Define learning modules -- in LAR we have 3 learning modules as we have to additionally train the SL predictor
        if config.mapping == 'learned_mapping':
            self.modules = [('actor', self.actor), ('critic', self.critic),('mapping_fct', self.mapping_fct)]
        else:
            self.modules = [('actor', self.actor), ('critic', self.critic)]

        # Define the action space matrix as tensor
        self.action_space_matrix = torch.tensor(self.action_space_matrix,dtype=torch.float32)

        # When requiring an action space matrix, define action id to action transformation, not required when using DNC/MinMax
        if not self.config.env.actionLiteral:
            self.action_trafo = self.action_id_to_tensor
        else:
            self.action_trafo = self.action_to_tensor # Dummy function



    def get_action(self, state,training):
        if self.initial_phase:
            # take random actions (uniformly in actual action space) to observe the interactions initially
            action = np.random.randint(self.action_dim)
            a_hat = self.mapping_fct.get_embedding(action).cpu().view(-1).data.numpy()
        else:
            state = tensor(state, dtype=float32, requires_grad=False)
            state = self.state_features.forward( state.view(1, -1))
            a_hat, _ = self.actor.get_action(state, training)
            action = self.mapping_fct.get_best_match(a_hat,state,self.critic)

            a_hat = a_hat.cpu().view(-1).data.numpy()

        return action, a_hat

    def update(self, s1, a1, a_hat_1, r1, s2, done):
        loss_actor = 0
        loss_critic = 0
        self.memory.add(s1, a1, a_hat_1, r1, s2, int(done != 1))
        if not self.initial_phase:
            self.trajectory.add(s1, a1, a_hat_1, r1, s2, int(done != 1))
            if self.trajectory.size >= self.config.batch_size or done:
                loss_actor,loss_critic = self.optimize()
                self.trajectory.reset()
        else:
            if self.memory.length >= self.config.buffer_size:
                self.initial_phase_training(max_epochs=self.config.initial_phase_epochs)
        return loss_actor,loss_critic

    def action_to_tensor(self,a1,a2):
        return a1,a2
    def action_id_to_tensor(self,a1,a2):
        id = a1.cpu().numpy()[0][0]
        action2= self.action_space_matrix[a2].view(1, -1)
        return self.action_space_matrix[id].view(1,-1),action2

    def optimize(self):
        s1, a1, a_hat_1, r1, s2, not_absorbing = self.trajectory.get_all()

        a2, a_hat_2 = self.get_action(s2,training=True)
        s1 = self.state_features.forward(s1)
        s2 = self.state_features.forward(s2)

        action1,action2 = self.action_trafo(a1,a2)


        # Define critic loss
        next_val = self.critic.forward(s2,action2).detach()
        val_exp  = r1 + self.config.gamma * next_val * not_absorbing
        val_pred = self.critic.forward(s1,action1)
        loss_critic = F.huber_loss(val_pred, val_exp) # F.mse_loss

        # Define actor loss
        # see https://github.com/pytorch/pytorch/issues/9442 for log of pdf
        td_error = (val_exp-val_pred).detach()
        logp, dist = self.actor.get_log_prob(s1, a_hat_1)
        loss_actor = -1.0 * torch.sum(td_error*logp)


        # Take one policy gradient step
        loss = loss_actor + loss_critic
        self.step(loss, clip_norm=1)


        # Take one unsupervised step
        if not self.config.true_embeddings and self.config.emb_lambda > 0:
            s1, a1, _, _, s2, _ = self.memory.sample(batch_size=self.config.sup_batch_size)
            self.self_supervised_update(s1, a1, s2, reg=self.config.emb_lambda)

        return loss_actor.cpu().data.numpy(),loss_critic.cpu().data.numpy()



    def self_supervised_update(self, s1, a1, s2, reg=1):
        self.clear_gradients()  # clear all the gradients from last run

        # If doing online updates, sharing the state features might be problematic!
        s1 = self.state_features.forward(s1)
        s2 = self.state_features.forward(s2)

        # ------------ optimize the embeddings ----------------
        loss_act_rep = self.mapping_fct.unsupervised_loss(s1, a1.view(-1), s2, normalized=True) * reg
        loss_act_rep.backward()

        # Directly call the optimizer's step fn to bypass lambda traces (if any)
        self.mapping_fct.optim.step()
        self.state_features.optim.step()

        return loss_act_rep.item()

    def initial_phase_training(self, max_epochs=-1):
        # change optimizer to Adam for unsupervised learning
        self.mapping_fct.optim = torch.optim.Adam(self.mapping_fct.parameters(), lr=1e-3)
        self.state_features.optim = torch.optim.Adam(self.state_features.parameters(), lr=1e-3)
        initial_losses = []

        print("Inital training phase started...")
        for counter in range(max_epochs):
            losses = []
            for s1, a1, _, _, s2, _ in self.memory.batch_sample(batch_size=self.config.sup_batch_size, randomize=True):
                loss = self.self_supervised_update(s1, a1, s2)
                losses.append(loss)

            initial_losses.append(np.mean(losses))
            if counter % 1 == 0:
                print("Epoch {} loss:: {}".format(counter, np.mean(initial_losses[-10:])))
                if self.config.only_phase_one:
                    self.save()
                    print("Saved..")

            # Terminate initial phase once action representations have converged.
            if len(initial_losses) >= 20 and np.mean(initial_losses[-10:]) + 1e-5 >= np.mean(initial_losses[-20:]):
                print("Converged...")
                break

        # Reset the optim to whatever is there in config
        self.mapping_fct.optim = self.config.optim(self.mapping_fct.parameters(), lr=self.config.embed_lr)
        self.state_features.optim = self.config.optim(self.state_features.parameters(), lr=self.config.state_lr)

        print('... Initial training phase terminated!')
        self.initial_phase = False
        self.save()

        if self.config.only_phase_one:
            exit()


