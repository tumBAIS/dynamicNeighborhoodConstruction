""""
Implements the solution of the MIP describing the NN, which implements the Q-function. Currently hard-coded for
2 NN layers, but arbitrary number of neurons per layer
"""


import numpy as np
import gurobipy as gp
from gurobipy import GRB
import torch


## Comments: Currently hard-coded for 2 NN layers, but arbitrary number of neurons per layer
class MathProg:
    def __init__(self,critic,state_dim,action_dim,config):
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.M = config.bigM

        # Neighborhood restriction variables and parameters
        self.max_hamming_distance = config.max_hamming_distance
        self.eps = config.perturb_scaler
        self.lb_a = config.smin*np.ones(self.action_dim).astype(int)
        self.ub_a = config.smax*np.ones(self.action_dim).astype(int)
        self.mu = 1 / (self.ub_a - self.lb_a)
        self.neurons = [config.hiddenLayerSize,config.hiddenLayerSize]
        self.scale_down_NN_factor = 1.0
        self.init_gb_model(critic)

    def init_gb_model(self,critic):
        self.gb_model = gp.Model('MathProg')
        self.a_plus = self.gb_model.addMVar((1,self.action_dim),vtype=GRB.INTEGER,lb=0, name='a_plus')
        self.a_minus = self.gb_model.addMVar((1, self.action_dim), vtype=GRB.INTEGER, lb=0, name='a_minus')
        self.a = self.gb_model.addMVar((1,self.action_dim),vtype=GRB.INTEGER, name='a')
        self.a[0].setAttr("lb", self.lb_a)
        self.a[0].setAttr("ub",self.ub_a)

        state = torch.tensor([np.ones(self.state_dim)],dtype=torch.float32)
        state_input_layer_output = torch.tensor(critic.relu(critic.fc1.forward(state)))/self.scale_down_NN_factor
        state_input_layer_output = np.array(state_input_layer_output[0].numpy(), dtype=np.float64)
        weights2 = np.array(critic.fc2.weight.detach().numpy(), dtype=np.float64)
        weights3 = np.array(critic.fc3.weight.detach().numpy(), dtype=np.float64)
        bias2 = np.array(critic.fc2.bias.detach().numpy(), dtype=np.float64)

        weights2 /= self.scale_down_NN_factor
        weights3 /= self.scale_down_NN_factor
        bias2 /= self.scale_down_NN_factor

        self.y = []
        self.z = []

        self.y.append(self.gb_model.addMVar((1,self.neurons[1]),vtype=GRB.CONTINUOUS, lb=0, name='y'+str(1)))
        self.z.append(self.gb_model.addMVar((1,self.neurons[1]), vtype=GRB.BINARY, name='z' + str(1)))
        l = 1


        ## NN Constraints according to Heeswijk and Poutré (2020), p.6(1068)
        ## index for layer in NN (only for critic, as critic counts ReLU layers separately, which are normally not own layers)
        ## l index for layer in NN (apart from critic)
        ## j index for neuron of predecessor layer
        ## k index for neuron of successor layer
        self.gb_model.addConstrs((gp.quicksum(weights2[k][j]*self.a[0][j-self.neurons[0]]
                                          for j in range(self.neurons[0],self.neurons[0]+self.action_dim)) + gp.quicksum(weights2[k][j]*state_input_layer_output[j]
                                          for j in range(0,self.neurons[0])) +bias2[k] <= self.y[0][0][k] for k in range(self.neurons[1])),name=str(1)+str(l))
        # Constraint 3 in Heeswijk and Poutré (2020), p.6(1068)
        self.gb_model.addConstrs(( self.y[0][0][k] <= self.M*self.z[0][0][k] for k in range(self.neurons[1])),name=str(2)+str(l))
        # Constraint 4 in Heeswijk and Poutré (2020), p.6(1068)
        self.gb_model.addConstrs((self.y[0][0][k] <= self.M * (1-self.z[0][0][k])+ gp.quicksum(weights2[k][j]*self.a[0][j-self.neurons[0]]
                                          for j in range(self.neurons[0],self.neurons[0]+self.action_dim)) + gp.quicksum(weights2[k][j]*state_input_layer_output[j]
                                          for j in range(0,self.neurons[0])) +bias2[k] for k in range(self.neurons[1])),name=str(3)+str(l))
        # Constraint 5 in Heeswijk and Poutré (2020), p.6(1068)
        self.gb_model.addConstrs((self.z[0][0][k] >= (gp.quicksum(weights2[k][j]*self.a[0][j-self.neurons[0]]
                                          for j in range(self.neurons[0],self.neurons[0]+self.action_dim)) + gp.quicksum(weights2[k][j]*state_input_layer_output[j]
                                          for j in range(0,self.neurons[0])) + bias2[k]) / self.M
             for k in range(self.neurons[1])), name=str(4)+str(l))
        # Constraint 6 in Heeswijk and Poutré (2020), p.6(1068)
        self.gb_model.addConstrs((self.z[0][0][k] <= 1+(gp.quicksum(weights2[k][j]*self.a[0][j-self.neurons[0]]
                                          for j in range(self.neurons[0],self.neurons[0]+self.action_dim)) + gp.quicksum(weights2[k][j]*state_input_layer_output[j]
                                          for j in range(0,self.neurons[0])) +bias2[k])/self.M for k in range(self.neurons[1])),name=str(5)+str(l))
        self.gb_model.setObjective(gp.quicksum(weights3[0][j]*self.y[0][0][j]
                                                      for j in range(self.neurons[1])), GRB.MAXIMIZE)
        self.gb_model.update()
        self.gb_model.Params.LogToConsole = 0
        self.gb_model.Params.IntFeasTol = 10 ** (-6)

        ## Comment in if more precise solutions required
        # self.gb_model.setParam("Presolve", 0)

    def solve(self,critic,state,rebuild_constraints,base_action):

        RHS_inputlayer = np.zeros(5*self.neurons[1])
        weights_input_layer = np.array(critic.fc2.weight.detach().numpy(),dtype=np.float64)

        state_input_layer_output = torch.tensor(critic.relu(critic.fc1.forward(state)), dtype=torch.float64)/self.scale_down_NN_factor
        state_input_layer_output = np.array(state_input_layer_output[0].numpy())
        input_bias = np.array(critic.fc2.bias.detach().numpy(),dtype=np.float64)

        weights_input_layer /= self.scale_down_NN_factor
        input_bias /= self.scale_down_NN_factor
        n = self.neurons[1]
        RHS_inputlayer[0:n] = -(np.matmul(weights_input_layer[:, :self.neurons[0]], state_input_layer_output) + input_bias)
        RHS_inputlayer[n:n+2*n] = 0.0
        RHS_inputlayer[2*n:3*n] = np.matmul(weights_input_layer[:, :self.neurons[0]], state_input_layer_output) + input_bias + self.M
        RHS_inputlayer[3*n:4*n] = (np.matmul(weights_input_layer[:, :self.neurons[0]], state_input_layer_output) + input_bias)/self.M
        RHS_inputlayer[4*n:5*n] = (np.matmul(weights_input_layer[:, :self.neurons[0]], state_input_layer_output) + input_bias)/self.M +1
        self.gb_model.setAttr("RHS", self.gb_model.getConstrs()[0:5*n], RHS_inputlayer)

        # ## Maximum perturbation distance
        lower_pert = np.maximum(base_action-self.eps,self.lb_a*np.ones(self.action_dim)).astype(int)
        upper_pert = np.minimum(base_action+self.eps,self.ub_a*np.ones(self.action_dim)).astype(int)
        self.a[0].setAttr("lb",lower_pert)
        self.a[0].setAttr("ub",upper_pert)
        self.lb = lower_pert[:]
        self.ub = upper_pert[:]
        self.mu = 1 / (self.ub - self.lb)

        ## Local branching constraint according to A. Lodi and M. Fischetti (2003)
        j1 = np.where((base_action == self.lb))[0]
        j2 = np.where((base_action == self.ub))[0]
        j3 = np.where((base_action != self.lb) & (base_action != self.ub))[0]
        self.gb_model.addConstr(gp.quicksum(self.mu[j]*(self.a[0][j]-self.lb[j]) for j in j1) + gp.quicksum(self.mu[j]*(self.ub[j]-self.a[0][j]) for j in j2)
                                +gp.quicksum(self.mu[j]*(self.a_plus[0][j]+self.a_minus[0][j]) for j in j3) <= self.max_hamming_distance,name='lbr0')
        k = 1
        for j in j3:
            self.gb_model.addConstr(base_action[j]+self.a_plus[0][j]-self.a_minus[0][j]==self.a[0][j],name='lbr'+str(k))
            k+=1

        ## Five NN Constraints according to Heeswijk and Poutré (2020), p.6(1068)
        try:
            if rebuild_constraints:
                weights_layer2 = np.array(critic.fc3.weight.detach().numpy(),dtype=np.float64)
                weights_layer2 /= self.scale_down_NN_factor

                # # Constraint coefficients
                for l in range(0,self.neurons[1]):
                    for k in range(self.neurons[0],self.neurons[0]+self.action_dim):
                        self.gb_model.chgCoeff(self.gb_model.getConstrs()[l],self.a[0][k-self.neurons[0]].tolist(),weights_input_layer[l,k])
                        self.gb_model.chgCoeff(self.gb_model.getConstrs()[l+2*self.neurons[1]],self.a[0][k-self.neurons[0]].tolist(),-weights_input_layer[l,k])
                        self.gb_model.chgCoeff(self.gb_model.getConstrs()[l+3*self.neurons[1]],self.a[0][k-self.neurons[0]].tolist(),-weights_input_layer[l,k]/self.M)
                        self.gb_model.chgCoeff(self.gb_model.getConstrs()[l+4*self.neurons[1]],self.a[0][k-self.neurons[0]].tolist(),-weights_input_layer[l,k]/self.M)
                for i in range(0,self.neurons[len(self.neurons)-1]):
                    self.y[0][0][i].Obj = weights_layer2[0][i]

            self.gb_model.optimize()
            if self.gb_model.status == 3 or self.gb_model.status == 4:
                print('infeasible or unbounded')
                self.scale_down_NN_factor *= 10.0
                self.init_gb_model(critic)
                action = torch.tensor([base_action],dtype=torch.int)
                objval = 0
            else:
                action = torch.tensor([self.a[0,:].X],dtype=torch.int)
                objval = self.gb_model.ObjVal+critic.fc3.bias[0]
                self.gb_model.remove(self.gb_model.getConstrs()[-(len(j3) + 1):])
            # Remove local branching constraints, as they have to be re-generated everytime
        except:
            action = torch.tensor([np.zeros(self.action_dim)])
            objval = 0
            print('some error occured')
            self.init_gb_model(critic)
        return action,objval
