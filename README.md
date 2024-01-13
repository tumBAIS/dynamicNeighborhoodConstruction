# Dynamic Neighborhood Construction for Structured Large Discrete Action Spaces

This project contains code for the paper titled "Dynamic Neighborhood Construction for Structured Large Discrete Action Spaces" by Fabian Akkerman, Julius Luy, Wouter van Heeswijk, and Maximilian Schiffer, see: https://arxiv.org/abs/2305.19891.


## Environment

The code is written in Python 3.8. We use PyTorch 1.13.1 to model neural network architectures and Gurobi 10.0.2 for solving mixed integer problems (MIP). A requirements.txt details further requirements of our project. We tested our project on Ubuntu 20.04, a Windows 11 environment, and a high-performance cluster.

## Folder Structure
The repository contains the following folders:

- **`Src/`**: All source code.
  - **`MappingFunctions/`**: Contains all mapping functions.
  - **`RL_Algorithms/`**: Contains RL implementations.
  - **`Utils/`**: Contains utility functions, such as savings and loading data.
- **`Environments/`**: Contains the studied environments.


On the first level you can see `run.py` which implements the overall policy training and evaluation loop.

### Src 

On the first level you can see a `parser.py`, wherein we set hyperparameters and environment variables, and `config.py`, which preprocesses inputs.

`MappingFunctions`: Contains ``DNC.py``, the main contribution of this project and the benchmarks we compare it against: 
* ``knn.py``: k-nearest neighbor (knn)
* ``ActionRepresentation.py``: Learned Action Representations (LAR)
* MinMax is obtained by setting ``--SA_search_steps`` to ``0`` in the parser file
* DNC w/o SA is obtained by setting ``--mapping=dnc_mapping`` and ``--neighbor_picking=greedy``
* Solving the MIP during training + testing is obtained by setting ``--mapping=dnc_mapping`` and -``-neighborhood_query=math_programming``
* Solving the MIP only during testing and using DNC during training is obtained by setting ``--mapping=dnc_mapping`` and ``--neighborhood_query=hybrid``


`RL_Algorithms`: 
* ``QAC_C2DMapping.py``: Gathers the main RL functionalities required for a Q-actor-critic (QAC) algorithm (e.g., actor and critic updates) and integrate the continuous-to-discrete (C2D) mappings. 
* ``Agent.py``: Groups several high-level RL agent functionalities to allow for further RL pipelines like the one in QAC_C2DMapping.py

`Utils`: 
* ``Actor.py`` and ``Critic.py``: Contain the neural network architectures for actor and critic respectively.
* ``Basis.py``: Contains the state representation module.
* ``Utils.py``: Contains several helper functions such as plotting
* ``MathProg.py``: Contains the implementation of a MIP representing the Q-value network.

### Environments
* `ToyMaze`: Contains the implementation of the maze environment and bases on the work from Chandak et al. (2019). Moreover, it contains the variation of the maze environment with teleporting tiles as introduced in this work.
* ``JobShop``: Contains the implementation of the jobshop scheduling environment.
* `InventoryControl`: Contains the implementation of the joint inventory replenishment problem similar to Vanvuchelen et al. (2022).


## To the make the code work

 * Create a local python environment by subsequently executing the following commands in the root folder
	* `python3 -m venv venv`
	* `source venv/bin/activate`
	* `python -m pip install -r requirements.txt`
	* `deactivate`

 * `Src/parser.py` Set your study's hyperparameters in this file, e.g., which environment to use or setting learning rates
 
 * `run.py` Execute this file using the command line `python3 run.py`.
 
 * Note that you might have to adapt your root folder's name to `dynamicNeighborhoodConstruction`
 
 * Note that the current hyperparameters are set such that a fast dummy training loop is executed on the Maze environment using the DNC mapping  
 
 * Note that knn and LAR rely on the initialization of an action space matrix or embedding vector, which will result in OOM-errors when the action space is too large, depending on the internal memory of the PC

## Acknowledgements
We would like to thank Yash Chandak for sharing his code and for answering our questions considering the learned action representation (LAR) method (Chandak et al. (2019)). Our code for the LAR benchmark meaningfully builds upon and extends his codebase.

## License
* [MIT license](https://opensource.org/license/mit/)
* Copyright 2024 © Fabian Akkerman, Julius Luy, Wouter van Heeswijk, and Maximilian Schiffer


## Bibliography

Yash Chandak, Georgios Theocharous, James Kostas, Scott Jordan, and Philip Thomas. Learning action
representations for reinforcement learning. In International conference on machine learning, pages 941–950.
PMLR, 2019.


Gabriel Dulac-Arnold, Richard Evans, Hado van Hasselt, Peter Sunehag, Timothy Lillicrap, Jonathan Hunt,
Timothy Mann, Theophane Weber, Thomas Degris, and Ben Coppin. Deep reinforcement learning in large
discrete action spaces. arXiv preprint arXiv:1512.07679, 2015.

Nathalie Vanvuchelen, Bram J. De Moor, and Robert N. Boute, The Use of Continuous Action Representations to Scale Deep Reinforcement Learning for Inventory Control. http://dx.doi.org/10.2139/ssrn.4253600 
