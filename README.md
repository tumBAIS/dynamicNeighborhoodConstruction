# Handling Large Discrete Action Spaces via Dynamic Neighborhood Construction

This project contains code for the paper titled "Handling Large Discrete Action Spaces via Dynamic Neighborhood Construction"


## Environment

The code is written in Python 3.8. We use PyTorch 1.13.1 to model neural network architectures. A requirements.txt details further requirements of our project. We tested our project on Ubuntu 20.04, a Windows 11 environment, and a high-performance cluster, whose details you find here: https://doku.lrz.de/linux-cluster-10333236.html.


## Folder Structure
The repository contains the following folders:

Src<br>		|-------Algorithms<br>     	  |-------Utils<br>
Environments <br>	|-------InventoryControl<br>     |-------RecommenderSystem<br>	|-------SystemToyMaze<br>


On the first level you can see run.py which implements the overall policy training and evaluation loop.

### Src 

On the first level you can see a parser.py, wherein we set hyperparameters and environment variables, and config.py, which preprocesses inputs.

`MappingFunctions`: Contains DNC.py, the main contribution of this project and the benchmark we compare it against: 
* knn.py: k-nearest neighbor (knn)
* ActionRepresentation.py: Learned Action Representations (LAR)
* MinMax is obtained by setting k=0 in DNC.py


`RL_Algorithms`: 
* QAC_C2DMapping.py: Gathers the main RL functionalities required for a Q-actor-critic (QAC) algorithm (e.g., actor and critic updates) and integrate mappings. 
* Agent.py: Groups several high-level RL agent functionalities to allow for further RL pipelines like the one in QAC_C2DMapping.py

`Utils`: 
* Policy.py and Critic.py: Contain the neural network architectures for actor and critic respectively.
* Basis.py: Contains the state representation module.
* utils.py: Contains several helper functions such as plotting

### Environments
* `ToyMaze`: Contains the implementation of the maze environment and bases on the work from Chandak et al. (2019).
* `Recommender`: Contains the implementation of the recommender environment similar to Dulac et al. (2015). Here, we also include the movies.csv from https://grouplens.org/datasets/movielens/25m/ and the script (preprocessing.py) we used to generate a tf-idf matrix.
* `JointInventoryReplenishment`: Contains the implementation of the joint inventory replenishment problem similar to Vanvuchelen et al. (2022).


## To the make the code work

 * Create a local python environment by executing subsequently in the root folder
 	* `python3 -m venv venv`
	* `source venv/bin/activate`
	* `python -m pip install -r requirements.txt`
	* `deactivate`
 * `Src/parser.py` Set your study's hyperparameters in this file, e.g., which environment to use or setting learning rates
 
 * `run.py` Execute this file using the command line `python3 run.py`.
 
 * Note that you might have to adapt your root folder's name to `DNC`
 

## Acknowledgements
We would like to thank Yash Chandak for sharing his code and for answering our questions considering the learned action representation (LAR) method (Chandak et al. (2019)). Our code meaningfully builds upon and extends his codebase.

## License
* [MIT license](https://opensource.org/license/mit/)
* Copyright 2023 © Fabian Akkerman, Julius Luy, Wouter van Heeswijk, and Maximilian Schiffer

## Bibliography

Yash Chandak, Georgios Theocharous, James Kostas, Scott Jordan, and Philip Thomas. Learning action
representations for reinforcement learning. In International conference on machine learning, pages 941–950.
PMLR, 2019.


Gabriel Dulac-Arnold, Richard Evans, Hado van Hasselt, Peter Sunehag, Timothy Lillicrap, Jonathan Hunt,
Timothy Mann, Theophane Weber, Thomas Degris, and Ben Coppin. Deep reinforcement learning in large
discrete action spaces. arXiv preprint arXiv:1512.07679, 2015.

Nathalie Vanvuchelen, Bram J. De Moor, and Robert N. Boute, The Use of Continuous Action Representations to Scale Deep Reinforcement Learning for Inventory Control. http://dx.doi.org/10.2139/ssrn.4253600 
# dynamicNeighborhoodConstruction
