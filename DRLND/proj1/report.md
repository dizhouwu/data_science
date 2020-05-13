# This page is for the first project in Udacity DRLND.
Learning Algorithm:
A DQN model is used (for the sake of less training time, a very simple NN is used with only 2 hidden layers.

Hyperparameters:
--Agent--

* BUFFER_SIZE = int(1e5)  # replay buffer size
* BATCH_SIZE = 64         # minibatch size
* GAMMA = 0.99            # discount factor
* TAU = 1e-3              # for soft update of target parameters
* LR = 5e-4               # learning rate 
* UPDATE_EVERY = 4        # how often to update the network

--Model--

* HIDDEN_UNITS = 64       # number of hidden units in the fc layer









Future work:
Improve DQN with better models using an arsenal of Double DQN, Prioritized Experience Replay, Dueling DQN, Distributional DQN, Noisy DQN, Rainbow.
