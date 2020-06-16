Model(ddpg_model.py)
The Actor-Critic architecture is applied. 
HYPERPARAMS:
FC1_UNITS = 256
FC2_UNITS =128
ACTIVATION:
leaky_relu

Agent: (maddpg_agent.py)

HYPERPARAMS:
BUFFER_SIZE = int(1e6)  # replay buffer size
BATCH_SIZE = 1024        # minibatch size
GAMMA = 0.99            # discount factor
TAU = 1e-3              # for soft update of target parameters
LR_ACTOR = 1e-4         # learning rate of the actor
LR_CRITIC = 1e-3        # learning rate of the critic
WEIGHT_DECAY = 0        # L2 weight decay
NUM_AGENTS = 2         # Number of agents in the environment
NUM_STEPS_TO_UPDATE = 10        # Number of times to update the networks at a given time step

main function:
HYPERPARAMS:
n_episodes=3000, max_t=1000
CHECK_EVERY: Checkpoint every number of episodes


Results:






![alt text](results.jpg)
