Model(ddpg_model.py)
The Actor-Critic architecture is applied. 
HYPERPARAMS:
FC1_UNITS = 256
FC2_UNITS =128
ACTIVATION:
leaky_relu

Agent: (maddpg_agent.py)

HYPERPARAMS:
BUFFER_SIZE = int(1e6)  # replay buffer size<br />
BATCH_SIZE = 1024        # minibatch size<br />
GAMMA = 0.99            # discount factor<br />
TAU = 1e-3              # for soft update of target parameters<br />
LR_ACTOR = 1e-4         # learning rate of the actor<br />
LR_CRITIC = 1e-3        # learning rate of the critic<br />
WEIGHT_DECAY = 0        # L2 weight decay<br />
NUM_AGENTS = 2         # Number of agents in the environment<br />
NUM_STEPS_TO_UPDATE = 10        # Number of times to update the networks at a given time step<br />

main function:
HYPERPARAMS:
n_episodes=3000, max_t=1000
CHECK_EVERY: Checkpoint every number of episodes


Results:






![alt text](results.jpg)
