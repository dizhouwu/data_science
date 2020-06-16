Enrivonment Description: <br />
In this environment, two agents control rackets to bounce a ball over a net. If an agent hits the ball over the net, it receives a reward of +0.1. If an agent lets a ball hit the ground or hits the ball out of bounds, it receives a reward of -0.01. Thus, the goal of each agent is to keep the ball in play.

The observation space consists of 8 variables corresponding to the position and velocity of the ball and racket. Each agent receives its own, local observation. Two continuous actions are available, corresponding to movement toward (or away from) the net, and jumping.

The task is episodic, and in order to solve the environment, your agents must get an average score of +0.5 (over 100 consecutive episodes, after taking the maximum over both agents). Specifically,

After each episode, we add up the rewards that each agent received (without discounting), to get a score for each agent. This yields 2 (potentially different) scores. We then take the maximum of these 2 scores.
This yields a single score for each episode.
The environment is considered solved, when the average (over 100 episodes) of those scores is at least +0.5.

Algorithm: Deep Deterministic Policy Gradient (DDPG)
Deep Deterministic Policy Gradient (DDPG) is an algorithm which concurrently learns a Q-function and a policy. It uses off-policy data and the Bellman equation to learn the Q-function, and uses the Q-function to learn the policy.


Model(ddpg_model.py)
The Actor-Critic architecture is applied. <br />
HYPERPARAMS:<br />
FC1_UNITS = 256<br />
FC2_UNITS =128<br />
ACTIVATION:<br />
leaky_relu<br />

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
n_episodes=3000, max_t=1000<br />
CHECK_EVERY: 50 Checkpoint every number of episodes<br />
scores_deque: deque(maxlen=100)


Results:






![alt text](results.jpg)<br />


Ideas for Future Work<br />
Change neural net architecture.<br />
Use algorithm other than DDPG.
