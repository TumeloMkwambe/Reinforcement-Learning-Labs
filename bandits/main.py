import numpy as np
from run import Run
import matplotlib.pyplot as plt
from algorithm import Greedy, Epsilon, UCB

def Runs(Run_, num_runs, num_actions, num_timesteps, q_1, epsilon, c):
    '''
    Args:
        Run_ (object): instance of Run class
        num_runs (int): number of runs
        num_action (int): number of arms/actions
        num_timesteps (int): duration of a single run
        q_1 (int): [hyperparameter] optimistic initial value
        epsilon (float): [hyperparameter] probability to which we select a random action over a greedy action
        c (int): [hyperparameter] exploration hyperparameter to control how much UCB explores
    '''

    for run in range(num_runs):
        Run_.initial_values()
        Greedy_ = Greedy("greedy", num_actions, q_1)
        Epsilon_ = Epsilon("epsilon", num_actions, epsilon)
        UCB_ = UCB("ucb", num_actions, c)

        for algorithm in [Greedy_, Epsilon_, UCB_]:
            for t in range(num_timesteps):
                action = algorithm.select_action(t)
                reward = Run_.update_rewards(algorithm.name, action, t)
                algorithm.update(action, reward, t)
    Run_.average_rewards(num_runs)


### * * * * * * * * EXPERIMENT 1 * * * * * * * * ### 

num_actions = 10
num_timesteps = 1000
num_runs = 100

q_1 = 5
epsilon = 0.1
c = 2

Run_ = Run(num_actions, num_timesteps)
Runs(Run_, num_runs, num_actions, num_timesteps, q_1, epsilon, c)

timesteps = np.arange(num_timesteps)

plt.figure(figsize=(10, 5))
plt.plot(timesteps, Run_.greedy_rewards, label='Greedy')
plt.plot(timesteps, Run_.epsilon_rewards, label='ϵ-Greedy')
plt.plot(timesteps, Run_.ucb_rewards, label='UCB')
plt.xlabel('Time Step')
plt.ylabel('Average Reward')
plt.title('Average Reward Over Time For Different Action Selection Methods')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()


### * * * * * * * * EXPERIMENT 2 * * * * * * * * ###
'''
num_actions = 10
num_timesteps = 1000
num_runs = 1000
timesteps = np.arange(num_timesteps)

q_1_hyperparameters = np.array([0, 1, 5, 10])
epsilon_hyperparameters = np.array([0.01, 0.1, 0.25, 0.5])
c_hyperparameters = np.array([0.5, 1, 2, 5])

greedy_rewards = []
epsilon_rewards = []
ucb_rewards = []

Run_ = Run(num_actions, num_timesteps)
for i in range(len(q_1_hyperparameters)):
    Run_.reset_rewards(num_timesteps)
    Runs(
        Run_, 
        num_runs, 
        num_actions, 
        num_timesteps, 
        q_1_hyperparameters[i], 
        epsilon_hyperparameters[i], 
        c_hyperparameters[i])
    greedy_rewards.append(Run_.greedy_rewards.copy())
    epsilon_rewards.append(Run_.epsilon_rewards.copy())
    ucb_rewards.append(Run_.ucb_rewards.copy())

fig, axs = plt.subplots(1, 3, figsize=(18, 5), sharey=True)

# Greedy Hyperparameters Plot
for rewards, q1 in zip(greedy_rewards, q_1_hyperparameters):
    axs[0].plot(timesteps, rewards, label=f'Q₁={q1}')
axs[0].set_title('Greedy (Initial Value Exploration)')
axs[0].set_xlabel('Time Step')
axs[0].set_ylabel('Average Reward')
axs[0].legend()
axs[0].grid(True)

# Epsilon-Greedy Hyperparameters Plot
for rewards, eps in zip(epsilon_rewards, epsilon_hyperparameters):
    axs[1].plot(timesteps, rewards, label=f'ϵ={eps}')
axs[1].set_title('ϵ-Greedy')
axs[1].set_xlabel('Time Step')
axs[1].legend()
axs[1].grid(True)

# UCB Hyperparameters Plot
for rewards, c in zip(ucb_rewards, c_hyperparameters):
    axs[2].plot(timesteps, rewards, label=f'c={c}')
axs[2].set_title('Upper-Confidence-Bound')
axs[2].set_xlabel('Time Step')
axs[2].legend()
axs[2].grid(True)

fig.suptitle('Experiment 2: Average Reward over Time for Varying Hyperparameters', fontsize=16)
plt.tight_layout()
plt.show()
'''