import random
import numpy as np

'''
Note***: 
    run-agnostic: updates across runs, retains information relevant for all runs
    run-dependent: re-initialized at (cleared before) each run, retains information related to run
'''

class Run:
    def __init__(self, num_actions, duration):
        '''
        Args:
            num_actions (int): number of actions/arms
            duration (int): time steps for a single run


        Attributes: 
            __num_actions (int): number of actions/arms
            greedy_rewards (np.array): array to record rewards from greedy actions for each time step
            epsilon_rewards (np.array): array to record rewards from epsilon-greedy actions for each time step
            ucb_rewards (np.array): array to record rewards from ucb actions for each time step
            __action_values (np.array): actual action values for a particular run
        '''

        self.__num_actions = num_actions # run-agnostic, kept constant for all runs
        self.greedy_rewards = np.zeros(duration) # run-agnostic, collect greedy action rewards for all runs for each time step (to average post-runs)
        self.epsilon_rewards = np.zeros(duration)  # run-agnostic, collect epsilon-greedy action rewards for all runs for each time step (to average post-runs)
        self.ucb_rewards = np.zeros(duration) # run-agnostic, collect ucb action rewards for all runs for each time step (to average post-runs)
        self.__action_values = None # run-dependent, initializes (samples) new true action-values at each run
    
    def initial_values(self):
        '''
        Objective: samples true action-values from a gaussian distribution
        '''

        self.__action_values = np.random.normal(0, np.sqrt(3), self.__num_actions)
    
    def update_rewards(self, algorithm, action, time_step):
        '''
        Args:
            algorithm (string): name of the algorithm which selected action which resulted in the reward
            action (int): action responsible for the reward
            time_step (int): time step at which action was selected and reward was received
        
        Objective: adds reward to action selection method rewards array for specified timestep
        '''

        action_value = self.__action_values[action]
        reward = np.random.normal(action_value, np.sqrt(1))
        if(algorithm == "greedy"):
            self.greedy_rewards[time_step] += reward
        if(algorithm == "epsilon"):
            self.epsilon_rewards[time_step] += reward
        if(algorithm == "ucb"):
            self.ucb_rewards[time_step] += reward
        return reward
    
    def average_rewards(self, num_runs):
        '''
        Args:
            num_runs (int): number of runs to average rewards over for all timesteps t
        '''
        
        self.greedy_rewards = self.greedy_rewards / num_runs
        self.epsilon_rewards = self.epsilon_rewards / num_runs
        self.ucb_rewards = self.ucb_rewards / num_runs
    
    def reset_rewards(self, duration):
        self.greedy_rewards = np.zeros(duration)
        self.epsilon_rewards = np.zeros(duration)
        self.epsilon_rewards = np.zeros(duration)