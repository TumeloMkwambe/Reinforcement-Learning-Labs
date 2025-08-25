import random
import numpy as np

'''
Note***: 
    run-agnostic: updates across runs, retains information relevant for all runs
    run-dependent: re-initialized at (cleared before) each run, retains information related to run
    algorithm-specific: depends on (changes by) the algorithm
'''

class Algorithm:
    def __init__(self, name, num_actions):
        '''
        Args:
            name (string): name of the algorithm (action-value method)
            num_actions (int): number of actions/arms
        
        Attributes:
            _estimated_values (np.array): estimated action values for all actions, initialized to 0
            _action_counts (np.array): number of times actions have been selected
        '''

        self.name = name # algorithm-specific
        self._num_actions = num_actions # run-agnostic
        self._estimated_values = np.zeros(num_actions) # run-dependent & algorithm-specific
        self._action_counts = np.zeros(num_actions) # run-dependent & algorithm-specific

    def update(self, action, reward, time_step):
        '''
        Args:
            action (int): action whose value we intend to update
            reward (float): reward obtained after taking action
            time_step (int): time step for action and reward
        
        Objective: performs incremental sample average to estimate selected action's action-value
        '''
        self._action_counts[action] += 1
        self._estimated_values[action] += 1 / self._action_counts[action] * (reward - self._estimated_values[action])


class Greedy(Algorithm):
    def __init__(self, name, num_actions, q_1=5):
        '''
        Args:
            name (string): name of the algorithm (action-value method)
            num_actions (int): number of actions/arms
            q_1 (int): optimistic initial value

        Attributes:
            __estimated_values (np.array): estimated action values for all actions, initialized to q_1,
                                           overwrites super class __estimated_values array
        '''

        super().__init__(name, num_actions)
        self._estimated_values = np.full((num_actions,), q_1) # run-dependent & algorithm-specific

    def select_action(self, t):
        '''
        Objective: selects action with the current best action-value and updates number of times action has been selected
        '''

        action = self._estimated_values.argmax()
        return action


class Epsilon(Algorithm):
    def __init__(self, name, num_actions, epsilon=0.1):
        '''
        Args:
            name (string): name of the algorithm (action-value method)
            num_actions (int): number of actions/arms
        
        Attributes:
            __epsilon (int): probability to which we select a random action over a greedy action
        '''

        super().__init__(name, num_actions)
        self.__epsilon = epsilon # run-dependent

    def select_action(self, t):
        '''
            Objective: selects action action with the current best action-value with probability 1 - epsilon and updates number of times action has been selected
        '''

        action = self._estimated_values.argmax() if random.random() > self.__epsilon else random.randint(0, self._num_actions - 1)
        return action


class UCB(Algorithm):
    def __init__(self, name, num_actions, c=2):
        '''
        Args:
            name (string): name of the algorithm (action-value method)
            num_actions (int): number of actions/arms
            c (int): exploration hyperparameter to control how much UCB explores
        '''

        super().__init__(name, num_actions)
        self.__c = c

    def select_action(self, t):
        '''
        Args:
            t (int): time step

        Objective: selects action with the best action-value estimate by UCB standard and updates number of times action has been selected
        '''

        ucb_values = self._estimated_values + self.__c * np.sqrt(np.log(t) / self._action_counts + 1e-5)
        action = np.argmax(ucb_values)
        return action
