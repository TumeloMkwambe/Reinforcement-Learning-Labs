###
# Group Members
# Mixo Khoza : 2429356
# Palesa Rapolaki : 2550752
# Tumelo Mkwambe : 2446873
# Karabo Ledwaba : 2569393
###

import numpy as np
from environments.gridworld import GridworldEnv
import timeit
import matplotlib.pyplot as plt
import io

def policy_evaluation(env, policy, discount_factor=1.0, theta=0.00001):
    """
    Evaluate a policy given an environment and a full description of the environment's dynamics.

    Args:

        env: OpenAI environment.
            env.P represents the transition probabilities of the environment.
            env.P[s][a] is a list of transition tuples (prob, next_state, reward, done).
            env.observation_space.n is a number of states in the environment.
            env.action_space.n is a number of actions in the environment.
        policy: [S, A] shaped matrix representing the policy.
        theta: We stop evaluation once our value function change is less than theta for all states.
        discount_factor: Gamma discount factor.

    Returns:
        Vector of length env.observation_space.n representing the value function.
    """

    def calculate_value(state, V):
        """
        Calculates the value given a state and a polict.
        """

        state_value = 0

        for action in env.P[state]:
            action_value = 0
            action_prob = policy[state][action]
            
            for prob, next_state, reward, done in env.P[state][action]:
                action_value += prob * (reward + discount_factor * V[next_state])

            state_value += action_prob * action_value
        
        return state_value

    V = np.zeros(env.observation_space.n)

    while True:
        delta = 0

        for state in range(env.observation_space.n):

            if all(done for action in env.P[state] for _, _, _, done in env.P[state][action]):
                continue

            v = V[state]
            V[state] = calculate_value(state, V)
            delta = max(delta, abs(v - V[state]))

        if delta < theta:
            break

    return V


def policy_iteration(env, policy_evaluation_fn=policy_evaluation, discount_factor=1.0):
    """
    Iteratively evaluates and improves a policy until an optimal policy is found.

    Args:
        env: The OpenAI environment.
        policy_evaluation_fn: Policy Evaluation function that takes 3 arguments:
            env, policy, discount_factor.
        discount_factor: gamma discount factor.

    Returns:
        A tuple (policy, V).
        policy is the optimal policy, a matrix of shape [S, A] where each state s
        contains a valid probability distribution over actions.
        V is the value function for the optimal policy.

    """

    def one_step_lookahead(state, V):
        """
        Helper function to calculate the value for all action in a given state.

        Args:
            state: The state to consider (int)
            V: The value to use as an estimator, Vector of length env.observation_space.n

        Returns:
            A vector of length env.action_space.n containing the expected value of each action.
        """

        A = np.zeros(env.action_space.n)

        for action in range(env.action_space.n):
        
            for prob, next_state, reward, done in env.P[state][action]:
                A[action] += prob * (reward + discount_factor * V[next_state])
        
        return A

    policy = np.ones((env.observation_space.n, env.action_space.n)) / env.action_space.n
    V = policy_evaluation_fn(env, policy, discount_factor, theta = 0.00001)

    while True:
        policy_stable = True

        for state in range(env.observation_space.n):
            old_action = np.argmax(policy[state])
            greedy_action = np.argmax(one_step_lookahead(state, V))
            policy[state] = np.zeros(env.action_space.n)
            policy[state][greedy_action] = 1.0

            if old_action != greedy_action:
                policy_stable = False
        
        if policy_stable:
            break
        
        V = policy_evaluation_fn(env, policy, discount_factor, theta = 0.00001)

    return policy, V



def value_iteration(env, theta=0.0001, discount_factor=1.0):
    """
    Value Iteration Algorithm.

    Args:
        env: OpenAI environment.
            env.P represents the transition probabilities of the environment.
            env.P[s][a] is a list of transition tuples (prob, next_state, reward, done).
            env.observation_space.n is a number of states in the environment.
            env.action_space.n is a number of actions in the environment.
        theta: We stop evaluation once our value function change is less than theta for all states.
        discount_factor: Gamma discount factor.

    Returns:
        A tuple (policy, V) of the optimal policy and the optimal value function.
    """

    def one_step_lookahead(state, V):
        """
        Helper function to calculate the value for all action in a given state.

        Args:
            state: The state to consider (int)
            V: The value to use as an estimator, Vector of length env.observation_space.n

        Returns:
            A vector of length env.action_space.n containing the expected value of each action.
        """

        A = np.zeros(env.action_space.n)

        for action in range(env.action_space.n):
        
            for prob, next_state, reward, done in env.P[state][action]:
                A[action] += prob * (reward + discount_factor * V[next_state])
        
        return A

    V = np.zeros(env.observation_space.n)

    while True:
        delta = 0

        for state in range(env.observation_space.n):

            if all(done for action in env.P[state] for _, _, _, done in env.P[state][action]):
                continue

            v = V[state]
            V[state] = max(one_step_lookahead(state, V))
            delta = max(delta, abs(v - V[state]))

        if delta < theta:
            break
    
    policy = np.zeros((env.observation_space.n, env.action_space.n))

    for state in range(env.observation_space.n):
        action_values = one_step_lookahead(state, V)
        optimal_action = np.argmax(action_values)
        policy[state][optimal_action] = 1.0

    return policy, V


def main():
    # Create Gridworld environment with size of 5 by 5, with the goal at state 24. Reward for getting to goal state is 0, and each step reward is -1
    env = GridworldEnv(shape=[5, 5], terminal_states=[
                       24], terminal_reward=0, step_reward=-1)
    state = env.reset()
    print("")
    env.render()
    print("")

    # TODO: generate random policy
    policy = np.ones((env.observation_space.n, env.action_space.n)) / env.action_space.n

    print("*" * 5 + " Policy evaluation " + "*" * 5)
    print("")

    # TODO: evaluate random policy
    v = policy_evaluation(env, policy, discount_factor = 1.0, theta = 0.00001)

    # TODO: print state value for each state, as grid shape
    grid_values = v.reshape(env.shape)
    print(f'Value For Each State: \n {grid_values} \n')

    # Test: Make sure the evaluated policy is what we expected
    expected_v = np.array([-106.81, -104.81, -101.37, -97.62, -95.07,
                           -104.81, -102.25, -97.69, -92.40, -88.52,
                           -101.37, -97.69, -90.74, -81.78, -74.10,
                           -97.62, -92.40, -81.78, -65.89, -47.99,
                           -95.07, -88.52, -74.10, -47.99, 0.0])
    np.testing.assert_array_almost_equal(v, expected_v, decimal=2)

    print("*" * 5 + " Policy iteration " + "*" * 5)
    print("")
    # TODO: use  policy improvement to compute optimal policy and state values
    policy, v = policy_iteration(env, policy_evaluation_fn = policy_evaluation, discount_factor = 1.0)

    # TODO Print out best action for each state in grid shape
    best_actions = np.argmax(policy, axis=1).reshape(env.shape)
    print(f'Best Action For Each State: \n {best_actions}')

    # TODO: print state value for each state, as grid shape
    grid_values = v.reshape(env.shape)
    print(f'Value For Each State: \n {grid_values} \n')

    # Test: Make sure the value function is what we expected
    expected_v = np.array([-8., -7., -6., -5., -4.,
                           -7., -6., -5., -4., -3.,
                           -6., -5., -4., -3., -2.,
                           -5., -4., -3., -2., -1.,
                           -4., -3., -2., -1., 0.])
    np.testing.assert_array_almost_equal(v, expected_v, decimal=1)

    print("*" * 5 + " Value iteration " + "*" * 5)
    print("")
    # TODO: use  value iteration to compute optimal policy and state values
    policy, v = value_iteration(env, theta = 0.0001, discount_factor = 1.0)

    # TODO Print out best action for each state in grid shape
    best_actions = np.argmax(policy, axis=1).reshape(env.shape)
    print(f'Best Action For Each State: \n {best_actions}')

    # TODO: print state value for each state, as grid shape
    grid_values = v.reshape(env.shape)
    print(f'Value For Each State: \n {grid_values} \n')

    # Test: Make sure the value function is what we expected
    expected_v = np.array([-8., -7., -6., -5., -4.,
                           -7., -6., -5., -4., -3.,
                           -6., -5., -4., -3., -2.,
                           -5., -4., -3., -2., -1.,
                           -4., -3., -2., -1., 0.])
    np.testing.assert_array_almost_equal(v, expected_v, decimal=1)


def exercises():

    env = GridworldEnv(shape=[5, 5], terminal_states=[
                       24], terminal_reward=0, step_reward=-1)
    state = env.reset()

    '''
    Exercise 1.1
    '''

    print("*" * 5 + " Exercise 1.1" + "*" * 5)

    policy = np.ones((env.observation_space.n, env.action_space.n)) / env.action_space.n

    trajectory = []

    while True:

        if all(done for action in env.P[state] for _, _, _, done in env.P[state][action]):
            break

        state_policy = policy[state]
        action = np.random.choice(np.arange(len(state_policy)), p = state_policy)
        trajectory.append((state, action))
        state, reward, done, none = env.step(action)

    trajectory_grid = ['o' for _ in range(env.observation_space.n)]
    trajectory_grid[24] = 'T'

    for entry in trajectory:
        trajectory_grid[entry[0]] = entry[1]

    trajectory_grid[state] = 'X'
    idx_action = {0: 'U', 1: 'R', 2: 'D', 3: 'L'}
    trajectory_grid = np.array([idx_action[entry] if entry in idx_action else entry for entry in trajectory_grid]).reshape(env.shape)

    trajectory = [(state, idx_action[action]) for state, action in trajectory]
    print(f'\n Trajectory (State, Action): {trajectory}')

    output = io.StringIO()
    for row in trajectory_grid:
        output.write("  ".join(row) + "\n")

    print("Trajectory Grid:\n" + output.getvalue())

    '''
    Exercise 4.1
    '''

    discount_rates = np.logspace(-0.2, 0, num=30)

    policy_iteration_times = []
    value_iteration_times = []

    for rate in discount_rates:
        policy_time = timeit.timeit(
            lambda: policy_iteration(env, policy_evaluation, rate),
            number = 10
        )
        policy_iteration_times.append(policy_time / 10)

        value_time = timeit.timeit(
            lambda: value_iteration(env, 0.0001, rate),
            number = 10
        )
        value_iteration_times.append(value_time / 10)

    plt.figure(figsize=(8, 5))
    plt.plot(discount_rates, policy_iteration_times, marker='o', label="Policy Iteration")
    plt.plot(discount_rates, value_iteration_times, marker='s', label="Value Iteration")
    plt.xlabel("Discount Factor (γ)")
    plt.ylabel("Average Time To Completion")
    plt.title("Policy Iteration vs Value Iteration For Different Discount Rates")
    plt.legend()
    plt.grid(True)
    plt.savefig('average_time.png', dpi=300, bbox_inches='tight')

if __name__ == "__main__":
    main()
    exercises()