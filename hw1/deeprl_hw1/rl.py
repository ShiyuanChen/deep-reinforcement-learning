# coding: utf-8
from __future__ import division, absolute_import
from __future__ import print_function, unicode_literals

import numpy as np


def evaluate_policy(env, gamma, policy, max_iterations=int(1e3), tol=1e-3):
    """Evaluate the value of a policy.

    See page 87 (pg 105 pdf) of the Sutton and Barto Second Edition
    book.

    http://webdocs.cs.ualberta.ca/~sutton/book/bookdraft2016sep.pdf

    Parameters
    ----------
    env: gym.core.Environment
      The environment to compute value iteration for. Must have nS,
      nA, and P as attributes.
    gamma: float
      Discount factor, must be in range [0, 1)
    policy: np.array
      The policy to evaluate. Maps states to actions.
    max_iterations: int
      The maximum number of iterations to run before stopping.
    tol: float
      Determines when value function has converged.

    Returns
    -------
    np.ndarray, int
      The value for the given policy and the number of iterations till
      the value function converged.
    """
    V = np.zeros(env.nS)
    
    for i in range(max_iterations):
        delta = 0
        for s in range(env.nS):
            v = V[s]
            new_Vs = 0
            for p, s_prime, r, is_terminal in env.P[s][policy[s]]:
                if is_terminal: 
                    new_Vs += p * r
                else:
                    new_Vs += p * (r + gamma * V[s_prime])

            V[s] = new_Vs
            delta = max(delta, abs(v - V[s]))
            
        if delta < tol:
            return V, i
    
    return V, max_iterations


def value_function_to_policy(env, gamma, value_function):
    """Output action numbers for each state in value_function.

    Parameters
    ----------
    env: gym.core.Environment
      Environment to compute policy for. Must have nS, nA, and P as
      attributes.
    gamma: float
      Discount factor. Number in range [0, 1)
    value_function: np.ndarray
      Value of each state.

    Returns
    -------
    np.ndarray
      An array of integers. Each integer is the optimal action to take
      in that state according to the environment dynamics and the
      given value function.
    """    
    policy = np.zeros(env.nS)
    for s in range(env.nS):
        max_r = -np.inf
        for a in range(env.nA):
            expected_r = 0
            for p, s_prime, r, is_terminal in env.P[s][a]:
                expected_r += p * (r + gamma * value_function[s_prime])
            if expected_r > max_r:
                max_r = expected_r
                policy[s] = a

    return policy


def improve_policy(env, gamma, value_func, policy):
    """Given a policy and value function improve the policy.

    See page 87 (pg 105 pdf) of the Sutton and Barto Second Edition
    book.

    http://webdocs.cs.ualberta.ca/~sutton/book/bookdraft2016sep.pdf

        Parameters
    ----------
    env: gym.core.Environment
      The environment to compute value iteration for. Must have nS,
      nA, and P as attributes.
    gamma: float
      Discount factor, must be in range [0, 1)
    value_func: np.ndarray
      Value function for the given policy.
    policy: dict or np.array
      The policy to improve. Maps states to actions.

    Returns
    -------
    bool, np.ndarray
      Returns true if policy changed. Also returns the new policy.
    """
    policy_stable = True
    for s in range(env.nS):
        old_action = policy[s]
        max_r = -np.inf
        for a in range(env.nA):
            expected_r = 0
            for p, s_prime, r, is_terminal in env.P[s][a]:
                expected_r += p * (r + gamma * value_func[s_prime])
            if expected_r > max_r:
                max_r = expected_r
                policy[s] = a

        if old_action != policy[s]:
            policy_stable = False
            
    return policy_stable, policy


def policy_iteration(env, gamma, max_iterations=int(1e3), tol=1e-3):
    """Runs policy iteration.

    See page 87 (pg 105 pdf) of the Sutton and Barto Second Edition
    book.

    http://webdocs.cs.ualberta.ca/~sutton/book/bookdraft2016sep.pdf

    You should use the improve_policy and evaluate_policy methods to
    implement this method.

    Parameters
    ----------
    env: gym.core.Environment
      The environment to compute value iteration for. Must have nS,
      nA, and P as attributes.
    gamma: float
      Discount factor, must be in range [0, 1)
    max_iterations: int
      The maximum number of iterations to run before stopping.
    tol: float
      Determines when value function has converged.

    Returns
    -------
    (np.ndarray, np.ndarray, int, int)
       Returns optimal policy, value function, number of policy
       improvement iterations, and number of value iterations.
    """
    policy = np.zeros(env.nS, dtype='int')
    value_func = np.zeros(env.nS)
    policy_stable = False
    num_val_iters = 0
    num_policy_iters = 0

    while not policy_stable:
        value_func, num_iters = evaluate_policy(env, gamma, policy, max_iterations, tol)
        policy_stable, policy = improve_policy(env, gamma, value_func, policy)
        num_val_iters += num_iters
        num_policy_iters += 1

    return policy, value_func, num_policy_iters, num_val_iters


def value_iteration(env, gamma, max_iterations=int(1e3), tol=1e-3):
    """Runs value iteration for a given gamma and environment.

    See page 90 (pg 108 pdf) of the Sutton and Barto Second Edition
    book.

    http://webdocs.cs.ualberta.ca/~sutton/book/bookdraft2016sep.pdf

    Parameters
    ----------
    env: gym.core.Environment
      The environment to compute value iteration for. Must have nS,
      nA, and P as attributes.
    gamma: float
      Discount factor, must be in range [0, 1)
    max_iterations: int
      The maximum number of iterations to run before stopping.
    tol: float
      Determines when value function has converged.

    Returns
    -------
    np.ndarray, iteration
      The value function and the number of iterations it took to converge.
    """
    V = np.zeros(env.nS)
    for i in range(max_iterations):
        delta = 0
        for s in range(env.nS):
            v = V[s]
            max_r = -np.inf
            for a in range(env.nA):
                expected_r = 0
                for p, s_prime, r, is_terminal in env.P[s][a]:
                    if is_terminal:
                        expected_r += p * r
                    else:
                        expected_r += p * (r + gamma * V[s_prime])
                max_r = max(max_r, expected_r)

            V[s] = max_r
            delta  = max(delta, abs(v - V[s]))

        if delta < tol:
            return V, i

    return V, max_iterations


def print_policy(policy, action_names):
    """Print the policy in human-readable format.

    Parameters
    ----------
    policy: np.ndarray
      Array of state to action number mappings
    action_names: dict
      Mapping of action numbers to characters representing the action.
    """
    str_policy = policy.astype('str')
    for action_num, action_name in action_names.items():
        np.place(str_policy, policy == action_num, action_name)

    print(str_policy)
