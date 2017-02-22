import gym

import deeprl_hw1.lake_envs as lake_env
import deeprl_hw1.rl as rl
import numpy as np
import time
import matplotlib.pyplot as plt


def plotColorbar(value_func, output=None):
    rows = int(np.sqrt(len(value_func)))    
    fig = plt.figure()

    plt.pcolor(np.reshape(value_func, (rows, rows)))
    plt.gca().invert_yaxis()
    plt.colorbar()
    plt.show()
    fig.savefig(output)


def main():
    # envs = ['Deterministic-4x4-FrozenLake-v0','Deterministic-8x8-FrozenLake-v0']
    # envs = ['Stochastic-4x4-FrozenLake-v0','Stochastic-8x8-FrozenLake-v0']
    envs = ['Deterministic-4x4-neg-reward-FrozenLake-v0']
    action_names = {0:'L', 1:'D', 2:'R', 3:'U'}

    for environment in envs:
        # create the environment
        env = gym.make(environment)
        print environment
        print "Policy Iteration"
        t0 = time.clock()
        policy, value_func, num_policy_iters, num_val_iters = rl.policy_iteration(env, 0.9)
        print "Execution Time:", time.clock() - t0
        print "Number of Policy Improvement Steps:", num_policy_iters
        print "Number of Policy Evaluation Steps:", num_val_iters
        rl.print_policy(policy, action_names)
        plotColorbar(value_func, environment + "_policy")

        print
        print "Value Iteration"
        t0 = time.clock()
        value_func, num_val_iters = rl.value_iteration(env, 0.9)
        value_policy = rl.value_function_to_policy(env, 0.9, value_func)
        print "Execution Time:", time.clock() - t0
        print "Number of Value Evaluation Steps:", num_val_iters
        rl.print_policy(value_policy, action_names)
        plotColorbar(value_func, environment + "_value")


if __name__ == '__main__':
    main()