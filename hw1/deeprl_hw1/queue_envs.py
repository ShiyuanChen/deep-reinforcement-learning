# coding: utf-8
"""Define the Queue environment from problem 3 here."""

from __future__ import (absolute_import, division, print_function,
                        unicode_literals)

from gym import Env, spaces
from gym.envs.registration import register
import numpy as np


class QueueEnv(Env):
    """Implement the Queue environment from problem 3.

    Parameters
    ----------
    p1: float
      Value between [0, 1]. The probability of queue 1 receiving a new item.
    p2: float
      Value between [0, 1]. The probability of queue 2 receiving a new item.
    p3: float
      Value between [0, 1]. The probability of queue 3 receiving a new item.

    Attributes
    ----------
    nS: number of states
    nA: number of actions
    P: environment model
    """
    metadata = {'render.modes': ['human']}

    SWITCH_TO_1 = 0
    SWITCH_TO_2 = 1
    SWITCH_TO_3 = 2
    SERVICE_QUEUE = 3

    def __init__(self, p1, p2, p3):
        self.action_space = spaces.Discrete(4)
        self.observation_space = spaces.MultiDiscrete(
            [(1, 3), (0, 5), (0, 5), (0, 5)])
        S = [(pos, q1, q2, q3) for pos in range(3) for q1 in range(6) for q2 in range(6) for q3 in range(6)]
        self.nS = len(S)
        self.nA = 4
        self.P = {}

        for s in S:
            self.P[s] = {}
            for a in range(self.nA):
                pos, q1, q2, q3 = s
                queue = [q1, q2, q3]

                r = 0
                if a < 3:
                    pos = a
                elif a == 3 and queue[pos] > 0:
                    queue[pos] -= 1
                    r += 1

                q1, q2, q3 = queue[0], queue[1], queue[2]
                p1_new = 0 if q1 == 5 else p1
                p2_new = 0 if q2 == 5 else p2
                p3_new = 0 if q3 == 5 else p3

                isAddition = [(a1, a2, a3) for a1 in range(2) for a2 in range(2) for a3 in range(2)]

                transition = []
                for (a1, a2, a3) in isAddition:
                    q1_prime = q1 + 1 if a1 else q1
                    q2_prime = q2 + 1 if a2 else q2
                    q3_prime = q3 + 1 if a3 else q3
                    if q1_prime > 5 or q2_prime > 5 or q3_prime > 5:
                        continue
                    
                    p = (p1_new if a1 else (1 - p1_new)) * (p2_new if a2 else (1 - p2_new)) * (p3_new if a3 else (1 - p3_new))
                    
                    transition.append((p, (pos, q1_prime, q2_prime, q3_prime), r, False))
                self.P[s][a] = transition


    def _reset(self):
        """Reset the environment.

        The server should always start on Queue 1.

        Returns
        -------
        (int, int, int, int)
          A tuple representing the current state with meanings
          (current queue, num items in 1, num items in 2, num items in
          3).
        """
        self.s = (0,0,0,0)
        return self.s

    def _step(self, action):
        """Execute the specified action.

        Parameters
        ----------
        action: int
          A number in range [0, 3]. Represents the action.

        Returns
        -------
        (state, reward, is_terminal, debug_info)
          State is the tuple in the same format as the reset
          method. Reward is a floating point number. is_terminal is a
          boolean representing if the new state is a terminal
          state. debug_info is a dictionary. You can fill debug_info
          with any additional information you deem useful.
        """
        debug_info = {}
        cum_sum = np.random.rand()
        for (p, s_prime, r, is_terminal) in self.P[self.s][action]:
            cum_sum -= p
            if(cum_sum < 0):
                self.s = s_prime
                return s_prime, r, is_terminal, debug_info

        return None, None, None, None

    def _render(self, mode='human', close=False):
        pass

    def _seed(self, seed=None):
        """Set the random seed.

        Parameters
        ----------
        seed: int, None
          Random seed used by numpy.random and random.
        """
        np.random.seed(seed)

    def query_model(self, state, action):
        """Return the possible transition outcomes for a state-action pair.

        This should be in the same format at the provided environments
        in section 2.

        Parameters
        ----------
        state
          State used in query. Should be in the same format at
          the states returned by reset and step.
        action: int
          The action used in query.

        Returns
        -------
        [(prob, nextstate, reward, is_terminal), ...]
          List of possible outcomes
        """
        return self.P[state][action]

    def get_action_name(self, action):
        if action == QueueEnv.SERVICE_QUEUE:
            return 'SERVICE_QUEUE'
        elif action == QueueEnv.SWITCH_TO_1:
            return 'SWITCH_TO_1'
        elif action == QueueEnv.SWITCH_TO_2:
            return 'SWITCH_TO_2'
        elif action == QueueEnv.SWITCH_TO_3:
            return 'SWITCH_TO_3'
        return 'UNKNOWN'


register(
    id='Queue-1-v0',
    entry_point='deeprl_hw1.queue_envs:QueueEnv',
    kwargs={'p1': .1,
            'p2': .9,
            'p3': .1})

register(
    id='Queue-2-v0',
    entry_point='deeprl_hw1.queue_envs:QueueEnv',
    kwargs={'p1': .1,
            'p2': .1,
            'p3': .1})
