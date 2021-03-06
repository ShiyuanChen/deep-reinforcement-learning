#!/usr/bin/env python
"""Run Atari Environment with DQN."""
import argparse
import os
import random

import numpy as np
import tensorflow as tf
import gym
from keras.layers import (Activation, Conv2D, Dense, Flatten, Input,
                          Permute)
from keras.models import Model
from keras.models import Sequential
from keras.optimizers import Adam, RMSprop

import deeprl_hw2 as tfrl
from deeprl_hw2.core import ReplayMemory, Preprocessor
from deeprl_hw2.dqn import DQNAgent
from deeprl_hw2.policy import GreedyPolicy, GreedyEpsilonPolicy, LinearDecayGreedyEpsilonPolicy
from deeprl_hw2.preprocessors import HistoryPreprocessor, AtariPreprocessor, PreprocessorSequence
from keras.callbacks import TensorBoard



def create_model(window, input_shape, num_actions,
                 model_name='q_network'):  # noqa: D103
    """Create the Q-network model.

    Use Keras to construct a keras.models.Model instance (you can also
    use the SequentialModel class).

    We highly recommend that you use tf.name_scope as discussed in
    class when creating the model and the layers. This will make it
    far easier to understnad your network architecture if you are
    logging with tensorboard.

    Parameters
    ----------
    window: int
      Each input to the network is a sequence of frames. This value
      defines how many frames are in the sequence.
    input_shape: tuple(int, int)
      The expected input image size.
    num_actions: int
      Number of possible actions. Defined by the gym environment.
    model_name: str
      Useful when debugging. Makes the model show up nicer in tensorboard.

    Returns
    -------
    keras.models.Model
      The Q-model.
    """
    input_dim = input_shape + (window,)
    
    # keras model. Same setting as in the paper
    model = Sequential()
    model.add(Conv2D(32, (8, 8), strides=(4, 4), activation='relu', input_shape=input_dim))
    model.add(Conv2D(64, (4, 4), strides=(2, 2), activation='relu'))
    model.add(Conv2D(64, (3, 3), strides=(1, 1), activation='relu'))
    model.add(Flatten())
    model.add(Dense(512, activation='relu'))
    model.add(Dense(num_actions, activation='linear'))
    print(model.summary())
    return model


def get_output_folder(parent_dir, env_name):
    """Return save folder.

    Assumes folders in the parent_dir have suffix -run{run
    number}. Finds the highest run number and sets the output folder
    to that number + 1. This is just convenient so that if you run the
    same script multiple times tensorboard can plot all of the results
    on the same plots with different names.

    Parameters
    ----------
    parent_dir: str
      Path of the directory containing all experiment runs.

    Returns
    -------
    parent_dir/run_dir
      Path to this run's save directory.
    """
    if not os.path.exists(parent_dir):
        os.makedirs(parent_dir)
    experiment_id = 0
    for folder_name in os.listdir(parent_dir):
        if not os.path.isdir(os.path.join(parent_dir, folder_name)):
            continue
        try:
            folder_name = int(folder_name.split('-run')[-1])
            if folder_name > experiment_id:
                experiment_id = folder_name
        except:
            pass
    experiment_id += 1

    parent_dir = os.path.join(parent_dir, env_name)
    parent_dir = parent_dir + '-run{}'.format(experiment_id)
    return parent_dir


def main():  # noqa: D103
    parser = argparse.ArgumentParser(description='Run DQN on Atari Breakout')
    parser.add_argument('--env', default='SpaceInvadersDeterministic-v0', help='Atari env name')
    parser.add_argument(
        '-o', '--output', default='atari-v0', help='Directory to save data to')
    parser.add_argument('--save', default='output/dqn', help='Directory to save model to')
    parser.add_argument('--seed', default=0, type=int, help='Random seed')
    parser.add_argument('--mode', choices=['train', 'test'], default='train')
    parser.add_argument('--load', default='output/dqn', help='Directory to load model from')

    args = parser.parse_args()

    outputpath = get_output_folder(args.save, args.env)
    print('Output Directory: ' + outputpath)

    # create DQN agent, create model, etc.
    env = gym.make(args.env)
    env.seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)
    num_actions = env.action_space.n
    window = 4
    input_shape = (84, 84)
    model = create_model(window, input_shape, num_actions)

    preprocessor = PreprocessorSequence([HistoryPreprocessor(history_length=4), AtariPreprocessor(new_size=input_shape)])
    memory = ReplayMemory(1000000, window_length=window)
    policy = LinearDecayGreedyEpsilonPolicy(GreedyEpsilonPolicy(epsilon=1.0), attr_name='eps', start_value=1.0, end_value=0.1, num_steps=1000000)

    # keras callback api for tensorboard
    callbacks = [TensorBoard(log_dir=os.path.join(outputpath, 'logs'), histogram_freq=0, write_graph=True, write_images=False)]
    dqn = DQNAgent(model, preprocessor, memory, policy, num_actions=num_actions, gamma=.99, target_update_freq=10000, num_burn_in=50000, 
                    train_freq=4, batch_size=32, output_dir=outputpath)

    # compile the model. loss will be redefined inside the function
    dqn.compile(Adam(lr=.00025), loss_func='mse')

    if args.mode == 'train':
        dqn.fit(env, callbacks=callbacks, num_iterations=5000000)
    else: #evaluate 100 episodes
        dqn.load_weights(args.load)
        dqn.evaluate(env, 100)
    

if __name__ == '__main__':
    main()
