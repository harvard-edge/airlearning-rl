# general training

import numpy as np
import setup
import gym
import time

import gym_airsim.envs
import gym_airsim

import argparse

from keras.models import Sequential, Model
from keras.layers import Dense, Activation, Flatten, Conv2D, Permute, concatenate, Input, Reshape
from keras.utils import plot_model
from keras.optimizers import Adam
import keras.backend as K

from PIL import Image

from rl.agents.dqn import DQNAgent
from rl.policy import LinearAnnealedPolicy, BoltzmannQPolicy, EpsGreedyQPolicy
from rl.memory import SequentialMemory
# from rl.core import MultiInputProcessor
from rl.processors import MultiInputProcessor

from callbacks import *

from keras.callbacks import History


def test():
	parser = argparse.ArgumentParser()
	parser.add_argument('--mode', choices=['train', 'test'], default='train')
	parser.add_argument('--env-name', type=str, default='AirSimEnv-v42')
	parser.add_argument('--weights', type=str, default=None)
	args, unknown = parser.parse_known_args()

	# Get the environment and extract the number of actions.
	env = gym.make(args.env_name)
	env._reset()
