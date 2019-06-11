import gym
import os, sys
from gym_airsim.envs.airlearningclient import *
import numpy as np
from keras import backend as K
import msgs
import settings

from ddpg import *

from ReplayBuffer import ReplayBuffer
from ActorNetwork import ActorNetwork
from CriticNetwork import CriticNetwork
from OU import OU

OU = OU()


def setup(mode='train', difficulty_level='default', env_name="AirSimEnv-v42"):
	env = gym.make(env_name)
	env.init_again(eval("settings." + difficulty_level + "_range_dic"))
	env.airgym.unreal_reset()  # must rest so the env accomodate the changes
	time.sleep(5)

	np.random.seed(123)
	env.seed(123)

	BUFFER_SIZE = settings.buffer_size
	BATCH_SIZE = settings.batch_size  # ToDo: Determine what this value is
	GAMMA = settings.gamma
	TAU = settings.tau  # Target Network HyperParameters
	LRA = settings.lra  # Learning rate for Actor
	LRC = settings.lrc  # Lerning rate for Critic

	# Drone controls
	MIN_THROTTLE = settings.min_throttle
	DURATION = settings.duration

	EXPLORE = settings.explore
	episode_count = settings.episode_count_cap
	max_steps = settings.nb_max_episodes_steps + 2
	total_steps = settings.training_steps_cap
	reward = 0
	done = False
	step = 0
	epsilon = 1
	# TensorFlow GPU optimization
	config = tf.ConfigProto()
	config.gpu_options.allow_growth = True
	sess = tf.Session(config=config)
	K.set_session(sess)

	depth_front_kdim = (1,) + env.state()[0].shape
	grey_front_kdim = (1,) + env.state()[1].shape
	depth_bottom_kdim = (1,) + env.state()[2].shape
	grey_bottom_kdim = (1,) + env.state()[3].shape
	depth_back_kdim = (1,) + env.state()[4].shape
	grey_back_kdim = (1,) + env.state()[5].shape
	vel_kdim = (1,) + env.state()[6].shape
	pos_kdim = (1,) + env.state()[7].shape

	action_dim = env.action_space.shape[0]  # pitch, roll, throttle, yaw_rate and duration
	# create actor network:
	actor = ActorNetwork(sess, depth_front_kdim, grey_front_kdim, depth_bottom_kdim, grey_bottom_kdim, depth_back_kdim,
	                     grey_back_kdim, vel_kdim, pos_kdim, action_dim, BATCH_SIZE, TAU,
	                     LRA)  # TODO: Figure out how to modify this

	critic = CriticNetwork(sess, depth_front_kdim, grey_front_kdim, depth_bottom_kdim, grey_bottom_kdim,
	                       depth_back_kdim, grey_back_kdim, vel_kdim, pos_kdim, action_dim, BATCH_SIZE, TAU,
	                       LRC)  # TODO: Figure out how to modify this

	buff = ReplayBuffer(BUFFER_SIZE)  # Create replay buffer

	agent = DDPGAgent(GAMMA, BATCH_SIZE, TAU, LRA, LRC, total_steps, max_steps, episode_count,
	                  MIN_THROTTLE, DURATION, env, actor, critic, buff)

	env.set_actor_critic(actor.model, critic.model)
	return agent, env


def train(agent, env):
	msgs.mode = 'train'
	agent.train()


def test(agent, env, file_path):
	msgs.mode = 'test'
	msgs.weight_file_under_test = file_path["actor"]
	agent.test('', file_path)


if __name__ == '__main__':
	train()
