import numpy as np
import setup
import gym
import time

import gym_airsim.envs
import gym_airsim

import argparse


def test():
	parser = argparse.ArgumentParser()
	parser.add_argument('--mode', choices=['train', 'test'], default='train')
	parser.add_argument('--env-name', type=str, default='AirSimEnv-v42')
	parser.add_argument('--weights', type=str, default=None)
	args, unknown = parser.parse_known_args()

	# Get the environment and extract the number of actions.
	env = gym.make(args.env_name)

	# Test for setting an item
	item_val = 10
	env.updateJson(("NumberOfObjects", item_val))
	assert (item_val == env.getItemCurGameConfig("NumberOfObjects")), "item val should be equal"

	# Test for passing the wrong number of inputs
	try:
		print(env.setRangeGameConfig("NumbeOfObjects", 1))
	except AssertionError:
		print("pass")
	# Test for setting a range
	range_val = list(range(10, 50))
	env.setRangeGameConfig(("NumberOfObjects", range_val))
	assert (range_val == env.getRangeGameConfig("NumberOfObjects")), "range_val should be equal"

	# Testing random sampling
	for i in range(0, 5):
		env.sampleGameConfig("NumberOfObjects")
		print(env.getItemCurGameConfig("NumberOfObjects"))

	# Test for setting a range
	range_val = [[20, 20, 3], [100, 100, 5]]
	env.setRangeGameConfig(("ArenaSize", range_val))
	assert (range_val == env.getRangeGameConfig("ArenaSize")), "range_val should be equal"

	for i in range(0, 5):
		env.sampleGameConfig("ArenaSize")
		print(env.getItemCurGameConfig("ArenaSize"))

	print("------------- Testing End now")
	for i in range(0, 10):
		env.sampleGameConfig("LevelDifficulty")
		print(env.getItemCurGameConfig("LevelDifficulty"))
