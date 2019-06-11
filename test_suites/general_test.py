# general training

# from rl.core import MultiInputProcessor

from callbacks import *
import dqn_airsim


def test(env_name="AirSimEnv-v42", mode='train', difficulty_level='default'):
	dqn, env = dqn_airsim.setup(env_name, mode, difficulty_level)
	dqn_airsim.test(dqn, env)


if __name__ == '__main__':
	test()
