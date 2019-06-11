import sys
import gym

import os
import tensorflow as tf
os.sys.path.insert(0, os.path.abspath('../../../settings_folder'))
import settings
import msgs
from gym_airsim.envs.airlearningclient import *
import callbacks
from multi_modal_policy import MultiInputPolicy
from stable_baselines.common.policies import MlpPolicy
from stable_baselines.common.policies import CnnPolicy
from stable_baselines.common.vec_env import DummyVecEnv
from stable_baselines import PPO2
from keras.backend.tensorflow_backend import set_session

def setup(difficulty_level='default', env_name = "AirSimEnv-v42"):
    config = tf.ConfigProto()
    config.gpu_options.per_process_gpu_memory_fraction = 0.6
    config.gpu_options.allow_growth = True
    set_session(tf.Session(config=config))

    env = gym.make(env_name)
    env.init_again(eval("settings."+difficulty_level+"_range_dic"))

    # Vectorized environments allow to easily multiprocess training
    # we demonstrate its usefulness in the next examples
    vec_env = DummyVecEnv([lambda: env])  # The algorithms require a vectorized environment to run
    agent = PPO2(MultiInputPolicy, vec_env, verbose=1)
    env.set_model(agent)

    return env, agent

def train(env, agent):


    # Train the agent
    agent.learn(total_timesteps=settings.training_steps_cap)

    agent.save()
def test(env, agent, filepath):
    model = PPO2.load(filepath)
    obs = env.reset()
    for i in range(settings.testing_nb_episodes_per_model):
        action, _states = model.predict(obs)
        obs, rewards, dones, info = env.step(action)

if __name__ == "__main__":
    env, agent = setup()
    train()
