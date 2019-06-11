import sys
import gym

import os
import tensorflow as tf
os.sys.path.insert(0, os.path.abspath('../../../settings_folder'))
import settings
import gym
import tensorflow as tf
import numpy as np

from stable_baselines.common.policies import FeedForwardPolicy, ActorCriticPolicy, register_policy, nature_cnn
from stable_baselines.a2c.utils import conv_grey, linear, conv_to_fc, batch_to_seq, seq_to_batch, lstm
from stable_baselines.common.vec_env import DummyVecEnv
from stable_baselines import A2C

class MultiInputPolicy(ActorCriticPolicy):
    def __init__(self, sess, ob_space, ac_space, n_env, n_steps, n_batch, reuse=False, **kwargs):
        super(MultiInputPolicy, self).__init__(sess, ob_space, ac_space, n_env, n_steps, n_batch, reuse=reuse, scale=False)

        with tf.variable_scope("model", reuse=reuse):

            #input = tf.placeholder(shape=self.processed_obs.shape,dtype=self.processed_obs.dtype )

            depth = self.processed_obs[n_batch:, :, :(settings.encoded_depth_H * settings.encoded_depth_W)]
            pos = self.processed_obs[n_batch:, :, (settings.encoded_depth_H * settings.encoded_depth_W):]

            if(n_batch == None):
                depth = tf.reshape(depth, shape=(-1, settings.encoded_depth_H, settings.encoded_depth_W, 1))
                pos = tf.reshape(pos, shape=(-1,1,settings.position_depth))
            else:
                depth = tf.reshape(depth, shape=(n_batch, settings.encoded_depth_H, settings.encoded_depth_W, 1))
                pos = tf.reshape(pos, shape=(n_batch, 1, settings.position_depth))

            # Convolutions on Depth Images
            activ = tf.nn.relu
            layer_1 = activ(conv_grey(depth, 'c1', n_filters=32, filter_size=8, stride=4, init_scale=np.sqrt(2), **kwargs))
            layer_2 = activ(conv_grey(layer_1, 'c2', n_filters=64, filter_size=4, stride=2, init_scale=np.sqrt(2), **kwargs))
            layer_3 = activ(conv_grey(layer_2, 'c3', n_filters=64, filter_size=3, stride=1, init_scale=np.sqrt(2), **kwargs))
            layer_3 = conv_to_fc(layer_3)

            image_encoded = tf.keras.layers.Flatten()(layer_3)

            # Fully connected layers for pos vector

            pos_layer_1 = tf.keras.layers.Dense(32, activation='relu', name= 'pos_fc_1')(pos)
            pos_layer_2 = tf.keras.layers.Dense(32, activation='relu', name= 'pos_fc_2')(pos_layer_1)
            pos_encoded = tf.keras.layers.Flatten()(pos_layer_2)
            joint_encoding = tf.keras.layers.concatenate([image_encoded, pos_encoded])
            x = tf.keras.layers.Dense(64, activation="tanh", name='pi_fc_0')(joint_encoding)
            pi_latent = tf.keras.layers.Dense(64, activation="tanh", name='pi_fc_1')(x)

            x1 = tf.keras.layers.Dense(64, activation="tanh", name='vf_fc_0')(joint_encoding)
            vf_latent = tf.keras.layers.Dense(64, activation="tanh", name='vf_fc_1')(x1)

            value_fn = tf.keras.layers.Dense(1, name='vf')(vf_latent)

            self.proba_distribution, self.policy, self.q_value = \
                self.pdtype.proba_distribution_from_latent(pi_latent, vf_latent, init_scale=0.01)

        self.value_fn = value_fn
        self.initial_state = None
        self._setup_init()

    def step(self, obs, state=None, mask=None, deterministic=False):
        if deterministic:
            action, value, neglogp = self.sess.run([self.deterministic_action, self._value, self.neglogp],
                                                   {self.obs_ph: obs})
        else:
            action, value, neglogp = self.sess.run([self.action, self._value, self.neglogp],
                                                   {self.obs_ph: obs})
        return action, value, self.initial_state, neglogp

    def proba_step(self, obs, state=None, mask=None):
        return self.sess.run(self.policy_proba, {self.obs_ph: obs})

    def value(self, obs, state=None, mask=None):
        return self.sess.run(self._value, {self.obs_ph: obs})



#register_policy('CustomPolicy', CustomPolicy)




