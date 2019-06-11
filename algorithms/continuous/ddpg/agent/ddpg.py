import numpy as np
import random
import math

from keras.models import model_from_json, Model

from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation, Flatten
from keras.optimizers import Adam
import tensorflow as tf
from keras.backend.tensorflow_backend import set_session
# from keras.engine.training import collect_trainable_weights
import json


class DDPGAgent(object):
	def __init__(self, gamma, batch_size, tau, lra, lrc, num_steps, max_steps,
	             episode_count, throttle, min_duration, env, actor, critic, replay_buffer,
	             random_process=None, **kwargs):

		# parameters
		self.gamma = gamma
		self.batch_size = batch_size
		self.tau = tau
		self.lra = lra
		self.lrc = lrc
		self.num_steps = num_steps
		self.max_steps = max_steps
		self.episode_count = episode_count
		self.epsilon = 1
		self.explore = 100000.
		self.throttle = throttle
		self.min_duration = min_duration
		self.step = 0

		# Related Objects
		self.env = env
		self.actor = actor
		self.critic = critic
		self.replay_buffer = replay_buffer
		self.random_process = random_process

		super(DDPGAgent, self).__init__(**kwargs)

	def train(self):
		action_dim = self.env.action_space.shape[0]  # pitch, roll, throttle, yaw_rate and duration
		step = 0
		for i in range(self.episode_count):
			print("Episode : " + str(i) + " Replay Buffer " + str(self.replay_buffer.count()))
			self.env._reset()
			total_reward = 0.
			# Todo: figure out a way to directly manipulate state variable instead of copying
			# self.depth_front, self.grey_front, self.depth_bottom, self.grey_bottom, self.depth_back, self.grey_back

			depth_front = self.env.state()[0]
			depth_front = np.expand_dims(depth_front, axis=0)  # for channel ( 1, 154, 256)
			depth_front = np.expand_dims(depth_front, axis=0)  # for sample ( 1, 1, 154, 256)

			grey_front = self.env.state()[1]
			grey_front = np.expand_dims(grey_front, axis=0)  # for channel ( 1, 144, 256)
			grey_front = np.expand_dims(grey_front, axis=0)  # for sample (1, 1, 144, 256)

			depth_bottom = self.env.state()[2]
			depth_bottom = np.expand_dims(depth_bottom, axis=0)  # for channel ( 1, 154, 256)
			depth_bottom = np.expand_dims(depth_bottom, axis=0)  # for sample ( 1, 1, 154, 256)

			grey_bottom = self.env.state()[3]
			grey_bottom = np.expand_dims(grey_bottom, axis=0)  # for channel ( 1, 144, 256)
			grey_bottom = np.expand_dims(grey_bottom, axis=0)  # for sample (1, 1, 144, 256)

			depth_back = self.env.state()[4]
			depth_back = np.expand_dims(depth_back, axis=0)  # for channel ( 1, 154, 256)
			depth_back = np.expand_dims(depth_back, axis=0)  # for sample ( 1, 1, 154, 256)

			grey_back = self.env.state()[5]
			grey_back = np.expand_dims(grey_back, axis=0)  # for channel ( 1, 144, 256)
			grey_back = np.expand_dims(grey_back, axis=0)  # for sample (1, 1, 144, 256)

			vel = self.env.state()[6]
			vel = np.expand_dims(vel, axis=0)  # for channel (1, 3)
			vel = np.expand_dims(vel, axis=0)  # for sample (1, 1, 3)
			pos = self.env.state()[7]
			pos = np.expand_dims(pos, axis=0)  # for channel (1, 4)
			pos = np.expand_dims(pos, axis=0)  # for sample  (1, 1, 4)

			s_t = [depth_front, grey_front, depth_bottom, grey_bottom, depth_back, grey_back, vel, pos]

			for j in range(self.max_steps):
				loss = 0
				self.epsilon -= 1.0 / self.explore
				a_t = np.zeros([1, action_dim])
				noise_t = np.zeros([1, action_dim])

				a_t_original = self.actor.model.predict(s_t)
				# TODO: use a generic logger?
				with open("actions_from_network.txt", "a") as myfile:
					myfile.write(str([a_t_original[0][0], a_t_original[0][1], a_t_original[0][2]]) + ", ")

				state, r_t, done, info = self.env._step(a_t_original)

				depth_front_t1 = np.expand_dims(state[0], axis=0)
				depth_front_t1 = np.expand_dims(depth_front_t1, axis=0)
				grey_front_t1 = np.expand_dims(state[1], axis=0)
				grey_front_t1 = np.expand_dims(grey_front_t1, axis=0)

				depth_bottom_t1 = np.expand_dims(state[2], axis=0)
				depth_bottom_t1 = np.expand_dims(depth_bottom_t1, axis=0)
				grey_bottom_t1 = np.expand_dims(state[3], axis=0)
				grey_bottom_t1 = np.expand_dims(grey_bottom_t1, axis=0)

				depth_back_t1 = np.expand_dims(state[4], axis=0)
				depth_back_t1 = np.expand_dims(depth_back_t1, axis=0)
				grey_back_t1 = np.expand_dims(state[5], axis=0)
				grey_back_t1 = np.expand_dims(grey_back_t1, axis=0)

				vel_t1 = np.expand_dims(state[6], axis=0)
				vel_t1 = np.expand_dims(vel_t1, axis=0)
				pos_t1 = np.expand_dims(state[7], axis=0)
				pos_t1 = np.expand_dims(pos_t1, axis=0)

				s_t1 = [depth_front_t1, grey_front_t1, depth_bottom_t1, grey_bottom_t1, depth_back_t1, grey_back_t1,
				        vel_t1, pos_t1]

				# replay buffer
				self.replay_buffer.add(s_t, a_t[0], r_t, s_t1, done)

				# Do the batch update
				batch = self.replay_buffer.getBatch(self.batch_size)

				depths_front_t = np.asarray([e[0][0][0] for e in batch])
				greys_front_t = np.asarray([e[0][1][0] for e in batch])
				depths_bottom_t = np.asarray([e[0][2][0] for e in batch])
				greys_bottom_t = np.asarray([e[0][3][0] for e in batch])
				depths_back_t = np.asarray([e[0][4][0] for e in batch])
				greys_back_t = np.asarray([e[0][5][0] for e in batch])
				vels_t = np.asarray([e[0][6][0] for e in batch])
				poss_t = np.asarray([e[0][7][0] for e in batch])
				actions = np.asarray([e[1] for e in batch])
				rewards = np.asarray([e[2] for e in batch])
				depths_front_t1 = np.asarray([e[3][0][0] for e in batch])
				greys_front_t1 = np.asarray([e[3][1][0] for e in batch])
				depths_bottom_t1 = np.asarray([e[3][2][0] for e in batch])
				greys_bottom_t1 = np.asarray([e[3][3][0] for e in batch])
				depths_back_t1 = np.asarray([e[3][4][0] for e in batch])
				greys_back_t1 = np.asarray([e[3][5][0] for e in batch])
				vels_t1 = np.asarray([e[3][6][0] for e in batch])
				poss_t1 = np.asarray([e[3][7][0] for e in batch])
				dones = np.asarray([e[4] for e in batch])
				y_t = np.asarray([e[1] for e in batch])

				action_target = self.actor.target_model.predict([depths_front_t1, greys_front_t1, depths_bottom_t1,
				                                                 greys_bottom_t1, depths_back_t1, greys_back_t1,
				                                                 vels_t1, poss_t1])

				target_q_values = self.critic.target_model.predict([depths_front_t1, greys_front_t1, depths_bottom_t1,
				                                                    greys_bottom_t1, depths_back_t1, greys_back_t1,
				                                                    vels_t1, poss_t1, action_target])

				for k in range(len(batch)):
					if dones[k]:
						y_t[k] = rewards[k]
					else:
						y_t[k] = rewards[k] + self.gamma * target_q_values[k]

				loss += self.critic.model.train_on_batch([depths_front_t, greys_front_t, depths_bottom_t,
				                                          greys_bottom_t, depths_back_t, greys_back_t,
				                                          vels_t, poss_t, action_target], y_t)

				a_for_grad = self.actor.model.predict([depths_front_t, greys_front_t, depths_bottom_t,
				                                       greys_bottom_t, depths_back_t, greys_back_t,
				                                       vels_t, poss_t])

				grads = self.critic.gradients([depths_front_t, greys_front_t, depths_bottom_t,
				                               greys_bottom_t, depths_back_t, greys_back_t,
				                               vels_t, poss_t], a_for_grad)

				self.actor.train([depths_front_t, greys_front_t, depths_bottom_t,
				                  greys_bottom_t, depths_back_t, greys_back_t,
				                  vels_t, poss_t], grads)

				self.actor.target_train()
				self.critic.target_train()

				total_reward += r_t
				s_t = s_t1

				print("Episode", i, "Step", step, "Reward", r_t, "Loss", loss)

				step += 1
				if (done or step > self.num_steps):
					break

			print("TOTAL REWARD @ " + str(i) + "-th Episode  : Reward " + str(total_reward))
			print("Total Step: " + str(step))
			print("")

	def check_for_weights(self, weight_files):
		try:
			self.actor.model.load_weights(weight_files["actor"])
			self.critic.model.load_weights(weight_files["critic"])
			self.actor.target_model.load_weights(weight_files["actor"])
			self.critic.target_model.load_weights(weight_files["critic"])
			return True
		except:
			print("Cannot find weights! Exiting!")
			return False

	def test(self, env, weight_files):
		self.check_for_weights(weight_files)
		action_dim = self.env.action_space.shape[0]  # pitch, roll, throttle, yaw_rate and duration
		step = 0

		for i in range(self.episode_count):
			print("Episode : " + str(i) + " Replay Buffer " + str(self.replay_buffer.count()))
			self.env._reset()
			total_reward = 0.
			# Todo: figure out a way to directly manipulate state variable instead of copying
			depth = self.env.state()[0]
			depth = np.expand_dims(depth, axis=0)  # for channel ( 1, 154, 256)
			depth = np.expand_dims(depth, axis=0)  # for sample ( 1, 1, 154, 256)

			grey = self.env.state()[1]
			grey = np.expand_dims(grey, axis=0)  # for channel ( 1, 144, 256)
			grey = np.expand_dims(grey, axis=0)  # for sample (1, 1, 144, 256)
			vel = self.env.state()[2]
			vel = np.expand_dims(vel, axis=0)  # for channel (1, 3)
			vel = np.expand_dims(vel, axis=0)  # for sample (1, 1, 3)
			pos = self.env.state()[3]
			pos = np.expand_dims(pos, axis=0)  # for channel (1, 4)
			pos = np.expand_dims(pos, axis=0)  # for sample  (1, 1, 4)

			s_t = [depth, grey, vel, pos]

			for j in range(self.max_steps):
				loss = 0
				self.epsilon -= 1.0 / self.explore
				a_t = np.zeros([1, action_dim])
				noise_t = np.zeros([1, action_dim])

				a_t_original = self.actor.model.predict(s_t)
				# TODO: use a generic logger?
				with open("actions_from_network.txt", "a") as myfile:
					myfile.write(str([a_t_original[0][0], a_t_original[0][1], a_t_original[0][2]]) + ", ")

				a_t[0][0] = a_t_original[0][0]
				a_t[0][1] = a_t_original[0][1]
				a_t[0][2] = a_t_original[0][2]

				throttle = self.throttle + random.gauss(0.1, 0.05)
				duration = self.min_duration
				action = [a_t[0][0], a_t[0][1], throttle, a_t[0][2], duration]

				state, r_t, done, info = self.env._step(action)

				depth_t1 = np.expand_dims(state[0], axis=0)
				depth_t1 = np.expand_dims(depth_t1, axis=0)
				grey_t1 = np.expand_dims(state[1], axis=0)
				grey_t1 = np.expand_dims(grey_t1, axis=0)
				vel_t1 = np.expand_dims(state[2], axis=0)
				vel_t1 = np.expand_dims(vel_t1, axis=0)
				pos_t1 = np.expand_dims(state[3], axis=0)
				pos_t1 = np.expand_dims(pos_t1, axis=0)

				s_t1 = [depth_t1, grey_t1, vel_t1, pos_t1]

				# replay buffer
				self.replay_buffer.add(s_t, a_t[0], r_t, s_t1, done)

				# Do the batch update
				batch = self.replay_buffer.getBatch(self.batch_size)

				depths_t = np.asarray([e[0][0][0] for e in batch])
				greys_t = np.asarray([e[0][1][0] for e in batch])
				vels_t = np.asarray([e[0][2][0] for e in batch])
				poss_t = np.asarray([e[0][3][0] for e in batch])
				actions = np.asarray([e[1] for e in batch])
				rewards = np.asarray([e[2] for e in batch])
				depths_t1 = np.asarray([e[3][0][0] for e in batch])
				greys_t1 = np.asarray([e[3][1][0] for e in batch])
				vels_t1 = np.asarray([e[3][2][0] for e in batch])
				poss_t1 = np.asarray([e[3][3][0] for e in batch])
				dones = np.asarray([e[4] for e in batch])
				y_t = np.asarray([e[1] for e in batch])

				action_target = self.actor.target_model.predict([depths_t1, greys_t1, vels_t1, poss_t1])
				target_q_values = self.critic.target_model.predict(
					[depths_t1, greys_t1, vels_t1, poss_t1, action_target])

				for k in range(len(batch)):
					if dones[k]:
						y_t[k] = rewards[k]
					else:
						y_t[k] = rewards[k] + self.gamma * target_q_values[k]

				total_reward += r_t
				s_t = s_t1

				print("Episode", i, "Step", step, "Action", action, "Reward", r_t, "Loss", loss)

				step += 1
				if (done or step > self.num_steps):
					break

			print("TOTAL REWARD @ " + str(i) + "-th Episode  : Reward " + str(total_reward))
			print("Total Step: " + str(step))
			print("")
