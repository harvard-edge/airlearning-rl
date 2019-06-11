import numpy as np
import math
from keras.models import model_from_json
from keras.models import Sequential, Model
from keras.constraints import max_norm
# from keras.engine.training import collect_trainable_weights
from keras.layers import Conv2D, Activation, Dense, Flatten, Input, merge, Lambda, concatenate, BatchNormalization, \
	initializers
from keras.optimizers import Adam
import tensorflow as tf
import keras.backend as K

HIDDEN1_UNITS = 32
HIDDEN2_UNITS = 64
HIDDEN3_UNITS = 128
HIDDEN4_UNITS = 256
HIDDEN5_UNITS = 512
HIDDEN6_UNITS = 1024


class ActorNetwork(object):
	def __init__(self, sess, depth_front_size, grey_front_size, depth_bottom_size, grey_bottom_size, depth_back_size,
	             grey_back_size, vel_size, pos_size, action_dim, BATCH_SIZE, TAU, LEARNING_RATE):
		self.sess = sess
		self.BATCH_SIZE = BATCH_SIZE
		self.TAU = TAU
		self.LEARNING_RATE = LEARNING_RATE

		K.set_session(sess)

		# Now create the model
		self.model, self.weights, self.depth_front, self.grey_front, self.depth_bottom, self.grey_bottom, \
		self.depth_back, self.grey_back, self.vel, self.pos = self.create_actor_network(depth_front_size,
		                                                                                grey_front_size,
		                                                                                depth_bottom_size,
		                                                                                grey_bottom_size,
		                                                                                depth_back_size,
		                                                                                grey_back_size,
		                                                                                vel_size,
		                                                                                pos_size,
		                                                                                action_dim
		                                                                                )
		self.target_model, self.target_weights, self.target_depth_front, self.target_grey_front, \
		self.target_depth_bottom, self.target_grey_bottom, self.target_depth_back, self.target_grey_back, \
		self.target_vel, self.target_pos = self.create_actor_network(depth_front_size, grey_front_size,
		                                                             depth_bottom_size, grey_bottom_size,
		                                                             depth_back_size, grey_back_size, vel_size,
		                                                             pos_size, action_dim
		                                                             )
		self.action_gradient = tf.placeholder(tf.float32, [None, action_dim])
		self.params_grad = tf.gradients(self.model.output, self.weights, -self.action_gradient)
		grads = zip(self.params_grad, self.weights)
		self.optimize = tf.train.AdamOptimizer(LEARNING_RATE).apply_gradients(grads)
		self.sess.run(tf.initialize_all_variables())

	def train(self, states, action_grads):
		self.sess.run(self.optimize, feed_dict={
			self.depth_front: states[0],
			self.grey_front: states[1],
			self.depth_bottom: states[2],
			self.grey_bottom: states[3],
			self.depth_back: states[4],
			self.grey_back: states[5],
			self.vel: states[6],
			self.pos: states[7],
			self.action_gradient: action_grads
		})

	def target_train(self):
		actor_weights = self.model.get_weights()
		actor_target_weights = self.target_model.get_weights()
		for i in range(len(actor_weights)):
			actor_target_weights[i] = self.TAU * actor_weights[i] + (1 - self.TAU) * actor_target_weights[i]
		self.target_model.set_weights(actor_target_weights)

	def create_actor_network(self, depth_front_size, grey_front_size, depth_bottom_size, grey_bottom_size,
	                         depth_back_size, grey_back_size, vel_size, pos_size, action_dim):
		print("Now we build the model")
		depth_front_image = Input(shape=depth_front_size)
		grey_front_image = Input(shape=grey_front_size)
		depth_bottom_image = Input(shape=depth_bottom_size)
		grey_bottom_image = Input(shape=grey_bottom_size)
		depth_back_image = Input(shape=depth_back_size)
		grey_back_image = Input(shape=grey_back_size)
		vel = Input(shape=vel_size)
		pos = Input(shape=pos_size)

		# build conv layer for Grey front image
		# gfi : grey forward image
		gfi = Conv2D(32, (4, 4), strides=(4, 4), kernel_initializer=initializers.random_normal(stddev=1e-4),
		             padding='same', kernel_constraint=max_norm(0.07))(grey_front_image)
		gfi = Activation('relu')(gfi)
		gfi = BatchNormalization()(gfi)

		gfi = Conv2D(64, (3, 3), strides=(3, 3), kernel_initializer=initializers.random_normal(stddev=1e-4),
		             padding='same', kernel_constraint=max_norm(0.07))(gfi)
		gfi = Activation('relu')(gfi)
		gfi = BatchNormalization()(gfi)

		gfi = Conv2D(64, (2, 2), strides=(1, 1), kernel_initializer=initializers.random_normal(stddev=1e-4),
		             padding='same', kernel_constraint=max_norm(0.07))(gfi)
		gfi = Activation('relu')(gfi)
		gfi = BatchNormalization()(gfi)
		gfi = Flatten()(gfi)

		# build conv layer for Grey bottom image
		# gboti : grey bottom image
		gboti = Conv2D(32, (4, 4), strides=(4, 4), kernel_initializer=initializers.random_normal(stddev=1e-4),
		               padding='same', kernel_constraint=max_norm(0.07))(grey_bottom_image)
		gboti = Activation('relu')(gboti)
		gboti = BatchNormalization()(gboti)

		gboti = Conv2D(64, (3, 3), strides=(3, 3), kernel_initializer=initializers.random_normal(stddev=1e-4),
		               padding='same', kernel_constraint=max_norm(0.07))(gboti)
		gboti = Activation('relu')(gboti)
		gboti = BatchNormalization()(gboti)

		gboti = Conv2D(64, (2, 2), strides=(1, 1), kernel_initializer=initializers.random_normal(stddev=1e-4),
		               padding='same', kernel_constraint=max_norm(0.07))(gboti)
		gboti = Activation('relu')(gboti)
		gboti = BatchNormalization()(gboti)
		gboti = Flatten()(gboti)

		# build conv layer for Grey bottom image
		# gbi grey back image
		gbi = Conv2D(32, (4, 4), strides=(4, 4), kernel_initializer=initializers.random_normal(stddev=1e-4),
		             padding='same', kernel_constraint=max_norm(0.07))(grey_back_image)
		gbi = Activation('relu')(gbi)
		gbi = BatchNormalization()(gbi)

		gbi = Conv2D(64, (3, 3), strides=(3, 3), kernel_initializer=initializers.random_normal(stddev=1e-4),
		             padding='same', kernel_constraint=max_norm(0.07))(gbi)
		gbi = Activation('relu')(gbi)
		gbi = BatchNormalization()(gbi)

		gbi = Conv2D(64, (2, 2), strides=(1, 1), kernel_initializer=initializers.random_normal(stddev=1e-4),
		             padding='same', kernel_constraint=max_norm(0.07))(gbi)
		gbi = Activation('relu')(gbi)
		gbi = BatchNormalization()(gbi)
		gbi = Flatten()(gbi)

		# build conv layer for depth image
		dfi = Conv2D(16, (3, 3), strides=(4, 4), kernel_initializer=initializers.random_normal(stddev=1e-4),
		             padding='same', kernel_constraint=max_norm(0.07))(depth_front_image)
		dfi = Activation('relu')(dfi)
		dfi = BatchNormalization()(dfi)
		dfi = Conv2D(32, (3, 3), strides=(3, 3), kernel_initializer=initializers.random_normal(stddev=1e-4),
		             padding='same', kernel_constraint=max_norm(0.07))(dfi)
		dfi = Activation('relu')(dfi)
		dfi = BatchNormalization()(dfi)
		dfi = Flatten()(dfi)

		# build conv layer for depth image
		# dboti: depth bottom image
		dboti = Conv2D(16, (3, 3), strides=(4, 4), kernel_initializer=initializers.random_normal(stddev=1e-4),
		               padding='same', kernel_constraint=max_norm(0.07))(depth_bottom_image)
		dboti = Activation('relu')(dboti)
		dboti = BatchNormalization()(dboti)
		dboti = Conv2D(32, (3, 3), strides=(3, 3), kernel_initializer=initializers.random_normal(stddev=1e-4),
		               padding='same', kernel_constraint=max_norm(0.07))(dboti)
		dboti = Activation('relu')(dboti)
		dboti = BatchNormalization()(dboti)
		dboti = Flatten()(dboti)

		# build conv layer for depth image
		# dbi: depth back image
		dbi = Conv2D(16, (3, 3), strides=(4, 4), kernel_initializer=initializers.random_normal(stddev=1e-4),
		             padding='same', kernel_constraint=max_norm(0.07))(depth_back_image)
		dbi = Activation('relu')(dbi)
		dbi = BatchNormalization()(dbi)
		dbi = Conv2D(32, (3, 3), strides=(3, 3), kernel_initializer=initializers.random_normal(stddev=1e-4),
		             padding='same', kernel_constraint=max_norm(0.07))(dbi)
		dbi = Activation('relu')(dbi)
		dbi = BatchNormalization()(dbi)
		dbi = Flatten()(dbi)

		h0_vel = Dense(HIDDEN1_UNITS, activation='relu', kernel_initializer=initializers.random_normal(stddev=1e-4),
		               kernel_constraint=max_norm(0.07))(vel)
		h0_vel = BatchNormalization()(h0_vel)
		h1_vel = Dense(HIDDEN2_UNITS, activation='relu', kernel_initializer=initializers.random_normal(stddev=1e-4),
		               kernel_constraint=max_norm(0.07))(h0_vel)
		h1_vel = BatchNormalization()(h1_vel)
		h1_vel = Flatten()(h1_vel)
		h0_pos = Dense(HIDDEN1_UNITS, activation='relu', kernel_initializer=initializers.random_normal(stddev=1e-4),
		               kernel_constraint=max_norm(0.07))(pos)
		h0_pos = BatchNormalization()(h0_pos)
		h1_pos = Dense(HIDDEN2_UNITS, activation='relu', kernel_initializer=initializers.random_normal(stddev=1e-4),
		               kernel_constraint=max_norm(0.07))(h0_pos)
		h1_pos = BatchNormalization()(h1_pos)
		h1_pos = Flatten()(h1_pos)
		sensor_fused = concatenate([dfi, gfi, dboti, gboti, dbi, gbi, h1_vel, h1_pos])

		d1 = Dense(HIDDEN2_UNITS, activation='relu', kernel_initializer=initializers.random_normal(stddev=1e-4),
		           kernel_constraint=max_norm(0.07))(sensor_fused)
		d1 = BatchNormalization()(d1)
		d2 = Dense(HIDDEN2_UNITS, activation='relu', kernel_initializer=initializers.random_normal(stddev=1e-4),
		           kernel_constraint=max_norm(0.07))(d1)
		d2 = BatchNormalization()(d2)
		d3 = Dense(HIDDEN2_UNITS, activation='relu', kernel_initializer=initializers.random_normal(stddev=1e-4),
		           kernel_constraint=max_norm(0.07))(d2)

		pitch = Dense(1, activation='tanh', kernel_initializer=initializers.random_normal(stddev=1e-4),
		              kernel_constraint=max_norm(0.07))(d3)
		roll = Dense(1, activation='tanh', kernel_initializer=initializers.random_normal(stddev=1e-4),
		             kernel_constraint=max_norm(0.07))(d3)
		yaw_rate = Dense(1, activation='tanh', kernel_initializer=initializers.random_normal(stddev=1e-4),
		                 kernel_constraint=max_norm(0.07))(d3)

		flight_control = concatenate([pitch, roll, yaw_rate])
		model = Model(
			input=[depth_front_image, grey_front_image, depth_bottom_image, grey_bottom_image, depth_back_image,
			       grey_back_image, vel, pos], output=flight_control)
		print(model.summary())
		return model, model.trainable_weights, depth_front_image, grey_front_image, depth_bottom_image, grey_bottom_image, depth_back_image, grey_back_image, vel, pos
