import numpy as np
import math
# from keras.initializations import normal, identity
from keras.models import model_from_json, load_model
# from keras.engine.training import collect_trainable_weights
from keras.models import Sequential
from keras.layers import Conv2D, Dense, Flatten, Input, merge, Lambda, Activation, concatenate, BatchNormalization, \
	initializers
from keras.models import Sequential, Model
from keras.optimizers import Adam
from keras.constraints import max_norm
import keras.backend as K
import tensorflow as tf

HIDDEN1_UNITS = 32
HIDDEN2_UNITS = 64
HIDDEN3_UNITS = 128
HIDDEN4_UNITS = 256
HIDDEN5_UNITS = 512
HIDDEN6_UNITS = 1024


class CriticNetwork(object):
	def __init__(self, sess, depth_front_size, grey_front_size, depth_bottom_size, grey_bottom_size, depth_back_size,
	             grey_back_size, vel_size, pos_size, action_size, BATCH_SIZE, TAU, LEARNING_RATE):
		self.sess = sess
		self.BATCH_SIZE = BATCH_SIZE
		self.TAU = TAU
		self.LEARNING_RATE = LEARNING_RATE
		self.action_size = action_size

		K.set_session(sess)

		# Now create the model
		self.model, self.action, self.depth_front, self.grey_front, self.depth_bottom, self.grey_bottom, self.depth_back, \
		self.grey_back, self.vel, self.pos = self.create_critic_network(
			depth_front_size, grey_front_size, depth_bottom_size, grey_bottom_size,
			depth_back_size, grey_back_size, vel_size, pos_size, action_size
		)
		self.target_model, self.target_action, self.target_depth_front, self.target_grey_front, self.target_depth_bottom, \
		self.target_grey_bottom, self.target_depth_back, self.target_grey_back, \
		self.target_vel, self.target_pos = self.create_critic_network(
			depth_front_size, grey_front_size, depth_bottom_size, grey_bottom_size,
			depth_back_size, grey_back_size, vel_size, pos_size, action_size
		)
		self.action_grads = tf.gradients(self.model.output, self.action)  # GRADIENTS for policy update
		self.sess.run(tf.initialize_all_variables())

	def gradients(self, states, actions):
		return self.sess.run(self.action_grads, feed_dict={
			self.depth_front: states[0],
			self.grey_front: states[1],
			self.depth_bottom: states[2],
			self.grey_bottom: states[3],
			self.depth_back: states[4],
			self.grey_back: states[5],
			self.vel: states[6],
			self.pos: states[7],
			self.action: actions
		})[0]

	def target_train(self):
		critic_weights = self.model.get_weights()
		critic_target_weights = self.target_model.get_weights()
		for i in range(len(critic_weights)):
			critic_target_weights[i] = self.TAU * critic_weights[i] + (1 - self.TAU) * critic_target_weights[i]
		self.target_model.set_weights(critic_target_weights)

	def create_critic_network(self, depth_front_size, grey_front_size, depth_bottom_size, grey_bottom_size,
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

		# build conv layer for grey image
		gfi = Conv2D(32, (4, 4), strides=(4, 4), kernel_initializer=initializers.random_normal(stddev=1e-4),
		             padding='same', kernel_constraint=max_norm(0.07))(grey_front_image)
		gfi = Activation('relu')(gfi)
		gfi = BatchNormalization()(gfi)

		gfi = Conv2D(32, (3, 3), strides=(3, 3), kernel_initializer=initializers.random_normal(stddev=1e-4),
		             padding='same', kernel_constraint=max_norm(0.07))(gfi)
		gfi = Activation('relu')(gfi)
		gfi = BatchNormalization()(gfi)

		gfi = Conv2D(64, (3, 3), strides=(3, 3), kernel_initializer=initializers.random_normal(stddev=1e-4),
		             padding='same', kernel_constraint=max_norm(0.07))(gfi)
		gfi = Activation('relu')(gfi)
		gfi = BatchNormalization()(gfi)
		gfi = Flatten()(gfi)

		# build conv layer for grey image
		gboti = Conv2D(32, (4, 4), strides=(4, 4), kernel_initializer=initializers.random_normal(stddev=1e-4),
		               padding='same', kernel_constraint=max_norm(0.07))(grey_bottom_image)
		gboti = Activation('relu')(gboti)
		gboti = BatchNormalization()(gboti)

		gboti = Conv2D(32, (3, 3), strides=(3, 3), kernel_initializer=initializers.random_normal(stddev=1e-4),
		               padding='same', kernel_constraint=max_norm(0.07))(gboti)
		gboti = Activation('relu')(gboti)
		gboti = BatchNormalization()(gboti)

		gboti = Conv2D(64, (3, 3), strides=(3, 3), kernel_initializer=initializers.random_normal(stddev=1e-4),
		               padding='same', kernel_constraint=max_norm(0.07))(gboti)
		gboti = Activation('relu')(gboti)
		gboti = BatchNormalization()(gboti)
		gboti = Flatten()(gboti)

		# build conv layer for grey image
		gbi = Conv2D(32, (4, 4), strides=(4, 4), kernel_initializer=initializers.random_normal(stddev=1e-4),
		             padding='same', kernel_constraint=max_norm(0.07))(grey_back_image)
		gbi = Activation('relu')(gbi)
		gbi = BatchNormalization()(gbi)

		gbi = Conv2D(32, (3, 3), strides=(3, 3), kernel_initializer=initializers.random_normal(stddev=1e-4),
		             padding='same', kernel_constraint=max_norm(0.07))(gbi)
		gbi = Activation('relu')(gbi)
		gbi = BatchNormalization()(gbi)

		gbi = Conv2D(64, (3, 3), strides=(3, 3), kernel_initializer=initializers.random_normal(stddev=1e-4),
		             padding='same', kernel_constraint=max_norm(0.07))(gbi)
		gbi = Activation('relu')(gbi)
		gbi = BatchNormalization()(gbi)
		gbi = Flatten()(gbi)

		# build conv layer for depth image
		dfi = Conv2D(32, (3, 3), strides=(4, 4), kernel_initializer=initializers.random_normal(stddev=1e-4),
		             padding='same', kernel_constraint=max_norm(0.07))(depth_front_image)
		dfi = Activation('relu')(dfi)
		dfi = BatchNormalization()(dfi)
		dfi = Conv2D(32, (3, 3), strides=(3, 3), kernel_initializer=initializers.random_normal(stddev=1e-4),
		             padding='same', kernel_constraint=max_norm(0.07))(dfi)
		dfi = Activation('relu')(dfi)
		dfi = BatchNormalization()(dfi)
		dfi = Conv2D(64, (3, 3), strides=(3, 3), kernel_initializer=initializers.random_normal(stddev=1e-4),
		             padding='same', kernel_constraint=max_norm(0.07))(dfi)
		dfi = BatchNormalization()(dfi)
		dfi = Flatten()(dfi)

		# build conv layer for depth image
		dboti = Conv2D(32, (3, 3), strides=(4, 4), kernel_initializer=initializers.random_normal(stddev=1e-4),
		               padding='same', kernel_constraint=max_norm(0.07))(depth_bottom_image)
		dboti = Activation('relu')(dboti)
		dboti = BatchNormalization()(dboti)
		dboti = Conv2D(32, (3, 3), strides=(3, 3), kernel_initializer=initializers.random_normal(stddev=1e-4),
		               padding='same', kernel_constraint=max_norm(0.07))(dboti)
		dboti = Activation('relu')(dboti)
		dboti = BatchNormalization()(dboti)
		dboti = Conv2D(64, (3, 3), strides=(3, 3), kernel_initializer=initializers.random_normal(stddev=1e-4),
		               padding='same', kernel_constraint=max_norm(0.07))(dboti)
		dboti = BatchNormalization()(dboti)
		dboti = Flatten()(dboti)

		# build conv layer for depth image
		dbi = Conv2D(32, (3, 3), strides=(4, 4), kernel_initializer=initializers.random_normal(stddev=1e-4),
		             padding='same', kernel_constraint=max_norm(0.07))(depth_back_image)
		dbi = Activation('relu')(dbi)
		dbi = BatchNormalization()(dbi)
		dbi = Conv2D(32, (3, 3), strides=(3, 3), kernel_initializer=initializers.random_normal(stddev=1e-4),
		             padding='same', kernel_constraint=max_norm(0.07))(dbi)
		dbi = Activation('relu')(dbi)
		dbi = BatchNormalization()(dbi)
		dbi = Conv2D(64, (3, 3), strides=(3, 3), kernel_initializer=initializers.random_normal(stddev=1e-4),
		             padding='same', kernel_constraint=max_norm(0.07))(dbi)
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
		d3 = BatchNormalization()(d3)

		A = Input(shape=[action_dim], name='action2')
		a1 = Dense(HIDDEN2_UNITS, activation='linear', kernel_constraint=max_norm(0.07))(A)

		h2 = concatenate([d3, a1])
		h3 = Dense(HIDDEN2_UNITS, activation='relu', kernel_initializer=initializers.random_normal(stddev=1e-4),
		           kernel_constraint=max_norm(0.07))(h2)
		V = Dense(action_dim, activation='linear')(h3)
		model = Model(
			input=[depth_front_image, grey_front_image, depth_bottom_image, grey_bottom_image, depth_back_image,
			       grey_back_image, vel, pos, A], output=V)
		print(model.summary())
		adam = Adam(lr=self.LEARNING_RATE)
		model.compile(loss='mse', optimizer=adam)
		return model, A, depth_front_image, grey_front_image, depth_bottom_image, grey_bottom_image, depth_back_image, grey_back_image, vel, pos
