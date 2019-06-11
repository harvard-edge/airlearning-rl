#ining
import os, sys
import numpy as np
import tensorflow as tf
import gym

from keras.models import Sequential, Model
from keras.layers import Dense, Flatten, Conv2D, concatenate, Input, Reshape
from keras.optimizers import Adam
from keras.backend.tensorflow_backend import set_session

from rl.agents.dqn import DQNAgent
from rl.policy import LinearAnnealedPolicy, EpsGreedyQPolicy
from rl.memory import SequentialMemory
#from rl.core import MultiInputProcessor
from rl.processors import MultiInputProcessor
from callbacks import *
import settings
from gym_airsim.envs.airlearningclient import *

def setup(difficulty_level='default', env_name = "AirSimEnv-v42"):
    #parser = argparse.ArgumentParser()
    #parser.add_argument('--mode', choices=['train', 'test'], default='train')
    #parser.add_argument('--env-name', type=str, default='AirSimEnv-v42')
    #parser.add_argument('--weights', type=str, default=None)
    #parser.add_argument('--difficulty-level', type=str, default="default") 
    #args = parser.parse_args()
    #args, unknown = parser.parse_known_args()
    config = tf.ConfigProto()
    config.gpu_options.per_process_gpu_memory_fraction = 0.6
    config.gpu_options.allow_growth = True
    set_session(tf.Session(config=config))

    # Get the environment and extract the number of actions.
    #msgs.algo = "DQN"
    env = gym.make(env_name)
    env.init_again(eval("settings."+difficulty_level+"_range_dic"))
    env.airgym.unreal_reset() #must rest so the env accomodate the changes
    time.sleep(5)

    np.random.seed(123)
    env.seed(123)
    nb_actions = env.action_space.n
    
    WINDOW_LENGTH = 1
    depth_shape = env.depth.shape
    vel_shape = env.velocity.shape
    dst_shape = env.position.shape

    # Keras-rl interprets an extra dimension at axis=0
    # added on to our observations, so we need to take it into account
    img_kshape = (WINDOW_LENGTH,) + depth_shape

    # Sequential model for convolutional layers applied to image
    image_model = Sequential()
    if(settings.policy=='deep'):
        image_model.add(Conv2D(128,(3, 3), strides=(3, 3), padding='valid', activation='relu', input_shape=img_kshape,
                           data_format="channels_first"))
        image_model.add(Conv2D(64, (3, 3), strides=(2, 2), padding='valid', activation='relu'))
        image_model.add(Conv2D(32, (3, 3), strides=(1, 1), padding='valid', activation='relu'))
        image_model.add(Conv2D(32, (1, 1), strides=(1, 1), padding='valid', activation='relu'))

        image_model.add(Flatten())

        # plot_model(image_model, to_file="model_conv_depth.png", show_shapes=True)
        # Input and output of the Sequential model
        image_input = Input(img_kshape)
        encoded_image = image_model(image_input)

        # Inputs and reshaped tensors for concatenate after with the image
        velocity_input = Input((1,) + vel_shape)
        distance_input = Input((1,) + dst_shape)

        vel = Reshape(vel_shape)(velocity_input)
        dst = Reshape(dst_shape)(distance_input)

        # Concatenation of image, position, distance and geofence values.
        # 3 dense layers of 256 units
        denses = concatenate([encoded_image, vel, dst])
        denses = Dense(1024, activation='relu')(denses)
        denses = Dense(1024, activation='relu')(denses)
        denses = Dense(512, activation='relu')(denses)
        denses = Dense(128, activation='relu')(denses)
        denses = Dense(64, activation='relu')(denses)

    else:
        image_model.add(Conv2D(32, (4, 4), strides=(4, 4), padding='valid', activation='relu', input_shape=img_kshape,
                               data_format="channels_first"))
        image_model.add(Conv2D(64, (3, 3), strides=(2, 2), padding='valid', activation='relu'))
        image_model.add(Conv2D(128, (2, 2), strides=(1, 1), padding='valid', activation='relu'))
        image_model.add(Conv2D(64, (1, 1), strides=(1, 1), padding='valid', activation='relu'))

        image_model.add(Flatten())

        # plot_model(image_model, to_file="model_conv_depth.png", show_shapes=True)
        # Input and output of the Sequential model
        image_input = Input(img_kshape)
        encoded_image = image_model(image_input)

        # Inputs and reshaped tensors for concatenate after with the image
        velocity_input = Input((1,) + vel_shape)
        distance_input = Input((1,) + dst_shape)

        vel = Reshape(vel_shape)(velocity_input)
        dst = Reshape(dst_shape)(distance_input)

        # Concatenation of image, position, distance and geofence values.
        # 3 dense layers of 256 units
        denses = concatenate([encoded_image, vel, dst])
        denses = Dense(256, activation='relu')(denses)
        denses = Dense(256, activation='relu')(denses)
        denses = Dense(256, activation='relu')(denses)

    # Last dense layer with nb_actions for the output
    predictions = Dense(nb_actions, kernel_initializer='zeros', activation='linear')(denses)
    model = Model(
        inputs=[image_input, velocity_input, distance_input],
        outputs=predictions
    )
    env.set_model(model)
    print(model.summary())
    # plot_model(model,to_file="model.png", show_shapes=True)
    #train = True
    #train_checkpoint = False

    # Finally, we configure and compile our agent. You can use every built-in Keras optimizer and
    # even the metrics!
    memory = SequentialMemory(limit=100000, window_length=WINDOW_LENGTH)  # reduce memmory
    processor = MultiInputProcessor(nb_inputs=3)

    # Select a policy. We use eps-greedy action selection, which means that a random action is selected
    # with probability eps. We anneal eps from 1.0 to 0.1 over the course of 1M steps. This is done so that
    # the agent initially explores the environment (high eps) and then gradually sticks to what it knows
    # (low eps). We also set a dedicated eps value that is used during testing. Note that we set it to 0.05c
    # so that the agent still performs some random actions. This ensures that the agent cannot get stuck.
    policy = LinearAnnealedPolicy(EpsGreedyQPolicy(), attr='eps', value_max=1., value_min=.1, value_test=0.0,
                                  nb_steps=100000)

    dqn = DQNAgent(model=model, processor=processor, nb_actions=nb_actions, memory=memory, nb_steps_warmup=settings.nb_steps_warmup,
                   enable_double_dqn=settings.double_dqn,
                   enable_dueling_network=False, dueling_type='avg',
                   target_model_update=1e-2, policy=policy, gamma=.99)


    
    dqn.compile(Adam(lr=0.00025), metrics=['mae'])

    # Load the check-point weights and start training from there
    return dqn,env

def train(dqn, env, train_checkpoint=False):
    msgs.mode = 'train'
    checkpoint_file = "checkpoints\\DQN\\level-3\\dqn_level_3_weights_154000.hf5"
    if train_checkpoint:
        try:
            dqn.load_weights(checkpoint_file)
            print("Loading checkpoint...\n")
        except (OSError):
            logger.warning("File not found")

    # Okay, now it's time to learn something! We visualize the training here for show, but this
    # slows down training quite a lot. You can always safely abort the training prematurely using
    # Ctrl + C.

    log_filename = 'dqn_level_3_log.json'
    weights_filename = 'dqn_level_3_{}.hf5'
    callbacks = [FileLogger(log_filename, interval=settings.logging_interval)]

    callbacks += [CheckPointLogger()]
    callbacks += [DataLogger()]
    dqn.fit(env, callbacks=callbacks, nb_steps=settings.training_steps_cap, nb_max_episode_steps=settings.nb_max_episodes_steps,  visualize=False, verbose=0, log_interval=settings.logging_interval)

    # After training is done, we save the final weights.
    dqn.save_weights(weights_filename.format(""), overwrite=True)

def test(dqn, env, file_path):
    # dqn.load_weights('checkpoints/DQN/level-3/dqn_level_3_weights_117000.hf5'.format(args.env_name))
    callbacks = [CheckPointLogger()]
    callbacks += [DataLogger()]
    msgs.mode = 'test'
    dqn.load_weights(file_path)
    msgs.weight_file_under_test = file_path
    #for i in range(0, settings.testing_nb_episodes_per_model):
    dqn.test(env, nb_episodes=settings.testing_nb_episodes_per_model, nb_max_episode_steps=settings.nb_max_episodes_steps+2, callbacks=callbacks, visualize=False)


if __name__ == '__main__':
    dqn,env= setup()
    test(dqn, env,["C:/blah2/dqn_level_3.hf5"])

