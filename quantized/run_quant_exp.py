import setup_path 
import argparse
import numpy as np
import matplotlib.pyplot as plt
import time
from gym_airsim.envs.airlearningclient import *
import os ,sys
import logging
from settings_folder import settings
from game_config_handler_class import *
from game_handler_class import *
import file_handling
import msgs
#from common import utils
import json
import copy
import gym
from gym import spaces
from gym.utils import seeding
from algorithms.continuous.ddpg.OU import OU
import random
from gym_airsim.envs.airlearningclient import *
from utils import append_log_file
import tensorflow as tf 

def run_exp():

    ## game init
    time.sleep(5)
    airl_client = AirLearningClient()
    client = airl_client.client
    client.confirmConnection()
    client.enableApiControl(True)
    client.armDisarm(True)
    client.takeoffAsync().join()    
   

    ss_state = np.zeros((20,))
    game_config_handler = GameConfigHandler()
    goal = utils.airsimize_coordinates(game_config_handler.get_cur_item("End"))

    ## tf lite interpreter init 
    interpreter = tf.lite.Interpreter(model_path="C:\\Users\\bpdui\\Documents\\airlearning_public\\airlearning\\airlearning-rl\\quantized\\converted_lite.tflite")
    interpreter.allocate_tensors()

    # Get input and output tensors.
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()
    input_shape = input_details[0]['shape']

    def step():
        ss_state[5:20] = ss_state[0:15]
        ss_state[0:5] = airl_client.get_SS_state(goal)
        
        input_data = np.array(ss_state*51,dtype=np.uint8)
        input_data = [np.clip(input_data,0,255)]
        
        interpreter.set_tensor(input_details[0]['index'], input_data)
        interpreter.invoke()
        output_data = interpreter.get_tensor(output_details[0]['index'])
        print(input_data)
        print(output_data)
        action = np.argmax(output_data)

        if action == 0:
            start, duration = airl_client.straight(settings.mv_fw_spd_3, settings.mv_fw_dur)  # move forward
        if action == 1:
            start, duration = airl_client.yaw_right(settings.yaw_rate_1_2, settings.rot_dur)  # yaw right
        if action == 2:
            start, duration = airl_client.yaw_right(settings.yaw_rate_2_4, settings.rot_dur)  # yaw left
    while(True):
        step()

    