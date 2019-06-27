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
logger = logging.getLogger(__name__)


def child_step(self, conn, airgym_obj):
        collided = False
        now = [0,0,0]
        track = 0.0
        old_depth = self.depth
        old_position = self.position
        old_velocity = self.velocity
        try:
            goal = [0,0,0]
            collided = airgym_obj.take_discrete_action(action)
            now = airgym_obj.drone_pos()
            track = airgym_obj.goal_direction(goal, now)
            self.depth = airgym_obj.getScreenDepthVis(track)
            self.position = airgym_obj.get_distance(goal)
            self.velocity = airgym_obj.drone_velocity()
            excp_occured = False
            conn.send([collided, now, track, self.depth, self.position, self.velocity, excp_occured])
            conn.close()
        except Exception as e:
            print(str(e) + "occured in child step")
            excp_occured = True

            conn.send([collided, now, track, old_depth, old_position, old_velocity, excp_occured])
            conn.close()



class AirSimEnv(gym.Env):
    #3.6.8 (self.airgym = None

    def __init__(self):
        # left depth, center depth, right depth, yaw
        if(settings.concatenate_inputs):
            STATE_POS = 3
            #STATE_VEL = 3
            STATE_DEPTH_H, STATE_DEPTH_W = 154, 256
            if(msgs.algo == "SAC"):
                self.observation_space = spaces.Box(low=-100000, high=1000000, shape=(( 1, STATE_POS + STATE_DEPTH_H * STATE_DEPTH_W)))
            else:
                self.observation_space = spaces.Box(low=-100000, high=1000000,
                                                    shape=((1, STATE_POS + STATE_DEPTH_H * STATE_DEPTH_W)))
        else:
            self.observation_space = spaces.Box(low=0, high=255, shape=(154, 256))
            #self.observation_space = spaces.Box(low=0, high=255, shape=(144, 256, 3))
        '''
        self.observation_space = spaces.Dict({"rgb": spaces.Box(low = 0, high=255, shape=(144, 256, 3)),
                                              "depth": spaces.Box(low = 0, high=255, shape=(144, 256,1)),
                                              "velocity": spaces.Box(low=-10, high=10, shape=(3,)),
                                              "position:":spaces.Box(low=np.Inf, high=np.NINF, shape=(4,))})
        '''
        self.total_step_count_for_experiment = 0 # self explanatory
        self.ease_ctr = 0  #counting how many times we ease the randomization and tightened it
        self.window_restart_ctr = 0 # counts the number of time we have restarted the window due to not meeting
                                    # the desired success_ratio
        if settings.profile:
            self.this_time = 0
            self.prev_time = 0
            self.loop_rate_list = []
            self.all_loop_rates = []
            self.take_action_list = []
            self.clct_state_list = []

        self.episodeInWindow = 0
        self.passed_all_zones = False #whether we met all the zone success
        self.weight_file_name = '' #only used for testing
        self.log_dic = {}
        #self.cur_zone_number = 0
        self.cur_zone_number_buff = 0
        self.success_count = 0
        self.success_count_within_window = 0
        self.success_ratio_within_window = 0
        self.episodeNInZone = 0 #counts the numbers of the episodes per Zone
                                #,hence gets reset upon moving on to new zone
        self.count = 0
        self.check_point = file_handling.CheckPoint()
        self.game_handler = GameHandler()
        self.OU = OU()
        self.game_config_handler = GameConfigHandler()
        if(settings.concatenate_inputs):
            self.concat_state = np.zeros((1, 1, STATE_POS + STATE_DEPTH_H * STATE_DEPTH_W), dtype=np.uint8)
        self.depth = np.zeros((154, 256), dtype=np.uint8)
        self.rgb = np.zeros((154, 256, 3), dtype=np.uint8)
        self.grey = np.zeros((144, 256), dtype=np.uint8)
        self.position = np.zeros((3,), dtype=np.float32)
        self.velocity = np.zeros((3,), dtype=np.float32)
        self.speed = 0
        self.track = 0
        self.prev_state = self.state()
        self.prev_info = {"x_pos": 0, "y_pos": 0}
        self.success = False
        self.zone = 0
        self.total_streched_ctr = 0

        self.actions_in_step = []
        self.position_in_step = []
        self.distance_in_step = []
        self.reward_in_step=[]
        self.total_reward = 0
        
        if(msgs.algo == "DDPG"):
            self.actor = ""
            self.critic= ""
        else:
            self.model = ""

        # pitch, yaw and roll are in radians ( min : -45 deg, max: 45 deg)
        if(msgs.algo == "DDPG "):
            self.action_space = spaces.Box(np.array([-0.785, -0.785, -0.785]),
                                       np.array([+0.785, +0.785, +0.785]),
                                       dtype=np.float32)  # pitch, roll, yaw_rate
        elif(msgs.algo == "PPO"):
            self.action_space = spaces.Box(np.array([-3.0, -3.0, -3.14]),
                                       np.array([+5.0, +5.0, 3.14]),
                                       dtype=np.float32)
        elif(msgs.algo == "SAC"):
            self.action_space = spaces.Box(np.array([-5.0, -5.0]),
                                       np.array([+5.0, +5.0]),
                                       dtype=np.float32)
        else:
            self.action_space = spaces.Discrete(25)

        self.goal = utils.airsimize_coordinates(self.game_config_handler.get_cur_item("End"))
        self.episodeN = 0
        self.stepN = 0

        self.allLogs = {'reward': [0]}
        self.allLogs['distance'] = [float(np.sqrt(np.power((self.goal[0]), 2) + np.power(self.goal[1], 2)))]
        self.allLogs['track'] = [-2]
        self.allLogs['action'] = [1]

        self._seed()

        self.airgym = AirLearningClient()

    def set_model(self, model):
        self.model = model
    def set_actor_critic(self, actor, critic):
        self.actor = actor
        self.critic = critic
    
    # This function was introduced (instead of the body to be merged into
    # __init__ because I need difficulty level as an argument but I can't
    # touch __init__ number of arguments
    def setConfigHandlerRange(self, range_dic):
        self.game_config_handler.set_range(*[el for el in range_dic.items()])
        self.game_config_handler.populate_zones()

    """ 
    def set_test_vars(self, weight_file_name, test_instance_number):
        self.weight_file_name = weight_file_name
        self.test_instance_number = test_instance_number
    """

    def init_again(self, range_dic): #need this cause we can't pass arguments to
                                     # the main init function easily
        self.game_config_handler.set_range(*[el for el in range_dic.items()])
        self.game_config_handler.populate_zones()
        self.sampleGameConfig()
        self.goal = utils.airsimize_coordinates(self.game_config_handler.get_cur_item("End"))
    
    def setRangeAndSampleAndReset(self, range_dic):
        self.game_config_handler.set_range(*[el for el in range_dic.items()])
        self.game_config_handler.populate_zones()
        self.sampleGameConfig()
        self.goal = utils.airsimize_coordinates(self.game_config_handler.get_cur_item("End"))
        if(os.name=="nt"):
            self.airgym.unreal_reset()
        time.sleep(5)

    def getGoal(self): #there is no setting for the goal, cause you set goal
                   #indirectory by setting End
        return self.goal

    def state(self):
        if(msgs.algo == "DDPG"):
            return self.depth
        elif(msgs.algo == "PPO"):
            return self.concat_state
        elif(msgs.algo == "SAC"):
            return self.concat_state
        elif(msgs.algo == "DQN-B"):
            return self.concat_state
        else:
            return self.depth, self.velocity, self.position
    
    def _seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def computeReward(self, now):
        # test if getPosition works here liek that
        # get exact coordiantes of the tip
        distance_now = np.sqrt(np.power((self.goal[0] - now[0]), 2) + np.power((self.goal[1] - now[1]), 2))
        distance_before = self.allLogs['distance'][-1]
        distance_correction = (distance_before - distance_now)
        r = -1

        # check if you are too close to the goal, if yes, you need to reduce the yaw and speed
        if distance_now < settings.slow_down_activation_distance:
            yaw_correction =  abs(self.track) * distance_now 
            velocity_correction = (settings.mv_fw_spd_4 - self.speed)* settings.mv_fw_dur
            r = r + distance_correction + velocity_correction
        else:
            r = r + distance_correction
        return r, distance_now
    def ddpg_add_noise_action(self, actions):
        noise_t = np.zeros([1, self.action_space.shape[0]])
        a_t = np.zeros([1, self.action_space.shape[0]])
        noise_t[0][0] = max(settings.epsilon, 0) * self.OU.function(actions[0][0], 0.0, 0.60, 0.30)  # pitch
        noise_t[0][1] = max(settings.epsilon, 0) * self.OU.function(actions[0][1], 0.0, 0.60, 0.30)  # roll
        noise_t[0][2] = max(settings.epsilon, 0) * self.OU.function(actions[0][2], 0.0, 0.60, 0.30)  # yaw_rate

        if(random.random() < 0.1):
            print("********Now we apply the brake***********")
            noise_t[0][0] = max(settings.epsilon, 0) * self.OU.function(actions[0][0], 0.1, 1.00, 0.10)
            noise_t[0][1] = max(settings.epsilon, 0) * self.OU.function(actions[0][1], 0.1, 1.00, 0.10)
            noise_t[0][2] = max(settings.epsilon, 0) * self.OU.function(actions[0][2], 0.1, 1.00, 0.10)

        a_t[0][0] = (actions[0][0] + noise_t[0][0])
        a_t[0][1] = (actions[0][1] + noise_t[0][1])
        a_t[0][2] = (actions[0][2] + noise_t[0][2])
        throttle = settings.min_throttle + random.gauss(0.1, 0.05)
        duration = settings.duration
        actions_with_noise = [a_t[0][0], a_t[0][1], throttle, a_t[0][2], duration]

        return actions_with_noise
    def ppo_call_back_emulator(self):
            if (msgs.mode == 'train'):
                append_log_file(self.episodeN, "verbose")
                append_log_file(self.episodeN, "")
                if not(msgs.success):
                    return
                weight_file_name = self.check_point.find_file_to_check_point(msgs.cur_zone_number)
                self.model.save(weight_file_name)
                with open(weight_file_name+"_meta_data", "w") as file_hndle:
                    json.dump(msgs.meta_data, file_hndle)
            elif (msgs.mode == 'test'):
                append_log_file(self.episodeN, "verbose")
                append_log_file(self.episodeN, "")
                with open(msgs.weight_file_under_test+"_test"+str(msgs.tst_inst_ctr) + "_meta_data", "w") as file_hndle:
                    json.dump(msgs.meta_data, file_hndle)
            else:
                print("this mode " + str(msgs.mode) + "is not defined. only train and test defined")
                exit(0)
    def sac_call_back_emulator(self):
            if (msgs.mode == 'train'):
                append_log_file(self.episodeN, "verbose")
                append_log_file(self.episodeN, "")
                if not(msgs.success):
                    return
                weight_file_name = self.check_point.find_file_to_check_point(msgs.cur_zone_number)
                self.model.save(weight_file_name)
                with open(weight_file_name+"_meta_data", "w") as file_hndle:
                    json.dump(msgs.meta_data, file_hndle)
            elif (msgs.mode == 'test'):
                append_log_file(self.episodeN, "verbose")
                append_log_file(self.episodeN, "")
                with open(msgs.weight_file_under_test+"_test"+str(msgs.tst_inst_ctr) + "_meta_data", "w") as file_hndle:
                    json.dump(msgs.meta_data, file_hndle)
            else:
                print("this mode " + str(msgs.mode) + "is not defined. only train and test defined")
                exit(0)

    def ddpg_call_back_emulator(self): #should be called at the end of each episode
        if(msgs.algo=="DDPG"):
            if (msgs.mode == 'train'):
                append_log_file(self.episodeN, "verbose")
                append_log_file(self.episodeN, "")
                if not(msgs.success):
                    return
                weight_file_name = self.check_point.find_file_to_check_point(msgs.cur_zone_number)
                weight_file_name = weight_file_name.replace('_critic', '')
                self.actor.save_weights(weight_file_name+"_actor", overwrite=True)
                self.critic.save_weights(weight_file_name+"_critic", overwrite=True)
                with open(weight_file_name+"_meta_data", "w") as file_hndle:
                    json.dump(msgs.meta_data, file_hndle)
            elif (msgs.mode == 'test'):
                append_log_file(self.episodeN, "verbose")
                append_log_file(self.episodeN, "")
                with open(msgs.weight_file_under_test+"_test"+str(msgs.tst_inst_ctr) + "_meta_data", "w") as file_hndle:
                    json.dump(msgs.meta_data, file_hndle)
            else:
                print("this mode " + str(msgs.mode) + "is not defined. only train and test defined")
                exit(0)
        else:
            return

    def update_success_rate(self):
        self.success_ratio_within_window = float(self.success_count_within_window/settings.update_zone_window)

    def update_zone_if_necessary(self):
        if (msgs.mode == 'train'):
            #TODO update_zone should be more general, i.e. called for other vars
            if self.success_rate_met():
                self.start_new_window()
                if (self.ease_ctr > 0):
                    self.tight_randomization()
                    return
                elif not(self.cur_zone_number_buff  == (settings.max_zone - 1)):
                    self.zone += 1
                    self.cur_zone_number_buff +=1
                else:
                    self.passed_all_zones = True
                self.update_zone("End")
                #self.success_count = e
        elif (msgs.mode == 'test'):
            if (self.episodeN % settings.testing_nb_episodes_per_zone == 0):
                if not(self.cur_zone_number_buff  == (settings.max_zone - 1)):
                    self.zone += 1
                    self.cur_zone_number_buff +=1
                self.update_zone("End")
        else:
            print("this mode " + str(msgs.mode) + "is not defined. only train and test defined")
            exit(0)

    def print_msg_of_inspiration(self):
        if (self.success_count_within_window %2 == 0):
            print("---------------:) :) :) Success, Be Happy (: (: (:------------ !!!\n")
        elif (self.success_count_within_window %3 == 0):
            print("---------------:) :) :) Success, Shake Your Butt (: (: (:------------ !!!\n")
        else:
            print("---------------:) :) :) Success, Oh Yeah! (: (: (:------------ !!!\n")

    def populate_episodal_log_dic(self):
        msgs.episodal_log_dic.clear()
        msgs.episodal_log_dic_verbose.clear()
        msgs.episodal_log_dic["cur_zone_number"] = msgs.cur_zone_number
        msgs.episodal_log_dic["success_ratio_within_window"] = self.success_ratio_within_window
        msgs.episodal_log_dic["success_count_within_window"] = self.success_count_within_window
        msgs.episodal_log_dic["success"] = msgs.success
        msgs.episodal_log_dic["stepN"] = self.stepN
        msgs.episodal_log_dic["episodeN"] = self.episodeN
        msgs.episodal_log_dic["episodeNInZone"] = self.episodeNInZone
        msgs.episodal_log_dic["episodeInWindow"] = self.episodeInWindow
        msgs.episodal_log_dic["ease_count"] = self.ease_ctr
        msgs.episodal_log_dic["total_streched_ctr"] = self.total_streched_ctr
        msgs.episodal_log_dic["restart_game_count"] = msgs.restart_game_count
        msgs.episodal_log_dic["total_reward"] = self.total_reward
        msgs.episodal_log_dic["total_step_count_for_experiment"] = self.total_step_count_for_experiment
        msgs.episodal_log_dic["goal"] = self.goal
        msgs.episodal_log_dic["distance_traveled"] = self.airgym.client.getMultirotorState().trip_stats.distance_traveled
        msgs.episodal_log_dic["energy_consumed"] = self.airgym.client.getMultirotorState().trip_stats.energy_consumed
        msgs.episodal_log_dic["flight_time"] = self.airgym.client.getMultirotorState().trip_stats.flight_time
        msgs.episodal_log_dic["time_stamp"] = self.airgym.client.getMultirotorState().timestamp
        msgs.episodal_log_dic["weight_file_under_test"] = msgs.weight_file_under_test

        #verbose
        msgs.episodal_log_dic_verbose = copy.deepcopy(msgs.episodal_log_dic)
        msgs.episodal_log_dic_verbose["reward_in_each_step"] = self.reward_in_step
        if (msgs.mode == "test"):
            msgs.episodal_log_dic_verbose["actions_in_each_step"] = self.actions_in_step
            msgs.episodal_log_dic_verbose["distance_in_each_step"] = self.distance_in_step
            msgs.episodal_log_dic_verbose["position_in_each_step"] = self.position_in_step
        elif (msgs.mode == "train"):
            return
        else:
            raise Exception(msgs.mode + "is not supported as a mode")

    def possible_to_meet_success_rate(self):
        best_success_rate_can_achieve_now =  float(((settings.update_zone_window - self.episodeInWindow) +\
                                                    self.success_count_within_window)/settings.update_zone_window)
        acceptable_success_rate =  settings.acceptable_success_rate_to_update_zone
        if (best_success_rate_can_achieve_now < acceptable_success_rate):
            return False
        else:
            return True

    def ease_randomization(self):
        for k, v in settings.environment_change_frequency.items():
            settings.environment_change_frequency[k] += settings.ease_constant
        self.ease_ctr += 1
        self.total_streched_ctr +=1

    def tight_randomization(self):
        for k, v in settings.environment_change_frequency.items():
            settings.environment_change_frequency[k] = max(
                settings.environment_change_frequency[k] - settings.ease_constant, 1)
        self.ease_ctr -=1
        self.total_streched_ctr +=1

    def start_new_window(self):
        self.window_restart_ctr = 0
        self.success_count_within_window = 0
        self.episodeInWindow = 0

    def restart_cur_window(self):
        self.window_restart_ctr +=1
        self.success_count_within_window = 0
        self.episodeInWindow = 0
        if (self.window_restart_ctr > settings.window_restart_ctr_threshold):
            self.window_restart_ctr = 0
            self.ease_randomization()

    def success_rate_met(self):
        acceptable_success_rate =  settings.acceptable_success_rate_to_update_zone
        return (self.success_ratio_within_window >= acceptable_success_rate)

    #check if possible to meet the success rate at all
    def restart_window_if_necessary(self):
        if not(self.possible_to_meet_success_rate()):
            self.restart_cur_window()

    def on_episode_end(self):
        self.update_success_rate()
        if(os.name=="nt"):
            msgs.meta_data = {**self.game_config_handler.cur_game_config.get_all_items()}
        self.populate_episodal_log_dic()

        if(msgs.algo == "DDPG"):
            self.ddpg_call_back_emulator()
        elif(msgs.algo == "PPO"):
            self.ppo_call_back_emulator()
        elif(msgs.algo == "SAC"):
            self.sac_call_back_emulator()

        self.restart_window_if_necessary()
        self.update_zone_if_necessary()

        self.actions_in_step = []
        self.distance_in_step = []
        self.reward_in_step = []
        self.position_in_step
        self.total_reward = 0

        self.allLogs['distance'] = [float(np.sqrt(np.power((self.goal[0]), 2) + np.power(self.goal[1], 2)))]



    """"
    def on_step_end(self):
        self.success_ratio_within_window = float(self.success_count_within_window/settings.update_zone_window)
        self.check_for_zone_update()
        msgs.meta_data= {**(msgs.meta_data), **self.game_config_handler.cur_game_config.get_all_items()}
    """

    def _step(self, action):
        msgs.success = False 
        msgs.meta_data = {}

        try:
            print("ENter Step"+str(self.stepN))
            self.addToLog('action', action)
            self.stepN += 1
            self.total_step_count_for_experiment +=1
            """
            parent_conn, child_conn = Pipe()
            print("cuasdfasdf") 
            p = Process(target=child_step, args=(child_conn, self.airgym))
            p.start()
            collided, now, track, self.depth, self.position, self.velocity, excp_occured = parent_conn.recv()
            p.join() 
             
            if (excp_occured):
                raise Exception("server exception happened") 
            """
            if(msgs.algo == "DDPG"):
                #self.actions_in_step.append([action[0][0], action[0][1], action[0][2]])
                self.actions_in_step.append([action[0], action[1], action[2]])
                action = self.ddpg_add_noise_action(action)
                collided = self.airgym.take_continious_action(float(action[0]), float(action[1]), float(action[2]), float(action[3]),
                                                 float(action[4]))
            elif(msgs.algo == "PPO"):
                self.actions_in_step.append([action[0], action[1], action[2]])
                collided = self.airgym.take_continious_action(action)
            elif(msgs.algo == "SAC"):
                self.actions_in_step.append([action[0], action[1]])
                collided = self.airgym.take_continious_action(action)
            else:
                if(settings.profile):
                    self.this_time = time.time()
                    if(self.stepN > 1):
                        self.loop_rate_list.append(self.this_time - self.prev_time)
                    self.prev_time = time.time()
                    take_action_start = time.time()
                collided = self.airgym.take_discrete_action(action)
                if(settings.profile):
                    take_action_end = time.time()
                self.actions_in_step.append(str(action))
                if(settings.profile):
                    self.take_action_list.append(take_action_end - take_action_start)

            if(settings.profile):
                clct_state_start = time.time()
            now = self.airgym.drone_pos()
            self.track = self.airgym.goal_direction(self.goal, now)
            #self.depth = self.airgym.getScreenDepthVis(self.track)
            self.concat_state = self.airgym.getConcatState(self.goal)
            self.rgb = self.airgym.getScreenRGB()
            self.position = self.airgym.get_distance(self.goal)
            self.velocity = self.airgym.drone_velocity()
            if(settings.profile):
                clct_state_end = time.time()
                self.clct_state_list.append(clct_state_end - clct_state_start)
            self.speed = np.sqrt(self.velocity[0]**2 + self.velocity[1]**2 +self.velocity[2]**2)
            print("Speed:"+str(self.speed))
            distance = np.sqrt(np.power((self.goal[0] - now[0]), 2) + np.power((self.goal[1] - now[1]), 2))
            
            if distance < settings.success_distance_to_goal:
                self.success_count +=1
                done = True
                self.print_msg_of_inspiration()
                self.success_count_within_window +=1
                self.success = True
                msgs.success = True
                # Todo: Add code for landing drone (Airsim API)
                reward = 1000.0
                #self.collect_data()
            elif self.stepN >= settings.nb_max_episodes_steps:
                done = True
                reward = -100.0
                self.success = False
            elif collided == True:
                done = True
                reward = -100.0
                self.success = False
            elif (now[2] < -15): # Penalize for flying away too high
                done = True
                reward = -100
                self.success = False
            else:
                reward, distance = self.computeReward(now)
                done = False
                self.success = False

            #Todo: penalize for more crazy and unstable actions

            self.allLogs['distance'] = [float(distance)]
            self.distance_in_step.append(distance)
            self.count +=1
            self.reward_in_step.append(reward)
            self.total_reward = sum(self.reward_in_step)
            self.position_in_step.append(str(now))
            info = {"x_pos":now[0], "y_pos":now[1]}

            state = self.state()
            self.prev_state = state
            self.prev_info = info

            #self.on_step_end()
            if (done):
                self.on_episode_end()

            return state, reward, done, info
        except Exception as e:
            print("------------------------- step failed ----------------  with"\
                    , e , " error")
            self.game_handler.restart_game()
            self.airgym = AirLearningClient()
            return self.prev_state, 0, True, self.prev_info

    def addToLog(self, key, value):
        if key not in self.allLogs:
            self.allLogs[key] = []
        self.allLogs[key].append(value)

    def on_episode_start(self):
        self.stepN = 0
        self.episodeN += 1
        self.episodeNInZone +=1
        #self.episodeInWindow = self.episodeNInZone % settings.update_zone_window
        self.episodeInWindow +=1
        now = self.airgym.drone_pos()
        self.track = self.airgym.goal_direction(self.goal, now)
        self.concat_state = self.airgym.getConcatState(self.goal)
        #self.depth = self.airgym.getScreenDepthVis(self.track)
        self.rgb = self.airgym.getScreenRGB()
        self.position = self.airgym.get_distance(self.goal)
        self.velocity = self.airgym.drone_velocity()
        msgs.cur_zone_number = self.cur_zone_number_buff  #which delays the update for cur_zone_number

    def _reset(self):
        try:
            if(settings.profile):
                if(self.stepN>1):
                    print("Avg loop rate:" + str(np.mean(self.loop_rate_list)))
                    self.all_loop_rates = copy.deepcopy(self.loop_rate_list)
                    print ("Action Time:" +str(np.mean(self.take_action_list)))
                    print("Collect State Time"+str(np.mean(self.clct_state_list)))
                if(self.stepN % 20 ==0):
                    print("Average Loop Rate:"+str(np.mean(self.all_loop_rates)))

            print("enter reset")
            self.randomize_env()
            print("done randomizing")
            if(os.name=="nt"):
                connection_established = self.airgym.unreal_reset()
                if not connection_established:
                    raise Exception
            print("done unreal_resetting")
            time.sleep(2)
            self.airgym.AirSim_reset()
            print("done arisim reseting")
            self.on_episode_start()
            print("done on episode start")
            state = self.state()
            self.prev_state = state
            return state

        except Exception as e:
            print("------------------------- reset failed ----------------  with"\
                    , e , " error")
            self.game_handler.restart_game()
            self.airgym = AirLearningClient()
            return self.prev_state

    def update_zone(self, *args):
        #all_keys = self.game_config_handler.game_config_range.find_all_keys()
        self.game_config_handler.update_zone(*args)
        self.episodeNInZone = 0

    # generate neew random environement if
    # it's time
    def randomize_env(self):
        vars_to_randomize = []
        for k, v in settings.environment_change_frequency.items():
            if (self.episodeN+1) %  v == 0:
                vars_to_randomize.append(k)

        if (len(vars_to_randomize) > 0):
            self.sampleGameConfig(*vars_to_randomize)
            self.goal = utils.airsimize_coordinates(self.game_config_handler.get_cur_item("End"))

    def updateJson(self, *args):
        self.game_config_handler.update_json(*args)

    def getItemCurGameConfig(self, key):
        return self.game_config_handler.get_cur_item(key)

    def setRangeGameConfig(self, *args):
        self.game_config_handler.set_range(*args)

    def getRangeGameConfig(self, key):
        return self.game_config_handler.get_range(key)

    def sampleGameConfig(self, *arg):
        self.game_config_handler.sample(*arg)
