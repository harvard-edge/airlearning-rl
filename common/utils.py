import random
import os
import subprocess
import numpy as np
import psutil
import json
import settings
import msgs
import matplotlib.pyplot as plt
import shutil
import pandas as pd
import subprocess


def parse_data(file_name):
    if (file_name == ''):
        file_hndl = open(os.path.join(settings.proj_root_path, "data", msgs.algo, msgs.mode + "_episodal_log.txt"), "r")
    else:
        file_hndl = open(file_name, "r")
    data = json.load(file_hndl)
    data_clusterd_based_on_key = {}
    for episode, episode_data in data.items():
        for key, value in episode_data.items():
            if not (key in data_clusterd_based_on_key.keys()):
                data_clusterd_based_on_key[key] = [value]
            else:
                data_clusterd_based_on_key[key].append(value)

    return data_clusterd_based_on_key


def santize_data(file):
    tmp = "tmp.txt"
    f1 = open(file, 'r')
    f2 = open(tmp, 'w')
    for line in f1:
        for substring in (("True", '"' + str(True) + '"'), ("False", '"' + str(False) + '"')):
            line = line.replace(*substring)
        f2.write(line)
    f1.close()
    f2.close()
    shutil.move(tmp, file)


def plot_data(file, data_to_inquire, mode="separate"):
    # santize_data(file)
    data = parse_data(file)
    for el in data_to_inquire:
        slice_interval = 1
        x = data[el[0]]
        y = data[el[1]]
        length = np.shape(x)[0]
        print(y)
        if(el[1] == 'success'):
            for i in range(length):
                if(y[i] =='True'):
                    y[i] = 1
                else:
                    y[i] = 0
        
        pass_arr = data['success']
        if(el[1] == 'distance_traveled'):
            x_loc = []
            y_loc = []
            for i in range(length):
                if(pass_arr[i] == 1 or pass_arr[i]=='True' ):
                    x_loc = np.append(x_loc,x[i])
                    y_loc = np.append(y_loc,y[i])
            x = x_loc
            y = y_loc
        
                
        print(y)

        x = x[0:(length-1):slice_interval]
        y = y[0:(length-1):slice_interval]
        def movingaverage (values, window):
            weights = np.repeat(1.0, window)/window
            sma = np.convolve(values, weights, 'valid')
            return sma

        y = movingaverage(y,100)
        x = x[len(x)-len(y):]
        ## slicing
        # x = x[0:(length-1):slice_interval]
        # y = y[0:(length-1):slice_interval]
        plt.plot(range(len(y)), y)
        plt.xlabel(el[0])
        plt.ylabel(el[1])
        assert (el[0] in data.keys())
        plt.legend()
        plt.draw()
        plt.figure()
    if (mode == "separate"):
        plt.show()
    # plt.draw()
    # plt.pause(.001)


def generate_csv(file):
    data = parse_data(file)
    data_frame = pd.DataFrame(data)
    data_frame.to_csv(file.replace("txt", "csv"), index=False)


def append_log_file(episodeN, log_mode="verbose"):
    with open(os.path.join(settings.proj_root_path, "data", msgs.algo, msgs.mode + "_episodal_log" + log_mode + ".txt"),
              "a+") as f:
        if (episodeN == 0):
            f.write('{\n')
        if (log_mode == "verbose"):
            f.write(
                '"' + str(episodeN) + '"' + ":" + str(msgs.episodal_log_dic_verbose).replace("\'", "\"").replace("True",
                                                                                                                 "\"True\"").replace(
                    "False", "\"False\"") + ",\n")
        # replace("\'", "\"") +",\n")
        else:
            f.write('"' + str(episodeN) + '"' + ":" + str(msgs.episodal_log_dic).replace("\'", "\"").replace("True",
                                                                                                             "\"True\"").replace(
                "False", "\"False\"") + ",\n")
        f.close()


def show_data_in_time():
    with open(settings.proj_root_path, "data", "DQN", "train_episodal_log.txt", "r") as f:
        data = json.load(f)
        print(data)


def airsimize_coordinates(pose):
    return [pose[0], pose[1], -pose[2]]


def list_diff(li1, li2):
    li_dif = [i for i in li1 + li2 if i not in li1 or i not in li2]
    return li_dif


def find_process_id_by_name(processName):
    '''
	Get a list of all the PIDs of a all the running process whose name contains
	the given string processName
	'''

    listOfProcessObjects = []

    # Iterate over the all the running process
    for proc in psutil.process_iter():
        try:
            pinfo = proc.as_dict(attrs=['pid', 'name'])
            # Check if process name contains the given name string.
            if processName.lower() in pinfo['name'].lower():
                listOfProcessObjects.append(pinfo['pid'])
        except Exception as e:
            pass

    return listOfProcessObjects;


def reset_msg_logs():
    # TODO this should be one by one cause then everytime we touch msgs we need to touch this as well
    msgs.success = False
    msgs.meta_data = {}
    msgs.episodal_log_dic = {}
    msgs.episodal_log_dic_verbose = {}
    msgs.cur_zone_number = 0
    msgs.weight_file_under_test = ''
    msgs.tst_inst_ctr = 0
    msgs.mode = ''
    msgs.restart_game_count = 0


def get_random_end_point(arena_size, split_index, total_num_of_splits):
    # distance from the walls
    wall_halo = floor_halo = roof_halo = 1
    goal_halo = settings.slow_down_activation_distance + 1

    sampling_quanta = .5  # sampling increment

    # how big the split is (in only one direction, i.e pos or neg)
    idx0_quanta = float((arena_size[0] - 2 * goal_halo - 2 * wall_halo)) / (2 * total_num_of_splits)
    idx1_quanta = float((arena_size[1] - 2 * goal_halo - 2 * wall_halo)) / (2 * total_num_of_splits)
    idx2_quanta = float((arena_size[2])) / (2 * total_num_of_splits)

    idx0_up_pos_bndry = (split_index + 1) * idx0_quanta
    idx1_up_pos_bndry = (split_index + 1) * idx1_quanta
    idx2_up_pos_bndry = (split_index + 1) * idx2_quanta

    if (settings.end_randomization_mode == "inclusive"):
        idx0_low_pos_bndry = 0
        idx1_low_pos_bndry = 0
        idx2_low_pos_bndry = 0
    else:
        idx0_low_pos_bndry = (split_index) * idx0_quanta
        idx1_low_pos_bndry = (split_index) * idx1_quanta
        idx2_low_pos_bndry = (split_index) * idx2_quanta

    assert (
            idx0_up_pos_bndry - idx0_low_pos_bndry > sampling_quanta), "End doesn't fit within the zone, expand the arena size or reduce number of zones"
    assert (
            idx1_up_pos_bndry - idx1_low_pos_bndry > sampling_quanta), "End doesn't fit within the zone, expand the arena size or reduce number of zones"
    assert (
            idx2_up_pos_bndry - idx2_low_pos_bndry > sampling_quanta), "End doesn't fit within the zone, expand the arena size or reduce number of zones"

    rnd_pos_idx0 = random.choice(list(np.arange(
        idx0_low_pos_bndry + goal_halo, idx0_up_pos_bndry + goal_halo, sampling_quanta)))
    rnd_pos_idx1 = random.choice(list(np.arange(
        idx1_low_pos_bndry + goal_halo, idx1_up_pos_bndry + goal_halo, sampling_quanta)))
    rnd_pos_idx2 = random.choice(list(np.arange(
        idx2_low_pos_bndry + goal_halo, idx2_up_pos_bndry + goal_halo, sampling_quanta)))

    rnd_neg_idx0 = random.choice(list(np.arange(
        -idx0_up_pos_bndry - goal_halo, -idx0_low_pos_bndry - goal_halo, sampling_quanta)))

    rnd_neg_idx1 = random.choice(list(np.arange(
        -idx1_up_pos_bndry - goal_halo, -idx1_low_pos_bndry - goal_halo, sampling_quanta)))

    rnd_neg_idx2 = random.choice(list(np.arange(
        -idx2_up_pos_bndry - goal_halo, -idx2_low_pos_bndry - goal_halo, sampling_quanta)))

    rnd_idx0 = random.choice([rnd_neg_idx0, rnd_pos_idx0])
    rnd_idx1 = random.choice([rnd_neg_idx1, rnd_pos_idx1])
    rnd_idx2 = random.choice([rnd_neg_idx2, rnd_pos_idx2])

    """
	idx0_up_pos_bndry = int(arena_size[0]/2)
	idx1__up_pos_bndry = int(arena_size[1]/2)
	idx2__up_pos_bndry = int(arena_size[2])

	idx0_neg_bndry = int(-1*arena_size[0]/2)
	idx1_neg_bndry = int(-1*arena_size[1]/2)
	idx2_neg_bndry = 0
	
	rnd_idx0 = random.choice(list(range(
		idx0_neg_bndry + end_halo, idx0_pos_bndry - end_halo)))
	
	rnd_idx1 = random.choice(list(range(
		idx1_neg_bndry + end_halo, idx1_pos_bndry - end_halo)))
	 
	rnd_idx2 = random.choice(list(range(
	   0 + floor_halo, idx2_pos_bndry - roof_halo)))
	"""
    grounded_idx2 = 0  # to force the end on the ground, otherwise, it'll
    # be fallen (due to gravity) but then distance
    # calculation to goal becomes faulty

    if (rnd_idx0 == rnd_idx1 == 0):  # to avoid being on the start position
        rnd_idx0 = idx0_pos_bndry - end_halo

    return [rnd_idx0, rnd_idx1, grounded_idx2]


# return [rnd_idx0, rnd_idx1, rnd_idx2]


def get_lib_addr():
    import rl.agents.dqn as blah
    # from shutil import copy2
    airsim_dir = os.path.dirname(blah.__file__)
    file_path = os.path.realpath(__file__)
    dir_path = os.path.dirname(file_path)
    print(airsim_dir)


def copy_json_to_server(filename):
    try:
        exit_code = os.system("copy " + filename + " " + settings.unreal_host_shared_dir)
        if not(exit_code == 0):
            raise Exception("couldn't copy the json file to the unreal_host_shared_dir")
    except Exception as e:
        print(str(e))
        exit(1)
