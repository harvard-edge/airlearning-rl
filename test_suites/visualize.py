import os
import pandas as pd

os.sys.path.insert(0, os.path.abspath('..\settings_folder'))
import settings
from utils import *
import msgs
import json
import matplotlib.pyplot as plt
import numpy as np

file_name = "C:\\workspace\\airlearning-rl\\data\\backup\\DDPG\\DDPG_2x_hardware_hyper_2_17_2019\\DDPG_2x_hardware_hyper_2_17_2019\\train_episodal_logverbose.txt"

# santize_data(file_name)

parsed_data = parse_data(file_name)
action_key = 'actions_in_each_step'
steps_key = 'stepN'
roll = []
pitch = []
yaw = []
total_steps = 0

for each_episode_steps in parsed_data[steps_key]:
	total_steps = total_steps + each_episode_steps
print(total_steps)

for each_episode_action in (parsed_data[action_key]):
	for each_step_action_in_episode in each_episode_action:
		roll.append(each_step_action_in_episode[0])
		pitch.append(each_step_action_in_episode[1])
		yaw.append(each_step_action_in_episode[2])

print(max(roll))
roll = np.array(roll, dtype=np.float32)
pitch = np.array(pitch, dtype=np.float32)
yaw = np.array(yaw, dtype=np.float32)

data_frame = pd.DataFrame({"Roll": roll, "Pitch": pitch, "Yaw": yaw})
data_frame.to_csv("ddpg_2x_hardware_hyper.csv", index=False)

plt.plot(roll)
plt.show()
plt.plot(pitch)
plt.show()
plt.plot(yaw)
plt.show()
