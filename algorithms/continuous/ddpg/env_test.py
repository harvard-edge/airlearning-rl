import airsim

import numpy as np
import os
import tempfile
import pprint
import time
from random import *

# connect to the AirSim simulator
client = airsim.MultirotorClient()
client.reset()
client.confirmConnection()
client.enableApiControl(True)
client.armDisarm(True)

# state = client.getMultirotorState()
# s = pprint.pformat(state)
# print("state: %s" % s)

client.takeoffAsync().join()
time.sleep(4)
client.moveByVelocityAsync(0, 0, -2, 5, drivetrain=0, vehicle_name='')
time.sleep(6)
flag = False
try:
	while (flag):
		pitch = uniform(-0.78, 0.78)
		roll = uniform(-0.78, 0.78)
		throttle = uniform(0.5, 10)
		yaw_rate = uniform(-0.78, 0.78)
		duration = uniform(0, 5)

		client.moveByAngleThrottleAsync(pitch, roll, throttle, yaw_rate, duration, vehicle_name='')
		print(client.getOrientation())
		time.sleep(5)
except KeyboardInterrupt:
	pass
client.moveToPositionAsync(30.5, -35, -10, 5)
time.sleep(5)
state = client.getMultirotorState()
s = pprint.pformat(state)
print("############################################################################################")
print("state: %s" % s)

# time.sleep(5)

print("moved!")
