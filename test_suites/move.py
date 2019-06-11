# simply moving the drone
#
import numpy as np
import gym
import time
import random
import airsim



def test():
	client = airsim.MultirotorClient(ip="127.0.0.1")
	client.confirmConnection()
	client.enableApiControl(True)
	client.armDisarm(True)

	client.moveByVelocityAsync(0, 0, -1, 3).join()
	try:
		for each in range(0,1000):
			pitch = random.uniform(-0.25,0.25)
			x = random.uniform(-3.0,5.0)
			y = random.uniform(-3.0,5.0)
			yaw = random.uniform(0,360)
			#print("step:"+str(each)+"["+str(pitch)+","+str(roll)+","+str(yaw)+"]")
			#client.moveByAngleZAsync(pitch, roll, -6, yaw, 0.3).join()
			#client.moveByVelocityZAsync(vx, vy, -6, 0.25, 1, yaw_mode=airsim.YawMode(False,0)).join()
			client.moveToPositionAsync(x, y, -6, 1, yaw_mode=airsim.YawMode(False, 0)).join()
		client.reset()
	except KeyboardInterrupt:
		client.reset()


	time.sleep(1)


test()
