import subprocess
import os
import time
import settings
import utils
import airsim
import msgs
import psutil
import platform

class GameHandler:
    def __init__(self):
        if not (settings.ip == '127.0.0.1'):
            return

        self.game_file = settings.game_file
        press_play_dir = os.path.dirname(os.path.realpath(__file__))
        #self.press_play_file = press_play_dir +"\\game_handling\\press_play\\Debug\\Debug"
        #self.press_play_file = press_play_dir +"\\press_play\\Debug\\press_play.exe"
        self.ue4_exe_path = settings.unreal_exec

        self.ue4_params = " -game"+" -ResX="+str(settings.game_resX)+ " -ResY="+str(settings.game_resY)+ \
                          " -WinX="+str(settings.ue4_winX)+ " -WinY="+str(settings.ue4_winY)+ " -Windowed"
        self.cmd = str('"'+ self.ue4_exe_path+ '"')+" "+str(self.game_file)+ str(self.ue4_params)
        assert(os.path.exists(self.ue4_exe_path)), "Unreal Editor executable:" + self.ue4_exe_path + "doesn't exist"
        assert(os.path.exists(self.game_file)), "game_file: " + self.game_file +  " doesn't exist"
        #assert(os.path.exists(self.press_play_file)), "press_play file: " + self.press_play_file +  " doesn't exist"


    def start_game_in_editor(self):
        if not (settings.ip == '127.0.0.1'):
            print("can not start the game in a remote machine")
            exit(0)

        unreal_pids_before_launch = utils.find_process_id_by_name("UE4Editor.exe")
        subprocess.Popen(self.cmd, shell=True)

        time.sleep(2)
        unreal_pids_after_launch = utils.find_process_id_by_name("UE4Editor.exe")
        diff_proc = []  # a list containing the difference between the previous UE4 processes
        # and the one that is about to be launched

        # wait till there is a UE4Editor process
        while not (len(diff_proc) == 1):
            time.sleep(3)
            diff_proc = (utils.list_diff(unreal_pids_after_launch, unreal_pids_before_launch))

        settings.game_proc_pid = diff_proc[0]
        #time.sleep(30)
        client = airsim.MultirotorClient(settings.ip)
        connection_established = False
        connection_ctr = 0  # counting the number of time tried to connect
        # wait till connected to the multi rotor
        time.sleep(1)
        while not (connection_established):
            try:
                #os.system(self.press_play_file)
                time.sleep(2)
                connection_established = client.confirmConnection()
            except Exception as e:
                if (connection_ctr >= settings.connection_count_threshold and msgs.restart_game_count >= settings.restart_game_from_scratch_count_threshold):
                    print("couldn't connect to the UE4Editor multirotor after multiple tries")
                    print("memory utilization:" + str(psutil.virtual_memory()[2]) + "%")
                    exit(0)
                if (connection_ctr == settings.connection_count_threshold):
                    self.restart_game()
                print("connection not established yet")
                time.sleep(5)
                connection_ctr += 1
                client = airsim.MultirotorClient(settings.ip)
                pass

        """ 
		os.system(self.game_file)
		time.sleep(30) 
		os.system(self.press_play_file)
		time.sleep(2)
		"""

    def kill_game_in_editor(self):
        process1_exist = False
        process2_exist = False
        tasklist = os.popen("tasklist").readlines()
        for el in tasklist:
            if "UE4Editor.exe" in el.split():
                process1_exist = True
            if "CrashReportClient.exe" in el.split():
                process2_exist = True
            if (process1_exist and process2_exist):
                break

        if (settings.game_proc_pid == ''):  # if proc not provided, find any Unreal and kill
            if (process1_exist):
                os.system("taskkill /f /im  " + "UE4Editor.exe")
        else:
            os.system("taskkill /f /pid  " + str(settings.game_proc_pid))
            time.sleep(2)
            settings.game_proc_pid = ''

        if (process2_exist):
            os.system("taskkill /f /im  " + "CrashReportClient.exe")

    def restart_game(self):
        if not (settings.ip == '127.0.0.1'):
            print("can not restart the game in a remote machine")
            exit(0)
        msgs.restart_game_count += 1
        self.kill_game_in_editor()  # kill in case there are any
        time.sleep(2)
        self.start_game_in_editor()
