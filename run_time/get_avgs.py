import os
os.sys.path.insert(0, os.path.abspath('..\settings_folder'))
import settings
from utils import *
import msgs
import pandas as pd
import numpy
import matplotlib.pyplot  as plt
def filter(data, key, value):
    result = {}
    res_index = []
    ctr = 0
    for el_val in data[key]:
        if el_val == value:
            res_index.append(ctr)
        ctr +=1

    for key in data:
        result[key] = []

    for key in data:
        for idx in res_index:
            result[key].append(data[key][idx])

    return result

def main():
    # set the following params
    msgs.algo = "DQN"
    msgs.mode = "train"

    # parse data
    input_file = "desktop_static_obstacles_final.txt"
    data_file = os.path.join(settings.proj_root_path, "data", "DQN", input_file)
    data = parse_data(data_file)

    # parse the success first
    result = filter(data, 'success', 'True')
    #result = data
    result_per_zone = {}
    for zone_idx in range (0,4):
        result_per_zone[zone_idx] = filter(result, 'cur_zone_number', zone_idx)

    so_far_success_count = 0
    for zone_idx in result_per_zone.keys():
        print("zone:" + str(zone_idx) + " success_count:" +\
              str(result_per_zone[zone_idx]["success_count_within_window"][-1] - so_far_success_count), end = "||")
        so_far_success_count = result_per_zone[zone_idx]["success_count_within_window"][-1]
        print("zone:" + str(zone_idx) + " distance_traveled:" + str(numpy.mean(result_per_zone[zone_idx]["distance_traveled"])), end = "||")
        print("zone:" + str(zone_idx) + " energy_consumed:" + str(numpy.mean(result_per_zone[zone_idx]["energy_consumed"])), end = "||")
        print("zone:" + str(zone_idx) + " flight_time:" + str(numpy.mean(result_per_zone[zone_idx]["flight_time"])), end = "||"),
        print("zone:" + str(zone_idx) + " stepN:" + str(numpy.mean(result_per_zone[zone_idx]["stepN"])))
        print("zone:" + str(zone_idx) + " std:" + str(numpy.std(result_per_zone[zone_idx]["stepN"])))
        print(result_per_zone[zone_idx]["goal"])

        print("zone:" + str(zone_idx) + " end_mag_mean:" + str(numpy.mean(list(map(lambda x: numpy.linalg.norm(x), result_per_zone[zone_idx]["goal"])))))
        #print("zone:" + str(zone_idx) + " end_mean:" + str(numpy.mean(result_per_zone[zone_idx]["goal"])))
        plt.hist(result_per_zone[zone_idx]["stepN"], range(0,200, 10))
        plt.title("zone:" + str(zone_idx) + " data")
        #plt.figure()
        plt.savefig(str(zone_idx) + input_file.replace("txt", "png"))


    # dumping into a csv
    """
    data_frame = pd.DataFrame(data)
    data_frame.to_csv(data_file.replace("txt", "csv"), index=False)

    data_file = os.path.join(settings.proj_root_path, "data", "DQN", "filterd.txt")
    data_frame = pd.DataFrame(result_zone2)
    data_frame.to_csv(data_file.replace("txt", "csv"), index=False)
    """

if __name__ == "__main__":
    main()


