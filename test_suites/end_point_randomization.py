import os

os.sys.path.insert(0, os.path.abspath('..\settings_folder'))
import settings
from utils import get_random_end_point


def test():
	arena_size = [60, 60, 20]
	total_num_of_splits = 3
	print("arena_size" + str(arena_size))
	print("total_num_of_splits" + str(total_num_of_splits))

	for split_index in range(0, total_num_of_splits):

		print("-------- split_index" + str(split_index))
		for i in range(0, 5):
			res = get_random_end_point(arena_size, split_index, total_num_of_splits)
			print("smapled point" + str(res))


test()
