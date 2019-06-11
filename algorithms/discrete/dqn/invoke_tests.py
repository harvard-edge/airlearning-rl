# import json_manipulation_test
# import reset_test
import os
import argparse

# grab the test name
parser = argparse.ArgumentParser()
parser.add_argument('--test_name', default='json_manipulation_test')
# parser.add_argument('--difficulty-level', type=str, default="easy")
# parser.add_argument('--mode', type=str, default="train")

# build its path
args, unknown = parser.parse_known_args()

test_file_dir = os.path.join(os.path.dirname(os.path.realpath(__file__)), "test_suites")
test_file_path = os.path.join(test_file_dir, args.test_name + ".py")
assert (os.path.isfile(test_file_path)), args.test_name + " doesn't exist"

# run the test
print("===============================>running " + str(args.test_name) + " test")
arg_dic = {}
for arg in unknown:
	arg_dic[arg[2:].split("=")[0]] = arg[2:].split("=")[1]
module = __import__(str(args.test_name))

module.test(**arg_dic)
