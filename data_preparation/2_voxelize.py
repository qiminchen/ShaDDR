import numpy as np
import cv2
import os
import argparse
import time

parser = argparse.ArgumentParser()
parser.add_argument("class_id", type=str, help="shapenet category id")
parser.add_argument("share_id", type=int, help="id of the share [0]")
parser.add_argument("share_total", type=int, help="total num of shares [1]")
FLAGS = parser.parse_args()

class_id = FLAGS.class_id
target_dir = "./" + class_id + "/"
if not os.path.exists(target_dir):
    print("ERROR: this dir does not exist: " + target_dir)
    exit(-1)

share_id = FLAGS.share_id
share_total = FLAGS.share_total

obj_names = os.listdir(target_dir)
obj_names = sorted(obj_names)

obj_names_ = []
for i in range(len(obj_names)):
    if i % share_total == share_id:
        obj_names_.append(obj_names[i])
obj_names = obj_names_


for i in range(len(obj_names)):
    this_name = target_dir + obj_names[i] + "/model_normalized.obj"
    print(i,len(obj_names),this_name)

    # get binvox
    command = "./binvox -bb -0.5 -0.5 -0.5 0.5 0.5 0.5 -d 512 -e " + this_name
    os.system(command)

    # rename
    command = "mv "+ target_dir + obj_names[i] + "/model_normalized.binvox" + " " + target_dir + obj_names[i] + "/model.binvox"
    os.system(command)
