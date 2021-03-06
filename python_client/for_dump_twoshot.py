import os, inspect, logging, glob, time, math, itertools, sys, shutil, pickle, random
from collections import deque, namedtuple
import socket
from PIL import Image
import tensorflow as tf
import numpy as np
from tensorflow.python.keras.applications.vgg16 import VGG16

# sys.path.append('../')
# from comm import *
import comm
import wrapper_python as wrap
import q_network
import dqn_utils

import pdb

# log 설정 ...
final_twoshotonekill_logger = logging.getLogger("final_twoshotonekill_logger")
# Debug < Info < Warning < Error < Critical
final_twoshotonekill_logger.setLevel(logging.DEBUG)
final_twoshotonekill_logger.addHandler(logging.StreamHandler())
final_twoshotonekill_logger.addHandler(logging.FileHandler("final_twoshotonekill_logger.log"))

# path 설정 ...
current_path = inspect.getfile(inspect.currentframe())
current_dir = os.path.dirname(os.path.abspath(current_path))
EXP_PATH=os.path.join(current_dir,"experiences_gathering")
SCR_PATH=os.path.join(current_dir,"screenshots")
SUMM_PATH=os.path.join(current_dir, "tensorboard") # tf.summary dir

if not os.path.exists(EXP_PATH):
	os.mkdir(EXP_PATH)
if not os.path.exists(SCR_PATH):
	os.mkdir(SCR_PATH)
if not os.path.exists(SUMM_PATH):
	os.mkdir(SUMM_PATH)

checkpoint_dir = os.path.join(current_dir, "checkpoints")
checkpoint_path = os.path.join(checkpoint_dir, "model") # checkpoint file path

if not os.path.exists(checkpoint_dir):
	os.makedirs(checkpoint_dir)

# episode_prefix	= '%s/startAt_%s'%(EXP_PATH, time.strftime("%Y%m%d_%H%M"))
run_start_dir = os.path.join(EXP_PATH, "startAt_%s"%time.strftime("%Y%m%d_%H%M"))


vgg16 = VGG16(weights= 'imagenet')

twoshotonekill_memory, twoshotonekill_dir, twoshotonekill_path  = dqn_utils.init_twoshot_onekill(EXP_PATH, current_dir, vgg16)
twoshotonekill_angles = {}
twoshotonekill_taptimes = {}
with open(os.path.join(EXP_PATH, 'twoshotonekill_memory'), 'wb') as f:
	pickle.dump(twoshotonekill_memory, f)
with open(os.path.join(EXP_PATH, 'twoshotonekill_dir'), 'wb') as f:
	pickle.dump(twoshotonekill_dir, f)
with open(os.path.join(EXP_PATH, 'twoshotonekill_path'), 'wb') as f:
	pickle.dump(twoshotonekill_path, f)
for i in range(len(twoshotonekill_path)):
	angle_action = twoshotonekill_memory[i][1][0]
	tap_action = twoshotonekill_memory[i][1][1]
	reward = twoshotonekill_memory[i][2]
	final_twoshotonekill_logger.debug("%s [%d, %d] %d"%(twoshotonekill_path[i], angle_action, tap_action, reward))
	if angle_action in twoshotonekill_angles:
		twoshotonekill_angles[angle_action].append(reward)
	else:
		twoshotonekill_angles[angle_action] = [reward]
	if tap_action in twoshotonekill_taptimes:
		twoshotonekill_taptimes[tap_action].append(reward)
	else:
		twoshotonekill_taptimes[angle_action] = [reward]
with open(os.path.join(EXP_PATH, 'twoshotonekill_angles'), 'wb') as f:
	pickle.dump(twoshotonekill_angles, f)
with open(os.path.join(EXP_PATH, 'twoshotonekill_taptimes'), 'wb') as f:
	pickle.dump(twoshotonekill_taptimes, f)
print('logger done')
final_twoshotonekill_logger.debug(twoshotonekill_angles.keys())
final_twoshotonekill_logger.debug(twoshotonekill_taptimes.keys())


# twoshotonekill_memory = []
# twoshotonekill_dir = []
# twoshotonekill_angles = {}
# twoshotonekill_taptimes = {}
levels = {'level1_':[], 'level2_':[], 'level3_':[], 'level4_':[], 'level5_':[], 'level6_':[], 'level7_':[], 'level8_':[],\
		'level9_':[], 'level10_':[], 'level11_':[], 'level12_':[], 'level13_':[], 'level14_':[], 'level15_':[], 'level16_':[],	\
		'level17_':[], 'level18_':[], 'level19_':[], 'level20_':[], 'level21_':[]}
# with open(os.path.join(EXP_PATH, 'twoshotonekill_dir'), 'rb') as f:
# 	twoshotonekill_dir = pickle.load(f)
# with open(os.path.join(EXP_PATH, 'twoshotonekill_memory'), 'rb') as f:
# 	twoshotonekill_memory = pickle.load(f)
# with open(os.path.join(EXP_PATH, 'twoshotonekill_angles'), 'rb') as f:
# 	twoshotonekill_angles = pickle.load(f)
# with open(os.path.join(EXP_PATH, 'twoshotonekill_taptimes'), 'rb') as f:
# 	twoshotonekill_taptimes = pickle.load(f)

levelkeys = levels.keys()
print("start to save levelkeys")
for idx in range(len(twoshotonekill_dir)):
	for key in levelkeys:
		if key in twoshotonekill_dir[idx]:
			action = twoshotonekill_memory[idx][1]
			reward = twoshotonekill_memory[idx][2]
			levels[key].append({'action':action, 'reward': reward})
			final_twoshotonekill_logger.debug("{} {} angle {} tap {} reward {}".format(twoshotonekill_dir[idx], key, action[0], action[1], reward) )

with open(os.path.join(EXP_PATH, 'twoshotonekill_sortlevel'), 'wb') as f:
	pickle.dump(levels, f)
