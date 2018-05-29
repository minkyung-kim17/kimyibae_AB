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
oneshotonekill_logger = logging.getLogger("oneshotonekill_logger")
# Debug < Info < Warning < Error < Critical
oneshotonekill_logger.setLevel(logging.DEBUG)
oneshotonekill_logger.addHandler(logging.StreamHandler())
oneshotonekill_logger.addHandler(logging.FileHandler("oneshotonekill_logger.log"))

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

# while True:
# 	pass
vgg16 = VGG16(weights= 'imagenet')

# oneshotonekill_memory, oneshotonekill_dir, oneshotonekill_path  = dqn_utils.init_oneshot_onekill(EXP_PATH, current_dir, vgg16)
# oneshotonekill_angles = {}
# oneshotonekill_taptimes = {}
# with open(os.path.join(EXP_PATH, 'oneshotonekill_memory'), 'wb') as f:
# 	pickle.dump(oneshotonekill_memory, f)
# with open(os.path.join(EXP_PATH, 'oneshotonekill_dir'), 'wb') as f:
# 	pickle.dump(oneshotonekill_dir, f)
# with open(os.path.join(EXP_PATH, 'oneshotonekill_path'), 'wb') as f:
# 	pickle.dump(oneshotonekill_path, f)
# for i in range(len(oneshotonekill_path)):
# 	angle_action = oneshotonekill_memory[i][1][0]
# 	tap_action = oneshotonekill_memory[i][1][1]
# 	reward = oneshotonekill_memory[i][2]
# 	oneshotonekill_logger.debug("%s [%d, %d] %d"%(oneshotonekill_path[i], angle_action, tap_action, reward))
# 	if angle_action in oneshotonekill_angles:
# 		oneshotonekill_angles[angle_action].append(reward)
# 	else:
# 		oneshotonekill_angles[angle_action] = [reward]
# 	if tap_action in oneshotonekill_taptimes:
# 		oneshotonekill_taptimes[tap_action].append(reward)
# 	else:
# 		oneshotonekill_taptimes[angle_action] = [reward]
# with open(os.path.join(EXP_PATH, 'oneshotonekill_angles'), 'wb') as f:
# 	pickle.dump(oneshotonekill_angles, f)
# with open(os.path.join(EXP_PATH, 'oneshotonekill_taptimes'), 'wb') as f:
# 	pickle.dump(oneshotonekill_taptimes, f)
# print('logger done')
# oneshotonekill_logger.debug(oneshotonekill_angles.keys())
# oneshotonekill_logger.debug(oneshotonekill_taptimes.keys())
oneshotonekill_memory = []
oneshotonekill_dir = []
oneshotonekill_angles = {}
oneshotonekill_taptimes = {}
levels = {'level1':[], 'level2':[], 'level3':[], 'level4':[], 'level5':[], 'level6':[], 'level7':[], 'level8':[],\
		'level9':[], 'level10':[], 'level11':[], 'level12':[], 'level13':[], 'level14':[], 'level15':[], 'level16':[],	\
		'level17':[], 'level18':[], 'level19':[], 'level20':[], 'level21':[]}
with open(os.path.join(EXP_PATH, 'oneshotonekill_dir'), 'rb') as f:
	oneshotonekill_dir = pickle.load(f)
with open(os.path.join(EXP_PATH, 'oneshotonekill_memory'), 'rb') as f:
	oneshotonekill_memory = pickle.load(f)
with open(os.path.join(EXP_PATH, 'oneshotonekill_angles'), 'rb') as f:
	oneshotonekill_angles = pickle.load(f)
with open(os.path.join(EXP_PATH, 'oneshotonekill_taptimes'), 'rb') as f:
	oneshotonekill_taptimes = pickle.load(f)

levelkeys = levels.keys()
print("start to save levelkeys")
for idx in range(len(oneshotonekill_dir)):
	for key in levelkeys:
		if key in oneshotonekill_dir[idx]:
			action = oneshotonekill_memory[idx][1]
			reward = oneshotonekill_memory[idx][2]
			levels[key].append({'action':action, 'reward': reward})
			oneshotonekill_logger.debug("{} {} angle {} tap {} reward {}".format(oneshotonekill_dir[idx], key, action[0], action[1], reward) )

with open(os.path.join(EXP_PATH, 'oneshotonekill_sortlevel'), 'wb') as f:
	pickle.dump(levels, f)
