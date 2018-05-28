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

####################################################################################
print('Gazua Angry Bird!') #########################################################
####################################################################################

update_target_estimator_every = 10

#################################
##### Initialize game environment
# connect to Server
print('Initialize connection to ABServer')
s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
s.settimeout(20)
s.connect(('localhost',2004))
_,_,_ = comm.comm_configure(s, 1004)

# connect to Java
wrapper = wrap.WrapperPython('127.0.0.1')

# initialize tensorflow session
# tf.reset_default_graph() # ??
# sess = tf.Session()

# initialize feature extraction
vgg16 = VGG16(weights= 'imagenet')

# set the action sets
valid_angles = list(range(5, 86, 5)) # 5도부터 85도까지 5도씩 증가
valid_taptimes = list(range(500, 2501, 100))  # 500부터 2500까지 100씩 증가

# Create a global step variable
global_step = tf.Variable(0, name='global_step', trainable=False)

angle_estimator = q_network.DQN_Estimator(scope="angle_estimator", output_size=len(valid_angles), summaries_dir=SUMM_PATH)
angle_target_estimator = q_network.DQN_Estimator(scope="angle_target_estimator", output_size=len(valid_angles))
taptime_estimator = q_network.DQN_Estimator(scope="taptime_estimator", output_size=len(valid_taptimes), summaries_dir=SUMM_PATH)
taptime_target_estimator = q_network.DQN_Estimator(scope="taptime_target_estimator", output_size=len(valid_taptimes))
# angle_estimator, angle_target_estimator = DQN_Estimator(obs_size, sess, fe, sc_parser, "angle", valid_angles) # 수정 필요
# taptime_estimator, taptime_target_estimator = DQN_Estimator(obs_size, sess, fe, sc_parser, "taptime", valid_taptimes) # 수정 필요

# Keeps track of useful statistics
EpisodeStats = namedtuple("Stats",["episode_lengths", "episode_rewards"])


stats = EpisodeStats( # level별 episode_length랑, episode_reward를 저장해 둘 수 있는 list
        episode_lengths=[[] for i in range(21)],
        episode_rewards=[[] for i in range(21)])

# pdb.set_trace()

with tf.Session() as sess:
	sess.run(tf.global_variables_initializer())

	# pdb.set_trace()
	saver = tf.train.Saver()

	latest_checkpoint = tf.train.latest_checkpoint(checkpoint_dir) # path를 반환
	if latest_checkpoint:
		print("Loading model checkpoint {}...\n".format(latest_checkpoint))
		saver.restore(sess, latest_checkpoint)

	total_t = sess.run(tf.train.get_global_step()) # 처음에 안됐었던 이유는, global_step이란 tensor 변수를 안만들어서임

	##
	epsilon_start = 1.0
	epsilon_end = 0.1
	epsilon_decay_steps = 500000
	epsilons = np.linspace(epsilon_start, epsilon_end, epsilon_decay_steps)

	policy_angle = dqn_utils.make_epsilon_greedy_policy(
	        angle_estimator,
	        len(valid_angles))

	policy_taptime = dqn_utils.make_epsilon_greedy_policy(
	        taptime_estimator,
	        len(valid_taptimes))

	########################################
	##### Populating replay memory (size: N)
	# 원래는 랜덤하게 N번의 shot을 해서 replay_memory를 채워야 하지만...
	# 각 레벨별로, 0도부터 90도까지 쏜 데이터를 replay_memory로 함.
	# pre_train을 넣어서, 이 replay_memory로 학습을 한 weight를 가져와서 시작하는 것도 고려.
	batch_size = 6
	discount_factor = 0.99
	Transition = namedtuple("Transition", ["state", "action", "reward", "next_state", "game_state"])
	replay_memory_size = 500000
	print('Populating replay memory...')
	replay_memory = []
	# replay_memory = np.load('').tolist()

	#####################
	##### Checkpoint load
	oneshotonekill_memory, oneshotonekill_dir, oneshotonekill_path  = dqn_utils.init_oneshot_onekill(EXP_PATH, current_dir, vgg16)
	with open(os.path.join(EXP_PATH, 'oneshotonekill_memory'), 'wb') as f:
		pickle.dump(oneshotonekill_memory, f)
	with open(os.path.join(EXP_PATH, 'oneshotonekill_dir'), 'wb') as f:
		pickle.dump(oneshotonekill_dir, f)
	with open(os.path.join(EXP_PATH, 'oneshotonekill_path'), 'wb') as f:
		pickle.dump(oneshotonekill_path, f)
	for i in range(len(oneshotonekill_path)):
		oneshotonekill_logger.debug("%s [%d, %d] %d"%(oneshotonekill_path[i], oneshotonekill_memory[i][1][0], oneshotonekill_memory[i][1][1], oneshotonekill_memory[i][2]))
	print('logger done')
	while True:
		pass


	print("session 종료")
