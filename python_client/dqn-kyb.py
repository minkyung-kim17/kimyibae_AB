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
dqn_logger = logging.getLogger("dqn_logger")
# Debug < Info < Warning < Error < Critical
dqn_logger.setLevel(logging.DEBUG)
dqn_logger.addHandler(logging.StreamHandler())
dqn_logger.addHandler(logging.FileHandler("dqn_logger.log"))

# path 설정 ...
current_path = inspect.getfile(inspect.currentframe())
current_dir = os.path.dirname(os.path.abspath(current_path))
EXP_PATH=os.path.join(current_dir,"experiences")
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
_,_,_ = comm.comm_configure(s, 1003)

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
########################

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
	Transition = namedtuple("Transition", ["state", "action", "reward", "next_state", "game_state"])
	replay_memory_size = 500000
	print('Populating replay memory...')
	replay_memory = []
	# replay_memory = np.load('').tolist()

	#####################
	##### Checkpoint load
	# replay memory로 pre_train한 network를 쓴다면, 여기서 load

	# pdb.set_trace()

	####################################################################################
	print('Start Learning!') ### 게임을 하면서, 학습을 하면서, policy를 업데이트 ##########
	####################################################################################

	i_episode = 0 # 전체 episode수
	i_episodes = [0]*21 # 각 레벨별 episode수

	# loss = None # 왠지 단순히 print하려고 하는것 같음
	dqn_utils.copy_model_parameters(sess, angle_estimator, angle_target_estimator)
	dqn_utils.copy_model_parameters(sess, taptime_estimator, taptime_target_estimator)

	while True:

		game_state = comm.comm_get_state(s, silent=False)
		dqn_utils.clear_screenshot(SCR_PATH+"/")

		if game_state=='UNKNOWN':
			print ("########################################################")
			print ("Unknown state")
			pass
		elif game_state=='MAIN_MENU':
			print ("########################################################")
			print ("Main menu state")
			pass
		elif game_state=='EPISODE_MENU':
			print ("########################################################")
			print ("Episode menu state")
			pass
		elif game_state=='LEVEL_SELECTION':
			# loss = None

			print ("########################################################")
			print ("Level selection state")

			comm.comm_load_level(s, np.random.randint(1,22), silent=False)

			print ("level is loaded")

		elif game_state=='LOADING':
			print ("########################################################")
			print ("Loading state")
			pass
		elif game_state=='WON':
			print ("########################################################")
			print ("Won state")
			# resater random level
			comm.comm_load_level(s, np.random.randint(1,22), silent=False)


		elif game_state=='LOST':
			print ("########################################################")
			print ("Lost state")
			# restart random level
			comm.comm_load_level(s, np.random.randint(1,22), silent=False)


		elif game_state=='PLAYING':
			print ("########################################################")
			print ("Playing state")

			i_episode += 1

			episode_dir = "%s_%d"%(run_start_dir,i_episode)
			if not os.path.exists(episode_dir):
				os.mkdir(episode_dir)

			# 새 에피소드 시작할 때마다 checkpoint 새로 생성
			saver.save(sess, checkpoint_path)
			current_level = comm.comm_get_current_level(s)

			i_episodes[current_level-1] += 1

			print("=============== Level",current_level,"===============")

			for t in itertools.count(): # 이 에피소드가 끝날때까지

				# pdb.set_trace()
				print('\n')
				print("\rStep {} ({}) @ Episode {} (Level {})".format(
				        t, total_t, i_episode, current_level), end="\n")
				sys.stdout.flush()

				shot_dir = os.path.join(episode_dir, "level%d_shot%d_%s"%(current_level, t, time.strftime('%Y%m%d_%H%M%S')))
				if not os.path.exists(shot_dir):
					os.mkdir(shot_dir)

				screenshot_path = shot_dir+"/s_%d.png"%t
				state_raw_img = comm.comm_do_screenshot(s, screenshot_path)
				save_path = screenshot_path+"_seg.png"
				state_img = wrapper.save_seg(screenshot_path, save_path)
				state = dqn_utils.get_feature_4096(model=vgg16, img_path=save_path) # 수정: 이 함수 안에서 크기 조절하는게 좋을 듯
				# pdb.set_trace()

				print('Choose action from given Q network model')

				# Epsilon for this time step
				epsilon = epsilons[min(total_t, epsilon_decay_steps-1)]

				# Add epsilon to Tensorboard
				episode_summary = tf.Summary() # 수정: 
				episode_summary.value.add(simple_value=epsilon, tag="epsilon")
				angle_estimator.summary_writer.add_summary(episode_summary, total_t)
				taptime_estimator.summary_writer.add_summary(episode_summary, total_t)	

				# Update the target estimator
				if total_t % update_target_estimator_every == 0:
					# pass # 수정: 여기 다시 만들어야 함 오류 팡팡팡
					dqn_utils.copy_model_parameters(sess, angle_estimator, angle_target_estimator)
					dqn_utils.copy_model_parameters(sess, taptime_estimator, taptime_target_estimator)
					# print("\nCopied model parameters to target network.")

				# pdb.set_trace()

				# Take a step (현재 policy로 다음 action을 정하네)
				angle_action_probs = policy_angle(sess, state, epsilon)
				taptime_action_probs = policy_taptime(sess, state, epsilon)

				angle_action_idx = np.random.choice(np.arange(len(angle_action_probs)), p=angle_action_probs)
				taptime_action_idx = np.random.choice(np.arange(len(taptime_action_probs)), p=taptime_action_probs)

				# make shot for shooting
				slingshot_rect = wrapper.get_slingshot(screenshot_path = screenshot_path)
				ref_point = dqn_utils.get_slingshot_refpoint(slingshot = slingshot_rect)
				max_mag = slingshot_rect[3]
				angle_action = valid_angles[angle_action_idx]
				taptime_action = valid_taptimes[taptime_action_idx]
				dx = -max_mag * math.cos(math.radians(angle_action))
				dy = max_mag * math.sin(math.radians(angle_action))
				action = [angle_action, taptime_action]
				print("Choose action: angle: {}, taptime: {}".format(angle_action, taptime_action), end="\n")

				# shoot
				start_score = wrapper.get_score_in_game(screenshot_path)
				shoot_complete = comm.comm_c_shoot_fast(s,ref_point[0], ref_point[1], dx, dy, 0, taptime_action)

				# pdb.set_trace() ##

				reward, new_score, next_screenshot_path, game_state = dqn_utils.get_score_after_shot(current_dir, wrapper, s, start_score)

				print("Get reward from choosen action: reward: {}".format(reward), end="\n")

				screenshot_path = shot_dir+"/s_%d.png"%(t+1)
				shutil.copy(next_screenshot_path, screenshot_path)
				save_path = screenshot_path+"_seg.png"
				state_img = wrapper.save_seg(screenshot_path, save_path)
				with open(os.path.join(shot_dir, 'action'), 'wb') as f:
					pickle.dump(action, f)
				with open(os.path.join(shot_dir, 'reward'), 'wb') as f:
					pickle.dump(reward, f)
				next_state = dqn_utils.get_feature_4096(model=vgg16, img_path=save_path)

				# If our replay memory is full, pop the first element
				if len(replay_memory) == replay_memory_size:
				    replay_memory.pop(0)

				# Save transition to replay memory
				replay_memory.append(Transition(state, action, reward, next_state, game_state))

				# Update statistics
				if t == 0:
					stats.episode_rewards[current_level-1].append(reward)
					stats.episode_lengths[current_level-1].append(t+1)
				else:
					stats.episode_rewards[current_level-1][-1] += (reward)
					stats.episode_lengths[current_level-1][-1] = (t+1)

					# [i_episodes[current_level-1]] += reward # i_episode번째의 episode의 총 reward를 얻기 위해 계속 누적
					# stats.episode_lengths[current_level][i_episode] = t # i_episode번째의 길이를 얻기 위해 t 값으로 계속 저장

				# minibatch로 q network weight update
				batch_size = 6
				discount_factor = 0.99

				if len(replay_memory) > batch_size:
					samples = random.sample(replay_memory, batch_size)
					states_batch, action_batch, reward_batch, next_states_batch, game_state_batch = map(np.array, zip(*samples))
					# (1,1,4096) (1,2) (1,) (1,)

					done_batch = np.array([1 if (game_state =='LOST' or 'WON') else 0 for game_state in game_state_batch])

					# angle_action_batch = np.array([action_batch[i][0] for i in range(batch_size)])
					# taptime_action_batch = np.array([action_batch[i][1] for i in range(batch_size)])

					angle_action_batch_idx = np.array([valid_angles.index(action_batch[i][0]) for i in range(batch_size)])
					taptime_action_batch_idx = np.array([valid_taptimes.index(action_batch[i][1]) for i in range(batch_size)])

					# 학습에 넣을 target reward 계산

					angle_q_values_next = angle_estimator.predict(sess, next_states_batch)
					best_angle_actions = np.argmax(angle_q_values_next, axis=1)
					taptime_q_values_next = taptime_estimator.predict(sess, next_states_batch)
					best_taptime_actions = np.argmax(taptime_q_values_next, axis=1)

					angle_q_values_next_target = angle_target_estimator.predict(sess, next_states_batch)
					taptime_q_values_next_target = taptime_target_estimator.predict(sess, next_states_batch)

					angle_targets_batch = reward_batch + np.invert(done_batch).astype(np.float32) * \
			            discount_factor * angle_q_values_next_target[np.arange(batch_size), best_angle_actions]

					taptime_targets_batch = reward_batch + np.invert(done_batch).astype(np.float32) * \
			            discount_factor * taptime_q_values_next_target[np.arange(batch_size), best_taptime_actions]

					# Perform gradient descent update
					states_batch = np.array(states_batch)
					angle_loss = angle_estimator.update(sess, states_batch, angle_action_batch_idx, angle_targets_batch)
					taptime_loss = taptime_estimator.update(sess, states_batch, taptime_action_batch_idx, taptime_targets_batch)

					print('Learning done! (angle_loss:', angle_loss, 'taptime_loss:', taptime_loss, ')')

				# state 판별:
				# if game_state가 playing이 아니면 :
				if game_state!='PLAYING':
					break

				# state = next_state
				total_t += 1

			print('\nOne episode is done')

			 # Add summaries to tensorboard
			episode_summary = tf.Summary()
			# 아래에서 node_name을 여러개 만들어서, 조건문?으로 각 level별 stat을 찍으면 될것 같음!
			# pdb.set_trace()
			# for i in range(21):
				# episode_summary.value.add(simple_value=stats.episode_rewards[i][-1], node_name="episode_reward_Level_%d"%(i+1), tag="episode_reward_Level_%d"%(i+1)) 
				# episode_summary.value.add(simple_value=stats.episode_lengths[i][-1], node_name="episode_length_Level_%d"%(i+1), tag="episode_length_Level_%d"%(i+1))

			episode_summary.value.add(simple_value=stats.episode_rewards[current_level-1][-1], node_name="episode_reward_Level_%d"%(current_level), tag="episode_reward_Level_%d"%(current_level)) 
			episode_summary.value.add(simple_value=stats.episode_lengths[current_level-1][-1], node_name="episode_length_Level_%d"%(current_level), tag="episode_length_Level_%d"%(current_level))

			angle_estimator.summary_writer.add_summary(episode_summary, total_t)
			angle_estimator.summary_writer.flush()
			taptime_estimator.summary_writer.add_summary(episode_summary, total_t)
			taptime_estimator.summary_writer.flush()

			# pdb.set_trace()

		print()

	# tensorboard 실행 방법: cmd 명령으로 실행 (tensorflow가 있는 환경에서)
	# tensorboard --logdir="C:\Users\mkkim\Github\kimyibae_AB\python_client\tensorboard\summaries_angle_estimator"
	# 그 이후 나온 url 실행 (ex. http://DESKTOP-5KPGCLS:6006)

	print("session 종료")
