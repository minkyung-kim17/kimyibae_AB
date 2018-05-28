import os, inspect, logging, glob, time, math, itertools, sys, shutil, plotting, pickle, random
from collections import deque, namedtuple
import socket
from PIL import Image
import tensorflow as tf
import numpy as np
# from tensorflow.python.keras.applications.vgg16 import VGG16

# sys.path.append('../')
# from comm import *
import comm
import wrapper_python as wrap
import q_network
import dqn_utils

import pdb

# log 설정 ...
gathering_logger = logging.getLogger("gathering_logger")
# Debug < Info < Warning < Error < Critical
gathering_logger.setLevel(logging.DEBUG)
gathering_logger.addHandler(logging.StreamHandler())
gathering_logger.addHandler(logging.FileHandler("gathering_logger.log"))

# path 설정 ...
current_path = inspect.getfile(inspect.currentframe())
current_dir = os.path.dirname(os.path.abspath(current_path))
EXP_PATH=os.path.join(current_dir,"experiences_gathering")
SCR_PATH=os.path.join(current_dir,"screenshots")

if not os.path.exists(EXP_PATH):
			os.mkdir(EXP_PATH)
# if not os.path.exists(SCR_PATH):
			# os.mkdir(SCR_PATH)

checkpoint_dir = os.path.join(current_dir, "checkpoints_gathering")
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

update_target_estimator_every = 50

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

# initialize feature extraction
# vgg16 = VGG16(weights= 'imagenet')

# set the action sets
valid_angles = list(range(5, 86, 1)) # 5도부터 85도까지 5도씩 증가
# valid_taptimes = np.array((range(500, 2501, 100)))/1000).tolist() # 500부터 2500까지 100씩 증가
valid_taptimes = list(range(500, 2501, 100)) # 500부터 2500까지 100씩 증가

# angle_estimator = q_network.DQN_Estimator(scope="angle_estimator", output_size=len(valid_angles), summaries_dir=None)
# angle_target_estimator = q_network.DQN_Estimator(scope="angle_target_estimator", output_size=len(valid_angles))
# taptime_estimator = q_network.DQN_Estimator(scope="taptime_estimator", output_size=len(valid_taptimes), summaries_dir=None)
# taptime_target_estimator = q_network.DQN_Estimator(scope="taptime_target_estimator", output_size=len(valid_taptimes))
# angle_estimator, angle_target_estimator = DQN_Estimator(obs_size, sess, fe, sc_parser, "angle", valid_angles) # 수정 필요
# taptime_estimator, taptime_target_estimator = DQN_Estimator(obs_size, sess, fe, sc_parser, "taptime", valid_taptimes) # 수정 필요

# level별 episode_length랑, episode_reward를 저장해 둘 수 있는 matrix??... plotting.py (MIT코드에서 가져옴)를 이용할 수 있음
# stats = plotting.EpisodeStats(
#         episode_lengths=np.zeros(num_episodes),
#         episode_rewards=np.zeros(num_episodes))
########################

# sess.run(tf.global_variables_initializer())
init = tf.global_variables_initializer()
# initialize tensorflow session
# tf.reset_default_graph() # ??
with tf.Session() as sess:
	sess.run(init)

	# saver = tf.train.Saver()

	latest_checkpoint = tf.train.latest_checkpoint(checkpoint_dir)
	if latest_checkpoint:
		print("Loading model checkpoint {}...\n".format(latest_checkpoint))
		# saver.restore(sess, latest_checkpoint)

	try:
		total_t = sess.run(tf.train.get_global_step())
	except:
		total_t = 0

	epsilon_start = 1.0
	epsilon_end = 0.1
	epsilon_decay_steps = 500000
	epsilons = np.linspace(epsilon_start, epsilon_end, epsilon_decay_steps)

	# policy_angle = dqn_utils.make_epsilon_greedy_policy(
	#         angle_estimator,
	#         len(valid_angles))
	#
	# policy_taptime = dqn_utils.make_epsilon_greedy_policy(
	#         taptime_estimator,
	#         len(valid_taptimes))

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

	load_level = 13
	angle_action_idx = 57
	angle_action = valid_angles[angle_action_idx]
	i_episode=0
	loss = None # 수정: 여기서 하는게 맞나... Level_selection안에 넣어놨었는데, 여기를 들어가지 않고 실행되는 경우도 있었음
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
			comm.comm_load_level(s, load_level, silent=False)
			print ("level is loaded")

		elif game_state=='LOADING':
			print ("########################################################")
			print ("Loading state")
			pass
		elif game_state=='WON':
			print ("########################################################")
			print ("Won state")
			# resater random level
			if angle_action_idx != len(valid_angles)-1:
				angle_action_idx += 1
				comm.comm_restart_level(s)
			elif angle_action_idx == len(valid_angles)-1:
				angle_action_idx = 0
				if comm.comm_get_current_level(s)<21:
					load_level +=1
					comm.comm_load_level(s, load_level)
				else:
					break


		elif game_state=='LOST':
			print ("########################################################")
			print ("Lost state")
			# restart random level
			if angle_action_idx != len(valid_angles)-1:
				angle_action_idx += 1
				comm.comm_restart_level(s)
			elif angle_action_idx == len(valid_angles)-1:
				angle_action_idx = 0
				if comm.comm_get_current_level(s)<21:
					load_level +=1
					comm.comm_load_level(s, load_level)
				else:
					break


		elif game_state=='PLAYING':
			print ("########################################################")
			print ("Playing state")

			i_episode += 1

			episode_dir = "%s_%d"%(run_start_dir,i_episode)
			if not os.path.exists(episode_dir):
				os.mkdir(episode_dir)

			# saver.save(tf.get_default_session(), checkpoint_path) # tf.get_default_session()이 none이 됨...
			# saver.save(sess, checkpoint_path)
			current_level = comm.comm_get_current_level(s)

			print("=============== Level",current_level,"===============")

			for t in itertools.count(): # 이 에피소드가 끝날때까지

				shot_dir = os.path.join(episode_dir, "level%d_shot%d_%s"%(current_level, t, time.strftime('%Y%m%d_%H%M%S')))
				if not os.path.exists(shot_dir):
					os.mkdir(shot_dir)
				while True:
					is_zoomed_out = comm.comm_fully_zoomout(s)
					if is_zoomed_out==1:
						break
				screenshot_path = shot_dir+"/s_%d.png"%t
				state_raw_img = comm.comm_do_screenshot(s, screenshot_path)
				save_path = screenshot_path+"_seg.png"
				state_img = wrapper.save_seg(screenshot_path, save_path)
				# state = dqn_utils.get_feature_4096(model=vgg16, img_path=save_path) # 수정: 이 함수 안에서 크기 조절하는게 좋을 듯
				# pdb.set_trace()

				print('Choose action from given Q network model')

				# # Epsilon for this time step
				# epsilon = epsilons[min(total_t, epsilon_decay_steps-1)]
				#
				# # Add epsilon to Tensorboard
				# # episode_summary = tf.Summary()
				# # episode_summary.value.add(simple_value=epsilon, tag="epsilon")
				# # q_estimator.summary_writer.add_summary(episode_summary, total_t)
				#
				# # Maybe update the target estimator
				# if total_t % update_target_estimator_every == 0:
				# 	pass # 수정: 여기 다시 만들어야 함 오류 팡팡팡
				# 	# dqn_utils.copy_model_parameters(sess, angle_estimator, angle_target_estimator)
				# 	# dqn_utils.copy_model_parameters(sess, taptime_estimator, taptime_target_estimator)
				# 	# print("\nCopied model parameters to target network.")

				# pdb.set_trace()
				# Print out which step we're on, useful for debugging.
				print("\rStep {} ({}) @ Episode {}, loss: {} ".format(
				        t, total_t, i_episode, loss), end="")
				sys.stdout.flush()

				# Take a step (현재 policy로 다음 action을 정하네)
				# angle_action_probs = policy_angle(sess, state, epsilon)
				# taptime_action_probs = policy_taptime(sess, state, epsilon)

				# angle_action_idx = np.random.choice(np.arange(len(angle_action_probs)), p=angle_action_probs)
				# taptime_action_idx = np.random.choice(np.arange(len(valid)))

				# make shot for shooting
				slingshot_rect = None
				while(slingshot_rect == None or slingshot_rect[0]==-1 or slingshot_rect[1]==-1):
					is_zoomed_out = 0;
					while is_zoomed_out !=1:
						is_zoomed_out = comm.comm_fully_zoomout(s)
					slingshot_rect = wrapper.get_slingshot(screenshot_path = screenshot_path)
				ref_point = dqn_utils.get_slingshot_refpoint(slingshot = slingshot_rect)
				max_mag = slingshot_rect[3]
				# angle_action = valid_angles[angle_action_idx]
				# taptime_action = valid_taptimes[taptime_action_idx]
				if t==0:
					# angle_action = 30
					angle_action = valid_angles[angle_action_idx]
				else:
					# 0 이상의 index를 추출, 0보다 작으면 0
					temp_index = max(int(np.random.normal(angle_action, 5)),0)
					angle_action = valid_angles[min(temp_index, len(valid_angles)-1)]
					# temp_index = np.random.choice(np.arange(len(angle_action_probs)), p=angle_action_probs)
					# angle_action = valid_angles[temp_index]
				taptime_action = np.random.choice(valid_taptimes)
				dx = -max_mag * math.cos(math.radians(angle_action))
				dy = max_mag * math.sin(math.radians(angle_action))
				action = [angle_action, taptime_action]

				# shoot
				print("angle_action, taptime_action ", angle_action, taptime_action)
				start_score = wrapper.get_score_in_game(screenshot_path)
				shoot_complete = comm.comm_c_shoot_fast(s,ref_point[0], ref_point[1], dx, dy, 0, taptime_action)
				# if taptime_action >0:
				# 	time.sleep(taptime_action)
				# 	is_clicked = comm.comm_click_in_center(s)

				# pdb.set_trace() ##

				reward, new_score, next_screenshot_path, game_state = dqn_utils.get_score_after_shot(current_dir, wrapper, s, start_score)
				print("start_score, reward, new_score : ",start_score, reward, new_score)

				screenshot_path = shot_dir+"/s_%d_%d_%d_%s.png"%(t+1, angle_action, taptime_action, game_state)
				shutil.copy(next_screenshot_path, screenshot_path)
				save_path = screenshot_path+"_seg.png"
				state_img = wrapper.save_seg(screenshot_path, save_path)
				with open(os.path.join(shot_dir, 'action'), 'wb') as f:
					pickle.dump(action, f)
				with open(os.path.join(shot_dir, 'reward'), 'wb') as f:
					pickle.dump(reward, f)
				# next_state = dqn_utils.get_feature_4096(model=vgg16, img_path=save_path)

				# If our replay memory is full, pop the first element
				# if len(replay_memory) == replay_memory_size:
				#     replay_memory.pop(0)

				# Save transition to replay memory
				# replay_memory.append(Transition(state, action, reward, next_state, game_state))

				# Update statistics
				# stats.episode_rewards[i_episode] += reward # i_episode번째의 episode의 총 reward를 얻기 위해 계속 누적
				# stats.episode_lengths[i_episode] = t # i_episode번째의 길이를 얻기 위해 t 값으로 계속 저장

				# minibatch로 q network weight update
				# batch_size = 6
				# discount_factor = 0.99
				#
				# if len(replay_memory) > batch_size:
				# 	samples = random.sample(replay_memory, batch_size)
				# 	states_batch, action_batch, reward_batch, next_states_batch, game_state_batch = map(np.array, zip(*samples))
				# 	# (1,1,4096) (1,2) (1,) (1,)
				#
				# 	done_batch = np.array([1 if (game_state =='LOST' or 'WON') else 0 for game_state in game_state_batch])
				#
				# 	# angle_action_batch = np.array([action_batch[i][0] for i in range(batch_size)])
				# 	# taptime_action_batch = np.array([action_batch[i][1] for i in range(batch_size)])
				#
				# 	angle_action_batch_idx = np.array([valid_angles.index(action_batch[i][0]) for i in range(batch_size)])
				# 	taptime_action_batch_idx = np.array([valid_taptimes.index(action_batch[i][1]) for i in range(batch_size)])
				#
				# 	# 학습에 넣을 target reward 계산
				#
				# 	angle_q_values_next = angle_estimator.predict(sess, next_states_batch)
				# 	best_angle_actions = np.argmax(angle_q_values_next, axis=1)
				# 	taptime_q_values_next = taptime_estimator.predict(sess, next_states_batch)
				# 	best_taptime_actions = np.argmax(taptime_q_values_next, axis=1)
				#
				# 	angle_q_values_next_target = angle_target_estimator.predict(sess, next_states_batch)
				# 	taptime_q_values_next_target = taptime_target_estimator.predict(sess, next_states_batch)
				#
				# 	angle_targets_batch = reward_batch + np.invert(done_batch).astype(np.float32) * \
			    #         discount_factor * angle_q_values_next_target[np.arange(batch_size), best_angle_actions]
				#
				# 	taptime_targets_batch = reward_batch + np.invert(done_batch).astype(np.float32) * \
			    #         discount_factor * taptime_q_values_next_target[np.arange(batch_size), best_taptime_actions]
				#
				# 	# Perform gradient descent update
				# 	states_batch = np.array(states_batch)
				# 	# angle_loss = angle_estimator.update(sess, states_batch, angle_action_batch, angle_targets_batch)
				# 	# taptime_loss = taptime_estimator.update(sess, states_batch, taptime_action_batch, taptime_targets_batch)
				#
				# 	angle_loss = angle_estimator.update(sess, states_batch, angle_action_batch_idx, angle_targets_batch)
				# 	taptime_loss = taptime_estimator.update(sess, states_batch, taptime_action_batch_idx, taptime_targets_batch)
				#
				# 	print('Learning done (loss: ', angle_loss, taptime_loss, ')')

				# state 판별:
				# if game_state가 playing이 아니면 :
				if game_state!='PLAYING':
					break

				# state = next_state
				total_t += 1

			print('\nOne episode is done')

		# monitoring-tensorflow board?
		# level별로, reward per episode 기록?
		print()

	## test()하는 모듈

	# init = tf.global_variables_initializer()
	# sess = tf.Session()
	# sess.run(init)

	# batch_size = 100

	# for epoch in range(15):
	#     total_cost = 0

	#     for i in range(total_batch):

	#         # replay memory에서 batch_size만큼 data를 가져옴

	#         _, cost_val = sess.run([optimizer, cost], feed_dict={X: state, Y: target_reward})
	#         total_cost += cost_val

	#     print('Epoch:', '%04d' % (epoch + 1),
	#           'Avg. cost =', '{:.3f}'.format(total_cost / total_batch))

	# is_correct = tf.equal(tf.argmax(model, 1), tf.argmax(Y, 1))
	# accuracy = tf.reduce_mean(tf.cast(is_correct, tf.float32))
	# print('정확도:', sess.run(accuracy,
	#                         feed_dict={X: mnist.test.images,
	#                                    Y: mnist.test.labels}))
	print("session 종료")
