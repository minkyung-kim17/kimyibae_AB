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

model_type = input("Write your model number (oneNN(1) or parNN(2)):")

# log 설정 ... # test라 일단 logger 뺌 ...?
# dqn_logger = logging.getLogger("dqn_logger")
# dqn_logger.setLevel(logging.DEBUG) # Debug < Info < Warning < Error < Critical
# dqn_logger.addHandler(logging.StreamHandler())
# dqn_logger.addHandler(logging.FileHandler("dqn_logger.log"))

# path 설정 ...
current_path = inspect.getfile(inspect.currentframe())
current_dir = os.path.dirname(os.path.abspath(current_path))
EXP_PATH=os.path.join(current_dir,"experiences_gathering-test") 
SCR_PATH=os.path.join(current_dir,"screenshots")
# SUMM_PATH=os.path.join(current_dir, "tensorboard-test") # test에서 tensorboard를 기록해야 하는가? 

if not os.path.exists(EXP_PATH):
	os.mkdir(EXP_PATH)
if not os.path.exists(SCR_PATH):
	os.mkdir(SCR_PATH)
# if not os.path.exists(SUMM_PATH):
	# os.mkdir(SUMM_PATH)

if model_type == 1:
	checkpoint_dir = os.path.join(current_dir, "checkpoints-oneNN") # saver.restore 할때는 여기 
	# checkpoint_path = os.path.join(checkpoint_dir, "model") # checkpoint file path, saver.save 할때는 여기
	# 우선 test에서 학습을 안하도록 했음... 
elif model_type == 2:
	checkpoint_dir = os.path.join(current_dir, "checkpoints-parNN") # saver.restore 할때는 여기

if not os.path.exists(checkpoint_dir):
	print("There is no checkpoint. Please learn first")
	# os.makedirs(checkpoint_dir)

run_start_dir = os.path.join(EXP_PATH, "startAt_%s"%time.strftime("%Y%m%d_%H%M"))

####################################################################################
print('Gazua Angry Bird!') #########################################################
####################################################################################

# update_target_estimator_every = 10 # test라 학습을 안하면, 여기도 필요없지 

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

# initialize feature extraction
vgg16 = VGG16(weights= 'imagenet')

# set the action sets
valid_angles = list(range(5, 86, 5)) # 5도부터 85도까지 5도씩 증가
valid_taptimes = list(range(500, 2501, 100))  # 500부터 2500까지 100씩 증가

# Create a global step variable # 일단 시작은 0이고, checkpoint를 불러오면 저장된 global_step이 들어오는건가...
global_step = tf.Variable(0, name='global_step', trainable=False)

angle_estimator = q_network.DQN_Estimator(scope="angle_estimator", output_size=len(valid_angles))
angle_target_estimator = q_network.DQN_Estimator(scope="angle_target_estimator", output_size=len(valid_angles))
taptime_estimator = q_network.DQN_Estimator(scope="taptime_estimator", output_size=len(valid_taptimes))
taptime_target_estimator = q_network.DQN_Estimator(scope="taptime_target_estimator", output_size=len(valid_taptimes))

# Keeps track of useful statistics
EpisodeStats = namedtuple("Stats",["episode_lengths", "episode_rewards"])
stats = EpisodeStats( # level별 episode_length랑, episode_reward를 저장해 둘 수 있는 list
        episode_lengths=[[] for i in range(21)],
        episode_rewards=[[] for i in range(21)])

# initialize and open session
with tf.Session() as sess:
	sess.run(tf.global_variables_initializer())

	saver = tf.train.Saver()

	latest_checkpoint = tf.train.latest_checkpoint(checkpoint_dir) # path를 반환
	# latest_checkpoint = tf.train.latest_checkpoint(checkpoint_dir, latest_filename="checkpoint") # checkpoint 여러개를 저장해 놨다가 쓸수 있을듯
	if latest_checkpoint:
		print("\n")
		print("Loading model checkpoint {}...".format(latest_checkpoint))
		saver.restore(sess, latest_checkpoint)
		
	total_t = sess.run(tf.train.get_global_step()) # 처음에 안됐었던 이유는, global_step이란 tensor 변수를 안만들어서임
	print("This checkpoint has been made after {} shots experience".format(total_t))

	epsilon = 0.2
	print("Epsilon used in this test is {}".format(epsilon))

	## test할때는... epsilon이 있어야 하는가?... 아주 학습이 잘 된 경우면 greedy로 뽑으면 되니까 epsilon=0? 혹시 몰라서 0.1로 아래 playing에서 줌
	# epsilon_start = 1.0
	# epsilon_end = 0.1
	# epsilon_decay_steps = 500000
	# epsilons = np.linspace(epsilon_start, epsilon_end, epsilon_decay_steps)

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
	
	# test에서 학습 안하니까... 필요는 없는데... 
	batch_size = 6
	discount_factor = 0.99
	Transition = namedtuple("Transition", ["state", "action", "reward", "next_state", "game_state"])
	replay_memory_size = 500000
	# print('Populating replay memory...')
	replay_memory = []
	# replay_memory = np.load('').tolist()

	#####################
	##### Checkpoint load
	# pretrain_memory = dqn_utils.init_replaymemory(5, EXP_PATH, current_dir, vgg16)
	# with open(os.path.join(EXP_PATH, 'pretrain_memory_5'), 'wb') as f:
	# 	pickle.dump(pretrain_memory, f)
	# with open(os.path.join(EXP_PATH, 'pretrain_memory_5'), 'rb') as f: # 여기가 마지막으로 사용한 코드 
		# pretrain_memory = pickle.load(f)
	# def run_pretrain():

	# while True:
	# # pdb.set_trace()
	# 	angle_loss, taptime_loss = dqn_utils.pretrain(pretrain_memory, valid_angles, valid_taptimes, angle_estimator, taptime_estimator, angle_target_estimator, taptime_target_estimator, sess, batch_size, discount_factor, pretrain = True)
	# 	saver.save(sess, checkpoint_path)
	# 	total_t += 1
	# 	if total_t % update_target_estimator_every == 0:
	# 		dqn_utils.copy_model_parameters(sess, angle_estimator, angle_target_estimator)
	# 		dqn_utils.copy_model_parameters(sess, taptime_estimator, taptime_target_estimator)
	# 		print("total_t:", total_t, "angle_loss:", angle_loss, "taptime_loss;", taptime_loss)

	# threads= []
	# import threading
	# num_threads=1
	# for i in range(num_threads):
	# 	t=threading.Thread(target=run_pretrain)
	# 	threads.append(t)
	# 	t.start()

	# replay memory로 pre_train한 network를 쓴다면, 여기서 load

	# pdb.set_trace()

	####################################################################################
	print('Start Learning!') ### 게임을 하면서, 학습을 하면서, policy를 업데이트 ##########
	####################################################################################

	i_episode = 0 # 전체 episode수
	i_episodes = [0]*21 # 각 레벨별 episode수

	# tf.train.get_global_step()이 0일때, 학습을 한번도 한적이 없을 때, 
	# estimator와 target_estimator의 초기값을 같게 맞추려고 밑에 copy_model_parameters를 했는데... 
	# 어느정도 학습이 된 checkpoint를 부르는 거면... 필요 없을 것 같기도...
	# dqn_utils.copy_model_parameters(sess, angle_estimator, angle_target_estimator)
	# dqn_utils.copy_model_parameters(sess, taptime_estimator, taptime_target_estimator)

	# pdb.set_trace()
	current_level = 1
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
			print ("########################################################")
			print ("Level selection state")

			current_level = 1 # test 시작
			comm.comm_load_level(s, current_level, silent=False)

			print ("level is loaded")

		elif game_state=='LOADING':
			print ("########################################################")
			print ("Loading state")
			pass

		elif game_state=='WON':
			print ("########################################################")
			print ("Won state")
			current_level += 1
			comm.comm_load_level(s, current_level, silent=False)

		elif game_state=='LOST':
			print ("########################################################")
			print ("Lost state")
			print ("\n")
			print ("done...")
			break

		elif game_state=='PLAYING':
			print ("########################################################")
			print ("Playing state")

			i_episode += 1

			episode_dir = "%s_%d"%(run_start_dir,i_episode)
			if not os.path.exists(episode_dir):
				os.mkdir(episode_dir)

			# 새 에피소드 시작할 때마다 checkpoint 새로 생성, test니까 생성 안하도록... ! 
			# saver.save(sess, checkpoint_path)
			# current_level = comm.comm_get_current_level(s)

			i_episodes[current_level-1] += 1

			print("=============== Level",current_level,"===============")

			for t in itertools.count(): # 이 에피소드가 끝날때까지

				print('\n')
				print("\rStep {} ({}) @ Episode {} (Level {})".format(
				        t, total_t, i_episode, current_level), end="\n") # 가장 처음 시작된 이 부분의 total_t에서 몇번의 shot을 learning한 애인지... 알수있다
				sys.stdout.flush()

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
				state = dqn_utils.get_feature_4096(model=vgg16, img_path=save_path) # 수정: 이 함수 안에서 크기 조절하는게 좋을 듯

				print('Choose action from given Q network model')

				# Epsilon for this time step
				# epsilon = 0.1 #epsilons[min(total_t, epsilon_decay_steps-1)] # 학습이 되고 난 후에는 이걸 고정시켜야 하나? , 일단 밖에서 정해봄

				# Add epsilon to Tensorboard
				# episode_summary = tf.Summary() # 수정: 이거 왜 두번 만들고 있었지... ? main client 파일도 수정해야 함 
				# episode_summary.value.add(simple_value=epsilon, tag="epsilon")
				# angle_estimator.summary_writer.add_summary(episode_summary, total_t)
				# taptime_estimator.summary_writer.add_summary(episode_summary, total_t)

				# Update the target estimator
				# if total_t % update_target_estimator_every == 0: # 여기도 test에서는 필요 없고
					# pass # 수정: 여기 다시 만들어야 함 오류 팡팡팡
					# dqn_utils.copy_model_parameters(sess, angle_estimator, angle_target_estimator)
					# dqn_utils.copy_model_parameters(sess, taptime_estimator, taptime_target_estimator)
					# print("\nCopied model parameters to target network.")

				# pdb.set_trace()

				# Take a step (현재 policy로 다음 action을 정하네)
				angle_action_probs = policy_angle(sess, state, epsilon)
				taptime_action_probs = policy_taptime(sess, state, epsilon)

				angle_action_idx = np.random.choice(np.arange(len(angle_action_probs)), p=angle_action_probs)
				taptime_action_idx = np.random.choice(np.arange(len(taptime_action_probs)), p=taptime_action_probs)

				# make shot for shooting
				slingshot_rect = None
				while(slingshot_rect == None or slingshot_rect[0]==-1 or slingshot_rect[1]==-1):
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


				# if len(replay_memory) > batch_size:
					# angle_loss, taptime_loss = dqn_utils.pretrain(replay_memory, valid_angles, valid_taptimes, angle_estimator, taptime_estimator, angle_target_estimator, taptime_target_estimator, sess, batch_size, discount_factor)
					# samples = random.sample(replay_memory, batch_size)
					# states_batch, action_batch, reward_batch, next_states_batch, game_state_batch = map(np.array, zip(*samples))
					# reward_batch = np.clip(reward_batch/10000, 0, 6)
					# # (1,1,4096) (1,2) (1,) (1,)
					#
					# done_batch = np.array([1 if (game_state =='LOST' or 'WON') else 0 for game_state in game_state_batch])
					#
					# # angle_action_batch = np.array([action_batch[i][0] for i in range(batch_size)])
					# # taptime_action_batch = np.array([action_batch[i][1] for i in range(batch_size)])
					#
					# angle_action_batch_idx = np.array([valid_angles.index(action_batch[i][0]) for i in range(batch_size)])
					# taptime_action_batch_idx = np.array([valid_taptimes.index(action_batch[i][1]) for i in range(batch_size)])
					#
					# # 학습에 넣을 target reward 계산
					#
					# angle_q_values_next = angle_estimator.predict(sess, next_states_batch)
					# best_angle_actions = np.argmax(angle_q_values_next, axis=1)
					# taptime_q_values_next = taptime_estimator.predict(sess, next_states_batch)
					# best_taptime_actions = np.argmax(taptime_q_values_next, axis=1)
					#
					# angle_q_values_next_target = angle_target_estimator.predict(sess, next_states_batch)
					# taptime_q_values_next_target = taptime_target_estimator.predict(sess, next_states_batch)
					#
					# angle_targets_batch = reward_batch + np.invert(done_batch).astype(np.float32) * \
			        #     discount_factor * angle_q_values_next_target[np.arange(batch_size), best_angle_actions]
					#
					# taptime_targets_batch = reward_batch + np.invert(done_batch).astype(np.float32) * \
			        #     discount_factor * taptime_q_values_next_target[np.arange(batch_size), best_taptime_actions]
					#
					# # Perform gradient descent update
					# states_batch = np.array(states_batch)
					# angle_loss = angle_estimator.update(sess, states_batch, angle_action_batch_idx, angle_targets_batch)
					# taptime_loss = taptime_estimator.update(sess, states_batch, taptime_action_batch_idx, taptime_targets_batch)

					# print('Learning done! (angle_loss:', angle_loss, 'taptime_loss:', taptime_loss, ')')

				# state 판별:
				# if game_state가 playing이 아니면 :
				if game_state!='PLAYING':
					break

				# state = next_state
				# total_t += 1 # test니까 +1 안함

			print('\nThis level is done')

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

		print()

	# tensorboard 실행 방법: cmd 명령으로 실행 (tensorflow가 있는 환경에서)
	# tensorboard --logdir="C:\Users\mkkim\Github\kimyibae_AB\python_client\tensorboard\summaries_angle_estimator"
	# 그 이후 나온 url 실행 (ex. http://DESKTOP-5KPGCLS:6006)

	print("session 종료")
