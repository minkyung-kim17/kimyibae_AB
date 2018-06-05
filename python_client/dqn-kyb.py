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
dqn_logger.setLevel(logging.DEBUG) # Debug < Info < Warning < Error < Critical
dqn_logger.addHandler(logging.StreamHandler())
dqn_logger.addHandler(logging.FileHandler("dqn_logger.log"))

# path 설정 ...
current_path = inspect.getfile(inspect.currentframe())
current_dir = os.path.dirname(os.path.abspath(current_path))
# EXP_PATH_gather=os.path.join(current_dir,"experiences_gathering")
EXP_PATH=os.path.join(current_dir,"experiences")
SCR_PATH=os.path.join(current_dir,"screenshots")
SUMM_PATH=os.path.join(current_dir, "tensorboard") # tf.summary dir

# if not os.path.exists(EXP_PATH_gather):
	# print('There is no replay_memory')
	# sys.exit()
if not os.path.exists(EXP_PATH):
	os.mkdir(EXP_PATH)
if not os.path.exists(SCR_PATH):
	os.mkdir(SCR_PATH)
if not os.path.exists(SUMM_PATH):
	os.mkdir(SUMM_PATH)

checkpoint_dir = os.path.join(current_dir, "checkpoints-oneNN")
checkpoint_path = os.path.join(checkpoint_dir, "model") # checkpoint file path

if not os.path.exists(checkpoint_dir):
	os.makedirs(checkpoint_dir)

run_start_dir = os.path.join(EXP_PATH, "startAt_%s"%time.strftime("%Y%m%d_%H%M"))


####################################################################################
print('Gazua Angry Bird!') #########################################################
####################################################################################

update_target_estimator_every = 10

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
# tf.reset_default_graph() # 주피터 노트북과 같이 컨텍스트가 유지되는 경우에는 graph reset을 해 줘야 한다고 함
						   # 일반 파이썬은 괜찮음

# initialize feature extraction
vgg16 = VGG16(weights= 'imagenet')

# set the action sets # 앗 여기도 csv 파일로 읽어서 만들면 편할 듯.. 나중에 각도 추가 하거나 뭐 할때
# valid_angles = list(range(5, 86, 5)) # 5도부터 85도까지 5도씩 증가
# valid_angles = dqn_utils.get_valid_angles()
# valid_taptimes = list(range(500, 2501, 100))  # 500부터 2500까지 100씩 증가
valid_angles = [8, 10, 11, 14, 17, 18, 19, 20, 21, 22, 23, 26, 30, 31, 34, 35, 36, 46, 61, 65, 67, 70] # 55제외 
valid_taptimes = [600, 700, 900, 1000, 1100, 1200, 1300, 1500, 1600, 1700, 1800, 2000, 2500]

# Create a global step variable
global_step = tf.Variable(0, name='global_step', trainable=False)

# initialize Q-network
estimator = q_network.DQN_Estimator(scope="estimator", angle_output_size=len(valid_angles), taptime_output_size=len(valid_taptimes), summaries_dir=SUMM_PATH)
target_estimator = q_network.DQN_Estimator(scope="target_estimator", angle_output_size=len(valid_angles), taptime_output_size=len(valid_taptimes))

# Keeps track of useful statistics
EpisodeStats = namedtuple("Stats",["episode_lengths", "episode_rewards"])
stats = EpisodeStats( # level별 episode_length랑, episode_reward를 저장해 둘 수 있는 list
        episode_lengths=[[] for i in range(21)],
        episode_rewards=[[] for i in range(21)])


##### Open tensorflow session
with tf.Session() as sess:
	sess.run(tf.global_variables_initializer()) 

	saver = tf.train.Saver()

	latest_checkpoint = tf.train.latest_checkpoint(checkpoint_dir) # path를 반환
	if latest_checkpoint:
		print("Loading model checkpoint {}...\n".format(latest_checkpoint))
		saver.restore(sess, latest_checkpoint)

	total_t = sess.run(tf.train.get_global_step()) 
	# 처음에 안됐었던 이유는, global_step이란 tensor 변수를 안만들어서임

	## user parameter
	epsilon_start = 1.0
	epsilon_end = 0.1
	epsilon_decay_steps = 1000000
	epsilons = np.linspace(epsilon_start, epsilon_end, epsilon_decay_steps)

	policy= dqn_utils.make_epsilon_greedy_policy(
	        estimator,
	        [len(valid_angles), len(valid_taptimes)])

	########################################
	##### Populating replay memory (size: N)
	# 원래는 랜덤하게 N번의 shot을 해서 replay_memory를 채워야 하지만...
	# 각 레벨별로, 0도부터 90도까지 쏜 데이터를 replay_memory로 함.
	batch_size = 6
	discount_factor = 0.99
	Transition = namedtuple("Transition", ["state", "action", "reward", "next_state", "game_state"])
	replay_memory_size = 500000
	print('Populating replay memory...')
	# replay_memory = [] # pretrain_memory가 사실 replay_memory가 되는게 아닌가? 
	# with open(os.path.join(EXP_PATH, 'pretrain_memory_5'), 'rb') as f:
	replay_memory_filename = 'replay_memory_0604_0057' # 가장 처음 만든 1-21memory
	with open(os.path.join(EXP_PATH, replay_memory_filename), 'rb') as f:
		replay_memory = pickle.load(f)
	with open(os.path.join(EXP_PATH, 'replay_memoryAll_2'), 'rb') as f: # 두번째 만든 1-21memory
		replay_memory_2 = pickle.load(f)
	with open(os.path.join(EXP_PATH, 'replay_memory_18_21'), 'rb') as f: # 두번째 만든 18-21memory
		replay_memory_3 = pickle.load(f)
	replay_memory = replay_memory + replay_memory_2 + replay_memory_3 # 2694, 4415
	# pdb.set_trace() 
	print("Loading replay memory ", replay_memory_filename)	

	# pretrain_memory = dqn_utils.init_replaymemory(5, EXP_PATH, current_dir, vgg16)
	# with open(os.path.join(EXP_PATH, 'pretrain_memory_5'), 'wb') as f:
	# 	pickle.dump(pretrain_memory, f)
	# with open(os.path.join(EXP_PATH, 'pretrain_memory_5'), 'rb') as f:
		# pretrain_memory = pickle.load(f)
	
	# while True:
	# # pdb.set_trace()
	# 	# angle_loss, taptime_loss = dqn_utils.pretrain(pretrain_memory, valid_angles, valid_taptimes, estimator, target_estimator, sess, batch_size, discount_factor)
	# 	loss = dqn_utils.pretrain(pretrain_memory, valid_angles, valid_taptimes, estimator, target_estimator, sess, batch_size, discount_factor)
	# 	saver.save(sess, checkpoint_path)
	# 	total_t += 1
	# 	if total_t % update_target_estimator_every == 0:
	# 		dqn_utils.copy_model_parameters(sess, estimator, target_estimator)
	# 		# dqn_utils.copy_model_parameters(sess, taptime_estimator, taptime_target_estimator)
	# 		# print("total_t:", total_t, "angle_loss:", angle_loss, "taptime_loss;", taptime_loss)
	# 		print("total_t:", total_t, "loss:", loss)

	def local_train(replay_memory, valid_angles, valid_taptimes, estimator, 
		target_estimator, sess, batch_size, discount_factor, total_t, update_target_estimator_every, saver, checkpoint_path):
		while True:
			loss = dqn_utils.pretrain(replay_memory, valid_angles, valid_taptimes, estimator, target_estimator, sess, batch_size, discount_factor)
			total_t += 1 # 이 코드가 실행될 때, 불러온 checkpoint에서부터 계속 더하기 1을 해나감
			# pdb.set_trace()
			# total_t = sess.run(tf.train.get_global_step()) # 이렇게 total_t를 계산해야 main thread랑 sync가 맞을 것 같은데... 왜 자꾸 에러ㄷㄷ
			if total_t % update_target_estimator_every == 0:
				saver.save(sess, checkpoint_path)
				dqn_utils.copy_model_parameters(sess, estimator, target_estimator)
			if total_t % (update_target_estimator_every*10) == 0:
				print("local_train || total_t:", total_t, "loss:", loss)

	threads= []
	import threading
	num_threads=1
	for i in range(num_threads):
		t=threading.Thread(target=local_train, args=(replay_memory, valid_angles, valid_taptimes, estimator, 
			target_estimator, sess, batch_size, discount_factor, total_t, update_target_estimator_every, saver, checkpoint_path))
		threads.append(t)
		t.start()

	# replay memory로 pre_train한 network를 쓴다면, 여기서 load
	## --> 강화학습에서 이렇게 하면 안됨. 미리 replay memory를 만드는 건 ok지만, 
	## 게임을 해나가면서 training을 해야 함... 

	####################################################################################
	print('Start Learning!') ### 게임을 하면서, 학습을 하면서, policy를 업데이트 ##########
	####################################################################################

	i_episode = 0 # 전체 episode수
	i_episodes = [0]*21 # 각 레벨별 episode수

	dqn_utils.copy_model_parameters(sess, estimator, target_estimator)

	training_start_level = 1
	current_level = training_start_level

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

			# comm.comm_load_level(s, np.random.randint(1,22), silent=False)
			comm.comm_load_level(s, current_level, silent=False) # initial current_level 1

			print ("level is loaded")

		elif game_state=='LOADING':
			print ("########################################################")
			print ("Loading state")
			pass

		elif game_state=='WON':
			print ("########################################################")
			print ("Won state")
			# resater random level
			# comm.comm_load_level(s, np.random.randint(1,22), silent=False)

			if current_level == 21 :#or lost_flag == 1:
				# lost_flag = 0
				current_level = training_start_level
			else:
				current_level +=1
				
			comm.comm_load_level(s, current_level, silent=False)


		elif game_state=='LOST':
			print ("########################################################")
			print ("Lost state")
			# restart random level
			# comm.comm_load_level(s, np.random.randint(1,22), silent=False)
			comm.comm_restart_level(s)

		elif game_state=='PLAYING':
			print ("########################################################")
			print ("Playing state")

			i_episode += 1

			episode_dir = "%s_%d"%(run_start_dir,i_episode)
			if not os.path.exists(episode_dir):
				os.mkdir(episode_dir)

			# 새 에피소드 시작할 때마다 checkpoint 새로 생성
			saver.save(sess, checkpoint_path) # thread안에서도 학습할때 checkpoint 저장하고 있음...
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

				while True:
					is_zoomed_out = comm.comm_fully_zoomout(s)
					if is_zoomed_out==1:
						break
				screenshot_path = shot_dir+"/s_%d.png"%t
				state_raw_img = comm.comm_do_screenshot(s, screenshot_path)
				save_path = screenshot_path+"_seg.png"
				state_img = wrapper.save_seg(screenshot_path, save_path)
				state = dqn_utils.get_feature_4096(model=vgg16, img_path=save_path) 
				# pdb.set_trace()

				print('Choose action from given Q network model')

				# Epsilon for this time step
				epsilon = epsilons[min(total_t, epsilon_decay_steps-1)]

				# Add epsilon to Tensorboard
				episode_summary = tf.Summary() # 수정: 샷을 할때마다 episode_summary 객체를 계속 새로 만드는 것인가...?
				episode_summary.value.add(simple_value=epsilon, tag="epsilon")
				estimator.summary_writer.add_summary(episode_summary, total_t)

				# Update the target estimator
				if total_t % update_target_estimator_every == 0:
					dqn_utils.copy_model_parameters(sess, estimator, target_estimator)
					# print("\nCopied model parameters to target network.")

				# pdb.set_trace()

				# Take a step (현재 policy(현재 Q-network)로 다음 action을 정함)
				angle_action_probs, taptime_action_probs = policy(sess, state, epsilon)

				angle_action_idx = np.random.choice(np.arange(len(angle_action_probs)), p=angle_action_probs)
				taptime_action_idx = np.random.choice(np.arange(len(taptime_action_probs)), p=taptime_action_probs)

				# make shot for shooting
				slingshot_rect = None
				while(slingshot_rect == None or slingshot_rect[0]==-1 or slingshot_rect[1]==-1):
					is_zoomed_out = 0;
					while is_zoomed_out !=1:
						is_zoomed_out = comm.comm_fully_zoomout(s)
					slingshot_rect = wrapper.get_slingshot(screenshot_path = screenshot_path)
				ref_point = dqn_utils.get_slingshot_refpoint(slingshot = slingshot_rect)
				max_mag = slingshot_rect[3]
				angle_action = valid_angles[angle_action_idx]
				taptime_action = valid_taptimes[taptime_action_idx]
				dx = -max_mag * math.cos(math.radians(angle_action))
				dy = max_mag * math.sin(math.radians(angle_action))
				action = [angle_action, taptime_action]
				print("Choose action: angle: {}, taptime: {}".format(angle_action, taptime_action), end="\n")
				print("   with probs: angle: {}, taptime: {}".format(angle_action_probs[angle_action_idx], taptime_action_probs[taptime_action_idx]), end="\n") 

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

				# Q-network update with minibatch
				# 게임 중에도 minibatch로 q network weight update ? 필요?
				## 어차피 이 게임의 experience가 들어가도 다시 random으로 minibatch sample을 고르는데...?
				## 그리고 이 짓을 thread가 하는데? 
				if len(replay_memory) > batch_size:
					loss = dqn_utils.pretrain(replay_memory, valid_angles, valid_taptimes, estimator, target_estimator, sess, batch_size, discount_factor)
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

					print('Learning done! (loss:', loss, ')')

				# state 판별:
				# if game_state가 playing이 아니면 :
				if game_state!='PLAYING':
					break

				# state = next_state
				# total_t += 1
				total_t = sess.run(tf.train.get_global_step()) # thread랑 같이 써서... 더하기를 하지말고 계속 받아오는 걸로...

			print('\nOne episode is done')

			 # Add summaries to tensorboard
			episode_summary.value.add(simple_value=stats.episode_rewards[current_level-1][-1], node_name="episode_reward_Level_%d"%(current_level), tag="episode_reward_Level_%d"%(current_level))
			episode_summary.value.add(simple_value=stats.episode_lengths[current_level-1][-1], node_name="episode_length_Level_%d"%(current_level), tag="episode_length_Level_%d"%(current_level))

			estimator.summary_writer.add_summary(episode_summary, total_t)
			estimator.summary_writer.flush()

		print()

	# tensorboard 실행 방법: cmd 명령으로 실행 (!!!!!! tensorflow가 있는 환경에서 !!!!!)
	# tensorboard --logdir="C:\Users\mkkim\Github\kimyibae_AB\python_client\tensorboard\summaries_angle_estimator"
	# 그 이후 나온 url 실행 (ex. http://DESKTOP-5KPGCLS:6006)

	print("session 종료")
