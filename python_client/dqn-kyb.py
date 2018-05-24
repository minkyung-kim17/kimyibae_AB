import os, inspect, logging, glob, time, math, itertools
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
# SCR_PATH=os.path.join(current_dir,"screenshots")

if not os.path.exists(EXP_PATH):
			os.mkdir(EXP_PATH) 
# if not os.path.exists(SCR_PATH):
			# os.mkdir(SCR_PATH) 

checkpoint_dir = os.path.join(current_dir, "checkpoints")
checkpoint_path = os.path.join(checkpoint_dir, "model") # checkpoint file path

if not os.path.exists(checkpoint_dir):
		os.makedirs(checkpoint_dir)

# episode_prefix	= '%s/startAt_%s'%(EXP_PATH, time.strftime("%Y%m%d_%H%M"))
run_start_dir = os.path.join(EXP_PATH, "startAt_%s"%time.strftime("%Y%m%d_%H%M"))

# while True:
# 	pass

def get_state(screenshots_list, state_name, playing_state):
	
	#######################################################################
	# Stack two images : After shooting image, waiting for shooting image.
	# params :
	# 	screenshots_list : two images to be stacked
	# 	state_name : image name to be saved
	# 	playing_state : 'FIRST_SHOT', 'FINAL_SHOT', 'AMONG_SHOTS'
	# return :
	# 	No return, save state_name.png
	#######################################################################
	
	cropped = []
	length = len(screenshots_list)
	if (length == 1):
		if playing_state == 'FIRST_SHOT':
			dqn_logger.debug("First shot... Stack images.")
			try:
				cropped.append(Image.open(current_dir+"/all_black.png"))
				cropped.append(Image.open(screenshots_list[0]).crop((70,45,760,390)))
			except Exception as e:
				dqn_logger.error("Error occurs at stacking two images.")
				dqn_logger.error(str(e))
		elif playing_state == 'FINAL_SHOT':
			dqn_logger.debug("Final shot... Stack images.")
			try:
				cropped.append(Image.open(screenshots_list[0]).crop((70,45,760,390)))
				cropped.append(Image.open(current_dir+"/all_black.png"))
			except Exception as e:
				dqn_logger.error("Error occurs at stacking two images.")
				dqn_logger.error(str(e))
		else:
			print("playing_state error, current state is ", playing_state)
	elif (length == 2):
		if playing_state == 'AMONG_SHOTS':
			dqn_logger.debug("Keep shooting... Stack images.")
			try:
				for screenshot in screenshots_list:
					cropped.append(Image.open(screenshot).crop((70,45,760,390)))
			except Exception as e:
				dqn_logger.error("Error occurs at stacking two images.")
				dqn_logger.error(str(e))
		else:
			dqn_logger.error("playing_state error, current state is ", playing_state)
	else:
		dqn_logger.error("More than two images in the list.")

	stacked = np.vstack([np.asarray(crop) for crop in cropped])
	stacked = Image.fromarray(stacked).resize((224,224))
	try:
		if not os.path.exists(EXP_PATH+'/'+state_name):
			os.mkdir(EXP_PATH+'/'+state_name) #screenshot 아래에 state_name으로 생김
		stacked.save(EXP_PATH+"/"+state_name+"/"+state_name+".png")
	except Exception as e:
		dqn_logger.error("Error occurs at saving stacked image.")
		dqn_logger.error(str(e))

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

# initialize tensorflow session
# tf.reset_default_graph() # ?? 
sess = tf.Session() 

# initialize feature extraction
vgg16 = VGG16(weights= 'imagenet')

# set the action sets
valid_angles = list(range(5, 86, 5)) # 5도부터 85도까지 5도씩 증가
valid_taptimes = list(range(500, 2501, 100)) # 500부터 2500까지 100씩 증가 

angle_estimator = q_network.DQN_Estimator(scope="angle_estimator", output_size=len(valid_angles), summaries_dir=None)
angle_target_estimator = q_network.DQN_Estimator(scope="angle_target_estimator", output_size=len(valid_angles))
taptime_estimator = q_network.DQN_Estimator(scope="taptime_estimator", output_size=len(valid_taptimes), summaries_dir=None)
angle_target_estimator = q_network.DQN_Estimator(scope="taptime_target_estimator", output_size=len(valid_taptimes)) 
# angle_estimator, angle_target_estimator = DQN_Estimator(obs_size, sess, fe, sc_parser, "angle", valid_angles) # 수정 필요
# taptime_estimator, taptime_target_estimator = DQN_Estimator(obs_size, sess, fe, sc_parser, "taptime", valid_taptimes) # 수정 필요 

########################

sess.run(tf.global_variables_initializer())

saver = tf.train.Saver()

latest_checkpoint = tf.train.latest_checkpoint(checkpoint_dir)
if latest_checkpoint:
	print("Loading model checkpoint {}...\n".format(latest_checkpoint))
	saver.restore(sess, latest_checkpoint)

try: 
	total_t = sess.run(tf.train.get_global_step()) 
except:
	total_t = 0

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
print('Populating replay memory...')
replay_memory = []
# replay_memory = np.load('').tolist()

#####################
##### Checkpoint load
# replay memory로 pre_train한 network를 쓴다면, 여기서 load

num_episodes = 10
####################################################################################
print('Start Learning!') ### 게임을 하면서, 학습을 하면서, policy를 업데이트 ##########
####################################################################################

for i_episode in range(num_episodes): # 문제가 있음: 다른 메뉴 --> playing으로 갈때, i_episode가 증가... 
									  # 여기를 while로 바꾸고, 다른 방법으로 i_episode를 count해야 할 듯
	
	episode_dir = "%s_%d"%(run_start_dir,i_episode) 
	if not os.path.exists(episode_dir):
			os.mkdir(episode_dir) 

	game_state = comm.comm_get_state(s, silent=False)

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
		loss = None
		
		print ("########################################################")
		print ("Level selection state")

		comm.comm_load_level(s, 1, silent=False)

		print ("level is loaded")

	elif game_state=='LOADING':
		print ("########################################################")
		print ("Loading state")
		pass
	elif game_state=='WON':
		print ("########################################################")
		print ("Won state")
		# resater random level

	elif game_state=='LOST':
		print ("########################################################")
		print ("Lost state")
		# restart random level 

	elif game_state=='PLAYING':
		print ("########################################################")
		print ("Playing state")

		# saver.save(tf.get_default_session(), checkpoint_path) # tf.get_default_session()이 none이 됨...
		saver.save(sess, checkpoint_path)
		current_level = comm.comm_get_current_level(s)
		last_score = 0;

		print("=============== Level",current_level,"===============")

		for t in itertools.count(): # 이 에피소드가 끝날때까지
			
			if t==0:
				last_score = 0;

			shot_dir = os.path.join(episode_dir, "level%d_shot%d_%s"%(current_level, t, time.strftime('%Y%m%d_%H%M%S')))
			if not os.path.exists(shot_dir):
				os.mkdir(shot_dir)

			screenshot_path = shot_dir+"/s_%d.png"%t
			state_raw_img = comm.comm_do_screenshot(s, screenshot_path)
			save_path = screenshot_path+"_seg.png"
			state_img = wrapper.save_seg(screenshot_path, save_path) # 함수 안에서 크기조절 
			state = dqn_utils.get_feature_4096(vgg16, save_path) #(1,4096)

			print('Choose action from given Q network model')

			# Epsilon for this time step
			epsilon = epsilons[min(total_t, epsilon_decay_steps-1)]

			# Add epsilon to Tensorboard
			# episode_summary = tf.Summary()
			# episode_summary.value.add(simple_value=epsilon, tag="epsilon")
			# q_estimator.summary_writer.add_summary(episode_summary, total_t)

			# Maybe update the target estimator
			if total_t % update_target_estimator_every == 0:
				pass # 여기 다시 만들어야 함 오류 팡팡팡
				# dqn_utils.copy_model_parameters(sess, angle_estimator, angle_target_estimator)
				# dqn_utils.copy_model_parameters(sess, taptime_estimator, taptime_target_estimator)
				# print("\nCopied model parameters to target network.")
			
			pdb.set_trace()   
			# Print out which step we're on, useful for debugging.
			print("\rStep {} ({}) @ Episode {}/{}, loss: {}".format(
			        t, total_t, i_episode + 1, num_episodes, loss), end="")
			sys.stdout.flush() # ?

			# Take a step (현재 policy로 다음 action을 정하네)
			angle_action_probs = angle_policy(sess, state, epsilon)
			taptime_action_probs = taptime_policy(sess, state, epsilon)
			
			angle_action_idx = np.random.choice(np.arange(len(angle_action_probs)), p=angle_action_probs)
			taptime_action_idx = np.random.choice(np.arange(len(taptime_action_probs)), p=taptime_action_probs)

			# make shot for shooting
			slingshot_rect = wrap.get_slingshot(screenshot_path = screenshot_path)
			ref_point = wrap.get_slingshot_refpoint(slingshot = slingshot_rect, silent = False)
			max_mag = slingshot_rect[3]
			angle_action = valid_angles[angle_action_idx]
			taptime_action = valid_taptime[taptime_action_idx]
			dx = -max_mag * math.cos(angle_action)
			dy = max_mag * math.sin(angle_action)
			
			# shoot
			shoot_complete = comm.comm_c_shoot_fast(s,ref_point[0], ref_point[1], dx, dy, 0, 0)
			if taptime_action >0:
				time.sleep(taptime_action)
				is_clicked = comm.comm_click_in_center(s)

			temp = get_score_in_game(self, screenshot_path)



			next_state, reward, done, _ = env.step(VALID_ACTIONS[action])
			
			next_state = state_processor.process(sess, next_state)
			next_state = np.append(state[:,:,1:], np.expand_dims(next_state, 2), axis=2)

			# If our replay memory is full, pop the first element
			if len(replay_memory) == replay_memory_size:
			    replay_memory.pop(0)

			# Save transition to replay memory
			replay_memory.append(Transition(state, action, reward, next_state, done))   

			# Update statistics
			stats.episode_rewards[i_episode] += reward # i_episode번째의 episode의 총 reward를 얻기 위해 계속 누적
			stats.episode_lengths[i_episode] = t # i_episode번째의 길이를 얻기 위해 t 값으로 계속 저장





			# action

			# 우리가 가지고 있는 q랑 policy로 action 고르고, 이걸 넘겨줘야함 
			# 실제 고른 action하고, reward 받고, replay memory에 저장
			# shoot_and_get_score: next_state 받기? 샷 찍음
			# state == WIN||LOST 면 s' 에 새 스크린 샷 저장
			# result & learning 

			# minibatch로 q network weight update
			samples = random.sample(replay_memory, batch_size)
			states_batch, action_batch, reward_batch, next_states_batch, done_batch = map(np.array, zip(*samples))

			# 학습에 넣을 target reward 계산
			q_values_next = q_estimator.predict(sess, next_states_batch)
			best_actions = np.argmax(q_values_next, axis=1)

			q_values_next_target = target_estimator.predict(sess, next_states_batch)

			targets_batch = reward_batch + np.invert(done_batch).astype(np.float32) * \
	            discount_factor * q_values_next_target[np.arange(batch_size), best_actions]

	        # Perform gradient descent update
			states_batch = np.array(states_batch)
			loss = q_estimator.update(sess, states_batch, action_batch, targets_batch)

			# state 판별:
			# if game_state가 playing이 아니면 :
			if done:
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
