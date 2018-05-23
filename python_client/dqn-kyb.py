import os, inspect, logging, glob
import socket
from PIL import Image
import tensorflow as tf
import numpy as np

# sys.path.append('../')
from comm import * 
from wrapper_python import *
import dqn_utils

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

checkpoint_dir = os.path.join(EXP_PATH, "checkpoints")
checkpoint_path = os.path.join(checkpoint_dir, "model")

if not os.path.exists(checkpoint_dir):
        os.makedirs(checkpoint_dir)

def get_state(screenshots_list, state_name, playing_state):
	'''
	#######################################################################
	# Stack two images : After shooting image, waiting for shooting image.
	# params :
	# 	screenshots_list : two images to be stacked
	# 	state_name : image name to be saved
	# 	playing_state : 'FIRST_SHOT', 'FINAL_SHOT', 'AMONG_SHOTS'
	# return :
	# 	No return, save state_name.png
	#######################################################################
	'''
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

def build_q_network(input_size=4096, hidden_size=[1024, 512], output_size=21):
	'''
	Builds the Tensorflow graph
	'''

	# Placeholders for our Q-network
	X = tf.placeholder(tf.float32, [None, input_size]) # [배치크기, feature size]
	Y = tf.placeholder(tf.float32, [None, output_size]) # 각 state에서 얻을 수 있는 target reward 값

	# Neural network with 2 hidden layer
	W1 = tf.Variable(tf.random_normal([input_size, hidden_size[0]], stddev=0.01))
	b1 = tf.Variable(tf.random_normal([hidden_size[0]], stddev=0.01))
	L1 = tf.nn.relu(tf.matmul(X, W1)+b1)

	W2 = tf.Variable(tf.random_normal([hidden_size[0], hidden_size[1]], stddev=0.01))
	b2 = tf.Variable(tf.random_normal([hidden_size[1]], stddev=0.01))
	L2 = tf.nn.relu(tf.matmul(L1, W2)+b2)

	W3 = tf.Variable(tf.random_normal([hidden_size[1], output_size], stddev=0.01))
	b3 = tf.Variable(tf.random_normal([output_size], stddev=0.01))
	output = tf.matmul(L2, W3)+b3 # relu 거치지 않고, softmax를 함

	loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(logits=cost, labels=Y))

	optimizer = tf.train.AdamOptimizer(0.001).minimize(loss)

    # return

####################################################################################
print('Gazua Angry Bird!') #########################################################
####################################################################################

#################################
##### Initialize game environment
# connect to Server
print('Initialize connection to ABServer')
s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
s.settimeout(20)
s.connect(('localhost',2004))
_,_,_ = comm_configure(s, 1003) # ?? 

# connect to Java
wrapper = WrapperPython('127.0.0.1') 

# initialize tensorflow session
sess = tf.Session()

# set the action sets
valid_angles = list(range(5, 86, 5)) # 5도부터 85도까지 5도씩 증가
valid_taptimes = list(range(500, 2501, 100)) # 500부터 2500까지 100씩 증가 
angle_estimator, angle_target = DQN_Estimator(obs_size, sess, fe, sc_parser, "angle", valid_angles) # 수정 필요
taptime_estimator, taptime_target = DQN_Estimator(obs_size, sess, fe, sc_parser, "taptime", valid_taptimes) # 수정 필요 

########################
saver = tf.train.Saver()

latest_checkpoint = tf.train.latest_checkpoint(checkpoint_dir)
if latest_checkpoint:
	print("Loading model checkpoint {}...\n".format(latest_checkpoint))
	saver.restore(sess, latest_checkpoint)

total_t = sess.run(tf.train.get_global_step()) # ... ?

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

####################################################################################
print('Start Learning!') ### 게임을 하면서, 학습을 하면서, policy를 업데이트 ##########
####################################################################################

# game_state: unknown, main, episode_menu, level_selection, loading, playing, won, lost

print('Learning with real episodes...')

for i_episode in range(num_episodes): # while 
	
	state = comm_get_state(s, silent=False)

	if state==UNKNOWN:
		print ("########################################################")
		print ("Unknown state")
		pass
	elif state==MAIN_MENU:
		print ("########################################################")
		print ("Main menu state")
		pass
	elif state==EPISODE_MENU:
		print ("########################################################")
		print ("Episode menu state")
		pass
	elif state==LEVEL_SELECTION:
		print ("########################################################")
		print ("Level selection state")

		print ("level is loaded")
	elif state==LOADING:
		print ("########################################################")
		print ("Loading state")
		pass
	elif state==WON:
		print ("########################################################")
		print ("Won state")
		# resater random level

	elif state==LOST:
		print ("########################################################")
		print ("Lost state")
		# restart random level 

	elif state==PLAYING:
		print ("########################################################")
		print ("Playing state")

		saver.save(tf.get_default_session(), checkpoint_path)

		for t in itertools.count(): # 이 에피소드가 끝날때까지
			
			if total_t % update_target_estimator_every == 0:
				print("Copying model parameters to target Q network...")
				copy_model_parameter()
			
			# state 받기? 샷 찍음, 

			print('Choose action from given Q network model')

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
