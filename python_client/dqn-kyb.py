import tensorflow as tf

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

# path 설정 ...
# current_path = 
# current_dir = 
# checkpoint_dir = 

# server랑 연결
print('Initialize connection to ABServer')

# DQN을 따라 replay memory를 모음? 아니면, 일단 cold start라 하고, 
# print('Populating replay memory...')


replay_memory = [] 

saver = tf.train.Saver()

latest_checkpoint = tf.train.latest_checkpoint(checkpoint_dir)
if latest_checkpoint:
	print("Loading model checkpoint {}...\n".format(latest_checkpoint))
	saver.restore(sess, latest_checkpoint)

total_t = sess.run(tf.train.get_global_step()) # ... ?

# epsilons = ?? 

# policy = ?? 

print('Populating replay memory...')
# replay memory 만들기



####################################################################################
print('Start Learning!') ### 게임을 하면서, 학습을 하면서, policy를 업데이트 ##########
####################################################################################

print('Learning with real episodes...')
for i_episode in range(num_episodes):
	saver.save(tf.get_default_session(), checkpoint_path)

	# episode의 시작 state

	for t in itertools.count(): # 이 에피소드가 끝날때까지
		if total_t % update_target_estimator_every == 0:
			print("Copying model parameters to target Q network...")
			copy_model_parameter()

		print('Choose action from given Q network model')
		# 실제 고른 action하고, reward 받고, replay memory에 저장

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

        if done:
            break

        state = next_state
        total_t += 1

    print('\nOne episode is done')


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