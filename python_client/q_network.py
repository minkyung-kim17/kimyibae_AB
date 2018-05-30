import os
import tensorflow as tf

import pdb

class DQN_Estimator():
    def __init__(self, batch_size = 6, scope="estimator_default", angle_output_size=None, taptime_output_size = None, summaries_dir=None):

        self.input_size = 4096
        self.hidden_size = [1024, 512]
        self.angle_output_size = angle_output_size
        self.taptime_output_size = taptime_output_size
        self.learning_rate = 1e-4
        # self.min_delta = -3
        # self.max_delta = 3
        self.clip_delta = 1.0
        self.batch_size = batch_size

        self.scope = scope
        # Writes Tensorboard summaries to disk
        self.summary_writer = None
        with tf.variable_scope(scope):
            # Build the graph
            self._build_Qnetwork()
            if summaries_dir:
                summary_dir = os.path.join(summaries_dir, "summaries_{}".format(scope))
                if not os.path.exists(summary_dir):
                    os.makedirs(summary_dir)
                self.summary_writer = tf.summary.FileWriter(summary_dir)

    def _build_Qnetwork(self, duel = True):
        """
        Builds the Tensorflow graph.
        """

        # Placeholders for our Q-network
        # Our input are feature vectors of shape 4096 each
        self.X = tf.Variable(tf.zeros([self.batch_size, self.input_size], dtype=tf.float32), name='X') # [배치크기, feature size]
        # The TD target value
        self.angle_Y = tf.Variable(tf.zeros([self.batch_size], dtype=tf.float32), name='angle_Y') # 각 state에서 얻을 수 있는 target reward 값
        self.taptime_Y = tf.Variable(tf.zeros([self.batch_size], dtype=tf.float32), name='taptime_Y') # 각 state에서 얻을 수 있는 target reward 값
        # Integer id of which action was selected
        self.angle_actions = tf.Variable(tf.zeros([self.batch_size], dtype=tf.int32), name="angle_actions") # idx
        self.taptime_actions = tf.Variable(tf.zeros([self.batch_size], dtype=tf.int32), name="taptime_actions")

        # batch_size = tf.shape(self.X)[0]
        weights = {}
        # Neural network with 2 hidden layer
        W1 = tf.Variable(tf.random_normal([self.input_size, self.hidden_size[0]], stddev=0.01), name="W1")
        b1 = tf.Variable(tf.random_normal([self.hidden_size[0]], stddev=0.01), name="b1")
        L1 = tf.nn.relu(tf.matmul(self.X, W1)+b1)

        W2 = tf.Variable(tf.random_normal([self.hidden_size[0], self.hidden_size[1]], stddev=0.01), name="W2")
        b2 = tf.Variable(tf.random_normal([self.hidden_size[1]], stddev=0.01), name="b2")
        L2 = tf.nn.relu(tf.matmul(L1, W2)+b2)

        angle_W3 = tf.Variable(tf.random_normal([self.hidden_size[1], self.angle_output_size], stddev=0.01), name="angle_W3")
        angle_b3 = tf.Variable(tf.random_normal([self.angle_output_size], stddev=0.01), name="angle_b3")

        taptime_W3 = tf.Variable(tf.random_normal([self.hidden_size[1], self.taptime_output_size], stddev=0.01), name="taptime_W3")
        taptime_b3 = tf.Variable(tf.random_normal([self.taptime_output_size], stddev=0.01), name="taptime_b3")

        weights['W1']=W1
        weights['b1']=b1
        weights['L1']=L1
        weights['W2']=W2
        weights['b2']=b2
        weights['L2']=L2
        weights['angle_W3']=angle_W3
        weights['angle_b3']=angle_b3
        weights['taptime_W3']=taptime_W3
        weights['taptime_b3']=taptime_b3


        # duel
        if duel == True:
            angle_v_W3 = tf.Variable(tf.random_normal([self.hidden_size[1], 1], stddev=0.01), name="angle_v_W3")
            angle_v_b3 = tf.Variable(tf.random_normal([1], stddev=0.01), name="angle_v_b3")
            weights['angle_v_W3']=angle_v_W3
            weights['angle_v_b3']=angle_v_b3
            self.angle_v = tf.matmul(L2, angle_v_W3)+angle_v_b3 # relu 거치지 않고, softmax를 함
            self.angle_advantage = tf.matmul(L2, angle_W3)+angle_b3
            self.angle_actions_q = self.angle_v + (self.angle_advantage - tf.reduce_mean(self.angle_advantage, reduction_indices = 1, keepdims = True))

            taptime_v_W3 = tf.Variable(tf.random_normal([self.hidden_size[1], 1], stddev=0.01), name="taptime_v_W3")
            taptime_v_b3 = tf.Variable(tf.random_normal([1], stddev=0.01), name="taptime_v_b3")
            weights['taptime_v_W3']=taptime_v_W3
            weights['taptime_v_b3']=taptime_v_b3
            self.taptime_v = tf.matmul(L2, taptime_v_W3)+taptime_v_b3 # relu 거치지 않고, softmax를 함
            self.taptime_advantage = tf.matmul(L2, taptime_W3)+taptime_b3
            self.taptime_actions_q = self.taptime_v + (self.taptime_advantage - tf.reduce_mean(self.taptime_advantage, reduction_indices = 1, keepdims = True))
        else:
            self.angle_actions_q = tf.matmul(L2, angle_W3)+angle_b3 # relu 거치지 않고, softmax를 함
            self.taptime_actions_q = tf.matmul(L2, taptime_W3)+taptime_b3 # relu 거치지 않고, softmax를 함

        self.angle_predictions = tf.nn.softmax(self.angle_actions_q)
        self.taptime_predictions = tf.nn.softmax(self.taptime_actions_q)
        # self.angle_predictions = tf.contrib.layers.fully_connected(fc1, len(VALID_ACTIONS)) # weight인지, final output인지
        # 수정: 여기가 최종 아웃풋이 되면 되는건지 확인....
        # 여기서 tf.nn.softmax(tf.matmul(L2, W3)+b3)를 하지 않아도 되는지...

        # Get the predictions for the chosen actions only
        # gather_indices = tf.range(batch_size) * tf.shape(self.angle_predictions)[1] + self.angle_actions # ?? 찍어보면 좋을듯
        # self.action_predictions = tf.gather(tf.reshape(self.angle_predictions, [-1]), gather_indices) # ??

        self.angle_action_one_hot = tf.one_hot(self.angle_actions, self.angle_output_size, 1.0, 0.0, name='angle_action_one_hot')
        predictions_of_chosen_angle_action =  tf.reduce_sum(self.angle_actions_q*self.angle_action_one_hot, reduction_indices = 1)

        self.taptime_action_one_hot = tf.one_hot(self.taptime_actions, self.taptime_output_size, 1.0, 0.0, name='taptime_action_one_hot')
        predictions_of_chosen_taptime_action =  tf.reduce_sum(self.taptime_actions_q*self.taptime_action_one_hot, reduction_indices = 1)
        # self.angle_delta = self.angle_Y - self.angle_predictions[self.angle_actions] # 이런식으로 indexing이 안되는거 같음.....
        self.angle_delta = self.angle_Y - predictions_of_chosen_angle_action
        self.taptime_delta = self.taptime_Y - predictions_of_chosen_taptime_action
        # self.clipped_delta = tf.clip_by_value(self.angle_delta, self.min_delta, self.max_delta, name="clipped_delta")

        # Calculate the loss
        # self.loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(logits=cost, labels=Y))
        # self.losses = tf.squared_difference(self.Y, self.action_predictions) # target_value랑 action?
        # self.loss = tf.reduce_mean(self.losses)
        angle_loss = tf.reduce_mean(tf.square(self.angle_delta))
        taptime_loss = tf.reduce_mean(tf.square(self.taptime_delta))
        self.loss = tf.Variable(angle_loss+taptime_loss, name = "loss")
        # Optimizer Parameters from original paper
        # self.optimizer = tf.train.RMSPropOptimizer(0.00025, 0.99, 0.0, 1e-6)
        # self.train_op = self.optimizer.minimize(self.loss, global_step=tf.train.get_global_step())
        self.optimizer = tf.train.AdamOptimizer( learning_rate = self.learning_rate )
        # self.train_op = self.optimizer.minimize(self.loss)

        # gvs = optimizer.compute_gradients(self.loss)
        # capped_gvs = [(tf.clip_by_value(grad, -1., 1.), var) for grad, var in gvs]
        # self.train_op = optimizer.apply_gradients(capped_gvs, global_step=tf.train.get_global_step())

        gradients, variables = zip(*self.optimizer.compute_gradients(self.loss))
        gradients, _ = tf.clip_by_global_norm(gradients, self.clip_delta)
        self.train_op = self.optimizer.apply_gradients(zip(gradients, variables), global_step=tf.train.get_global_step())

        # self.train_op = self.optimizer.minimize(self.loss, global_step=tf.train.get_global_step())
        # self.train_op = self.optimizer.minimize(self.loss, var_list=list(self.q_w.values()), global_step=self.global_step)
        # self.optimizer = tf.train.AdamOptimizer(0.001).minimize(loss)

        # Summaries for Tensorboard
        self.summaries = tf.summary.merge([
            tf.summary.scalar("loss", self.loss),
            tf.summary.histogram("angle_q_values_hist", self.angle_predictions),
            tf.summary.scalar("angle_max_q_value", tf.reduce_max(self.angle_predictions)),
            tf.summary.histogram("taptime_q_values_hist", self.taptime_predictions),
            tf.summary.scalar("taptime_max_q_value", tf.reduce_max(self.taptime_predictions))
        ])
        return weights

    def predict(self, sess, s):
        """
        Predicts action values.

        Args:
          sess: Tensorflow session
          s: State input of shape [batch_size, 4, 160, 160, 3]

        Returns:
          Tensor of shape [batch_size, NUM_VALID_ACTIONS] containing the estimated
          action values.
        """
        # pdb.set_trace()
        return sess.run([self.angle_predictions, self.taptime_predictions], {self.X: s})

    def update(self, sess, s, angle_a, taptime_a, angle_y, taptime_y):
        """
        Updates the estimator towards the given targets.

        Args:p          ses: Tensorflow session object
          s: State input of shape [batch_size, 4, 160, 160, 3]
          a: Chosen actions of shape [batch_size]
          y: Targets of shape [batch_size]

        Returns:
          The calculated loss on the batch.
        """
        # pdb.set_trace()
        # print('in update')
        # print()
        feed_dict = { self.X: s, self.angle_actions: angle_a, self.taptime_actions: taptime_a, self.angle_Y: angle_y, self.taptime_Y: taptime_y}
        # summaries, global_step, _, loss = sess.run(
        #     [self.summaries, tf.train.get_global_step(), self.train_op, self.loss],
        #     feed_dict)
        summaries, _, loss = sess.run(
            [self.summaries, self.train_op, self.loss],
            feed_dict)
        # summaries, _, loss = sess.run(
            # [self.summaries, self.train_op, self.loss],
            # feed_dict)
        if self.summary_writer:
            # self.summary_writer.add_summary(summaries, global_step)
            self.summary_writer.add_summary(summaries)
        return loss
