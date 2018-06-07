import os
import tensorflow as tf

import pdb

class DQN_Estimator():
    def __init__(self, scope="estimator", output_size=None, summaries_dir=None, angle_as_state = 0):

        self.input_size = 4096 + angle_as_state
        self.hidden_size = [1024, 512]
        self.output_size = output_size
        self.learning_rate = 1e-4
        # self.min_delta = -3
        # self.max_delta = 3
        self.clip_delta = 1.0

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
        self.X = tf.placeholder(shape=[None, self.input_size], dtype=tf.float32, name='X') # [배치크기, feature size]
        # The TD target value
        self.Y = tf.placeholder(shape=[None], dtype=tf.float32, name='Y') # 각 state에서 얻을 수 있는 target reward 값
        # Integer id of which action was selected
        self.actions = tf.placeholder(shape=[None], dtype=tf.int32, name="actions")

        batch_size = tf.shape(self.X)[0]
        weights = {}
        # Neural network with 2 hidden layer
        W1 = tf.Variable(tf.random_normal([self.input_size, self.hidden_size[0]], stddev=0.01), name="W1")
        b1 = tf.Variable(tf.random_normal([self.hidden_size[0]], stddev=0.01), name="b1")
        L1 = tf.nn.relu(tf.matmul(self.X, W1)+b1)

        W2 = tf.Variable(tf.random_normal([self.hidden_size[0], self.hidden_size[1]], stddev=0.01), name="W2")
        b2 = tf.Variable(tf.random_normal([self.hidden_size[1]], stddev=0.01), name="b2")
        L2 = tf.nn.relu(tf.matmul(L1, W2)+b2)

        W3 = tf.Variable(tf.random_normal([self.hidden_size[1], self.output_size], stddev=0.01), name="W3")
        b3 = tf.Variable(tf.random_normal([self.output_size], stddev=0.01), name="b3")

        weights['W1']=W1
        weights['b1']=b1
        weights['L1']=L1
        weights['W2']=W2
        weights['b2']=b2
        weights['L2']=L2
        weights['W3']=W3
        weights['b3']=b3


        # duel
        if duel == True:
            v_W3 = tf.Variable(tf.random_normal([self.hidden_size[1], 1], stddev=0.01), name="v_W3")
            v_b3 = tf.Variable(tf.random_normal([1], stddev=0.01), name="v_b3")
            weights['v_W3']=v_W3
            weights['v_b3']=v_b3
            self.v = tf.matmul(L2, v_W3)+v_b3 # relu 거치지 않고, softmax를 함
            self.advantage = tf.matmul(L2, W3)+b3
            self.actions_q = self.v + (self.advantage - tf.reduce_mean(self.advantage, reduction_indices = 1, keepdims = True))
        else:
            self.actions_q = tf.matmul(L2, W3)+b3 # relu 거치지 않고, softmax를 함

        self.predictions = tf.nn.softmax(self.actions_q)
        # self.predictions = tf.contrib.layers.fully_connected(fc1, len(VALID_ACTIONS)) # weight인지, final output인지
        # 수정: 여기가 최종 아웃풋이 되면 되는건지 확인....
        # 여기서 tf.nn.softmax(tf.matmul(L2, W3)+b3)를 하지 않아도 되는지...

        # Get the predictions for the chosen actions only
        # gather_indices = tf.range(batch_size) * tf.shape(self.predictions)[1] + self.actions # ?? 찍어보면 좋을듯
        # self.action_predictions = tf.gather(tf.reshape(self.predictions, [-1]), gather_indices) # ??

        self.action_one_hot = tf.one_hot(self.actions, self.output_size, 1.0, 0.0, name='action_one_hot')
        predictions_of_chosen_action =  tf.reduce_sum(self.actions_q*self.action_one_hot, reduction_indices = 1)

        # self.delta = self.Y - self.predictions[self.actions] # 이런식으로 indexing이 안되는거 같음.....
        self.delta = self.Y - predictions_of_chosen_action
        # self.clipped_delta = tf.clip_by_value(self.delta, self.min_delta, self.max_delta, name="clipped_delta")

        # Calculate the loss
        # self.loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(logits=cost, labels=Y))
        # self.losses = tf.squared_difference(self.Y, self.action_predictions) # target_value랑 action?
        # self.loss = tf.reduce_mean(self.losses)
        self.loss = tf.reduce_mean(tf.square(self.delta), name="loss")

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
            tf.summary.histogram("q_values_hist", self.predictions),
            tf.summary.scalar("max_q_value", tf.reduce_max(self.predictions))
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
        return sess.run(self.predictions, {self.X: s})

    def update(self, sess, s, a, y):
        """
        Updates the estimator towards the given targets.

        Args:
          sess: Tensorflow session object
          s: State input of shape [batch_size, 4, 160, 160, 3]
          a: Chosen actions of shape [batch_size]
          y: Targets of shape [batch_size]

        Returns:
          The calculated loss on the batch.
        """
        # pdb.set_trace()
        # print('in update')
        # print()
        feed_dict = { self.X: s, self.actions: a, self.Y: y}
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
