import os 
import tensorflow as tf

class DQN_Estimator():
    def __init__(self, scope="estimator", output_size=None, summaries_dir=None):
        
        self.input_size = 4096
        self.hidden_size = [1024, 512]
        self.output_size = output_size

        self.scope = scope
        # Writes Tensorboard summaries to disk
        self.summary_writer = None
        with tf.variable_scope(scope):
            # Build the graph
            self._build_Qnetwork()
            if summaries_dir: # 일단 이부분은 제낌
                summary_dir = os.path.join(summaries_dir, "summaries_{}".format(scope))
                if not os.path.exists(summary_dir):
                    os.makedirs(summary_dir)
                self.summary_writer = tf.summary.FileWriter(summary_dir)

    def _build_Qnetwork(self):
        """
        Builds the Tensorflow graph.
        """

        # Placeholders for our Q-network
        # Our input are feature vectors of shape 4096 each
        self.X = tf.placeholder(shape=[None, self.input_size], dtype=tf.float32, name='X') # [배치크기, feature size]
        # The TD target value
        self.Y = tf.placeholder(shape=[None, self.output_size], dtype=tf.float32, name='Y') # 각 state에서 얻을 수 있는 target reward 값
        # Integer id of which action was selected
        self.actions = tf.placeholder(shape=[None], dtype=tf.int32, name="actions")
        
        batch_size = tf.shape(self.X)[0] 

        # Neural network with 2 hidden layer
        W1 = tf.Variable(tf.random_normal([self.input_size, self.hidden_size[0]], stddev=0.01))
        b1 = tf.Variable(tf.random_normal([self.hidden_size[0]], stddev=0.01))
        L1 = tf.nn.relu(tf.matmul(self.X, W1)+b1)

        W2 = tf.Variable(tf.random_normal([self.hidden_size[0], self.hidden_size[1]], stddev=0.01))
        b2 = tf.Variable(tf.random_normal([self.hidden_size[1]], stddev=0.01))
        L2 = tf.nn.relu(tf.matmul(L1, W2)+b2)

        W3 = tf.Variable(tf.random_normal([self.hidden_size[1], self.output_size], stddev=0.01))
        b3 = tf.Variable(tf.random_normal([self.output_size], stddev=0.01))
        self.predictions = tf.matmul(L2, W3)+b3 # relu 거치지 않고, softmax를 함
        # self.predictions = tf.contrib.layers.fully_connected(fc1, len(VALID_ACTIONS)) # weight인지, final output인지

        # Get the predictions for the chosen actions only
        gather_indices = tf.range(batch_size) * tf.shape(self.predictions)[1] + self.actions # ?? 찍어보면 좋을듯
        self.action_predictions = tf.gather(tf.reshape(self.predictions, [-1]), gather_indices) # ?? 

        # Calculate the loss
        # self.loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(logits=cost, labels=Y))
        self.losses = tf.squared_difference(self.Y, self.action_predictions) # target_value랑 action?
        self.loss = tf.reduce_mean(self.losses)

        # Optimizer Parameters from original paper
        self.optimizer = tf.train.RMSPropOptimizer(0.00025, 0.99, 0.0, 1e-6)
        self.train_op = self.optimizer.minimize(self.loss, global_step=tf.train.get_global_step())
        # self.optimizer = tf.train.AdamOptimizer(0.001).minimize(loss)

        # Summaries for Tensorboard
        self.summaries = tf.summary.merge([
            tf.summary.scalar("loss", self.loss),
            tf.summary.histogram("loss_hist", self.losses),
            tf.summary.histogram("q_values_hist", self.predictions),
            tf.summary.scalar("max_q_value", tf.reduce_max(self.predictions))
        ])

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
        feed_dict = { self.X: s, self.Y: y, self.actions: a }
        summaries, global_step, _, loss = sess.run(
            [self.summaries, tf.train.get_global_step(), self.train_op, self.loss],
            feed_dict)
        if self.summary_writer:
            self.summary_writer.add_summary(summaries, global_step)
        return loss
