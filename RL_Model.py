
import numpy as np
import tensorflow as tf

from SumTree import  SumTree
from Memory import Memory

np.random.seed(1)
tf.set_random_seed(1)




class DQN(object):
    '''
    DQN结构
    '''
    def __init__(
            self,
            n_actions, # 动作数量
            n_features, # 每个state所有observation的数量
            learning_rate=0.005,
            reward_decay=0.9, # gamma，奖励衰减值
            e_greedy=0.9, # 贪婪值，用来决定是使用贪婪模式还是随机模式
            replace_target_iter=500, # Target_Net更行轮次
            memory_size=10000, # 记忆库大小
            batch_size=32,
            e_greedy_increment=None,
            output_graph=False,
            prioritized=True, # 是否使用优先记忆
            sess=None,):
        self.n_actions = n_actions
        self.n_features = n_features
        self.lr = learning_rate
        self.gamma = reward_decay
        self.epsilon_max = e_greedy
        self.target_net_update_period = replace_target_iter
        self.memory_size = memory_size
        self.batch_size = batch_size
        self.epsilon_increment = e_greedy_increment
        self.epsilon = 0 if e_greedy_increment is not None else self.epsilon_max

        self.prioritized = prioritized # 是否是用优先级记忆

        self.global_step_counter = 0

        self._build_net()

        t_params = tf.get_collection('target_net_params')
        q_params = tf.get_collection('q_net_params')
        self.update_target_net = [tf.assign(t, e) for t, e in zip(t_params, q_params)]

        if self.prioritized: # 使用SumTree
            self.memory = Memory(capacity=memory_size) # 构建一个容量为memory size的记忆库
        else:  # 不使用优先级记忆策略，用一个numpy数组表示记忆
            self.memory = np.zeros((self.memory_size, n_features*2+2))

        if sess is None:
            self.sess = tf.Session()
            self.sess.run(tf.global_variables_initializer())
        else:
            self.sess = sess

        if output_graph:
            tf.summary.FileWriter("logs/", self.sess.graph)

        self.cost_his = []

    def _build_net(self):
        '''
        创建两个神经网络
        :return:
        '''
        self.input_state = tf.placeholder(tf.float32, [None, self.n_features], name='input_state')
        self.output_target = tf.placeholder(tf.float32, [None, self.n_actions], name='output_target')
        self.input_weights = tf.placeholder(tf.float32, [None, 1], name='IS_weights') # 每个训练数据在计算loss时的权重

        # 构建Q_Net
        with tf.variable_scope('q_net'):
            c_names, n_l1, w_initializer, b_initializer = \
                ['q_net_params', tf.GraphKeys.GLOBAL_VARIABLES], 20, \
                tf.random_normal_initializer(0., 0.3), tf.constant_initializer(0.1)
            self.q_eval = self.build_layers(self.input_state, c_names, n_l1, w_initializer, b_initializer, True)
        # 构建Q_Net的训练loss
        with tf.variable_scope('loss'):
            self.abs_errors = tf.reduce_sum(tf.abs(self.output_target - self.q_eval), axis=1)
            self.loss = tf.reduce_mean(self.input_weights * tf.squared_difference(self.output_target, self.q_eval))
        # 构建Q_Net 的训练操作
        with tf.variable_scope('train'):
            self._train_op = tf.train.RMSPropOptimizer(self.lr).minimize(self.loss)

        # 初始化并构建Target_Net
        self.input_state_ = tf.placeholder(tf.float32, [None, self.n_features], name='s_')
        with tf.variable_scope('target_net'): # 构建Target_Net
            c_names = ['target_net_params', tf.GraphKeys.GLOBAL_VARIABLES]
            self.q_next = self.build_layers(self.input_state_, c_names, n_l1, w_initializer, b_initializer, False)


    def build_layers(self, s, c_names, n_l1, w_initializer, b_initializer, trainable):
        '''构建一个双隐藏层神经网络'''
        with tf.variable_scope('l1'):
            w1 = tf.get_variable('w1', [self.n_features, n_l1], initializer=w_initializer,
                                 collections=c_names, trainable=trainable)
            b1 = tf.get_variable('b1', [1, n_l1], initializer=b_initializer,
                                 collections=c_names, trainable=trainable)
            l1 = tf.nn.relu(tf.matmul(s, w1) + b1)

        with tf.variable_scope('l2'):
            w2 = tf.get_variable('w2', [n_l1, self.n_actions], initializer=w_initializer,
                                 collections=c_names, trainable=trainable)
            b2 = tf.get_variable('b2', [1, self.n_actions], initializer=b_initializer,
                                 collections=c_names, trainable=trainable)
            out = tf.matmul(l1, w2) + b2
        return out


    # 将从环境中获得的记忆数据存储到DQN的记忆库中
    def store_transition(self, s, a, r, s_):
        transition = np.hstack((s, [a, r], s_)) # 将数据转换成array
        self.memory.store(transition)

    def choose_action(self, observation):
        '''根据输入的state选择行为，90%的概率选择最优行为，10%概率随机'''
        observation = observation[np.newaxis, :]
        if np.random.uniform() < self.epsilon:
            actions_value = self.sess.run(self.q_eval, feed_dict={self.input_state: observation})
            action = np.argmax(actions_value)
        else:
            action = np.random.randint(0, self.n_actions)
        return action

    def learn(self):
        if self.global_step_counter % self.target_net_update_period == 0:
            self.sess.run(self.update_target_net)

        tree_idx, batch_memory, memory_weights = self.memory.sample(self.batch_size)

        feed = {self.input_state_: batch_memory[:, -self.n_features:],
                           self.input_state: batch_memory[:, :self.n_features]}
        q_next, q_eval = self.sess.run([self.q_next, self.q_eval], feed_dict=feed) # 正向传播Q_Net和Target_Net

        # 只计算被选择的ation对应的loss，其他action产生的loss记为0
        output_target = q_eval.copy()
        batch_index = np.arange(self.batch_size, dtype=np.int32)
        eval_act_index = batch_memory[:, self.n_features].astype(int)
        reward = batch_memory[:, self.n_features + 1]
        output_target[batch_index, eval_act_index] = reward + self.gamma * np.max(q_next, axis=1)

        # 获得本次训练的loss，进而将其更新到SumTree中
        feed = {self.input_state: batch_memory[:, :self.n_features],
                    self.output_target: output_target, self.input_weights: memory_weights}
        _, abs_errors, self.cost = self.sess.run([self._train_op, self.abs_errors, self.loss],
                                     feed_dict=feed)
        self.memory.batch_update(tree_idx, abs_errors)     # 将训练过的记忆数据更新到SumTree中

        self.cost_his.append(self.cost)

        self.epsilon = self.epsilon + self.epsilon_increment if self.epsilon < self.epsilon_max else self.epsilon_max
        self.global_step_counter += 1
