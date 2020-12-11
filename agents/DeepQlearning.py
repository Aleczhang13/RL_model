import numpy as np
import pandas as pd
from environments.maze_env import Maze
import tensorflow as tf
class DeepQNetwork(object):
    def __init__(self, n_actions,n_features, learning_rate=0.001, reward_decay=0.9,e_greedy=0.9,replace_target_iter=300, memory_size=500,batch_size=32,e_greedy_increment=None,output_graph=False):
        self.n_actions = n_actions
        self.n_features = n_features
        self.lr = learning_rate
        self.gamma = reward_decay
        self.epsilon_max = e_greedy
        self.replace_target_iter = replace_target_iter
        self.memory_size = memory_size # 记忆上线
        self.batch_size = batch_size
        self.epsilon_increment = e_greedy_increment

        self.epslion = 0 if e_greedy_increment is not None else self.epslion_max

        # 记录相应的学习的次数，判断是否需要进行target_net参数
        self.learn_step_counter = 0
        # 初始化全 0 记忆 [s, a, r, s_]， 实际上feature为状态的维度，n_features*2分别记录s和s_，+2记录a和r
        self.memory= np.zeros((self.memory_size, n_features*2+2))
        self._build_net()

        # 替换相应的target net的参数
        t_params = tf.get_collection("target_net_params")
        e_params = tf.get_collection("eval_net_params")

        self.replace_target_op = [tf.assign(t, e) for t,e in zip(t_params, e_params)]

        self.session = tf.Session()
        if output_graph:
            tf.summary.FileWriter("logs/", self.session.graph)

        self.session.run(tf.global_variables_initializer())
        self.cost_his = []

    #建立相应的menory
    def store_transition(self, s, a, r, s_):
        if not hasattr(self, "memory_counter"):
            self.memory_counter = 0

        transition = np.hstack((s,[a, r], s_))

        # 总 memory 大小是固定的, 如果超出总大小, 旧 memory 就被新 memory 替换
        index = self.memory_counter % self.memory_size
        self.memory[index,:] = transition

        self.memory_counter += 1

    def _build_net(self):
        tf.reset_default_graph() # 清空计算图

        self.s = tf.placeholder(tf.float32, [None, self.n_features], name = "s")
        self.q_target = tf.placeholder(tf.float32, [ None, self.n_actions], name = "Q_target")

        with tf.variable_scope("eval_net"):
            c_names = ["eval_net_params", tf.GraphKeys.GLOBAL_VARIABLES]
            n_l1 = 10
            w_initializer = tf.random_normal_initializer(0,0.3)
            b_initializer = tf.constant_initializer(0.1)

            with tf.variable_scope("l1"):
                w1 = tf.get_variable("w1", [self.n_features, n_l1], initializer=w_initializer,collections=c_names)
                b1 = tf.get_variable('b1', [1, n_l1], initializer=b_initializer, collections=c_names)
                l1 = tf.nn.relu(tf.matmul(self.s, w1) + b1)

            # eval_network第二层全连接神经网络
            with tf.variable_scope('l1'):
                w2 = tf.get_variable('w2', [n_l1, self.n_actions], initializer=w_initializer, collections=c_names)
                b2 = tf.get_variable('b2', [1, self.n_actions], initializer=b_initializer, collections=c_names)
                # 求出q估计值，长度为n_actions的向量
                self.q_eval = tf.matmul(l1, w2) + b2

        # 创建target network，输入选择一个action后的状态s_,输出q_target
        self.s_ = tf.placeholder(tf.float32, [None, self.n_features], name='s_')  # 接收下个 observation
        with tf.variable_scope('target_net'):
            c_names = ['target_net_params', tf.GraphKeys.GLOBAL_VARIABLES]

            # target_net 的第一层fc， collections 是在更新 target_net 参数时会用到
            with tf.variable_scope('l1'):
                w1 = tf.get_variable('w1', [self.n_features, n_l1], initializer=w_initializer,
                                     collections=c_names)
                b1 = tf.get_variable('b1', [1, n_l1], initializer=b_initializer, collections=c_names)
                l1 = tf.nn.relu(tf.matmul(self.s_, w1) + b1)

            # target_net 的第二层fc
            with tf.variable_scope('l2'):
                w2 = tf.get_variable('w2', [n_l1, self.n_actions], initializer=w_initializer,
                                     collections=c_names)
                b2 = tf.get_variable('b2', [1, self.n_actions], initializer=b_initializer, collections=c_names)
                # 申请网络输出
                self.q_next = tf.matmul(l1, w2) + b2

        with tf.variable_scope('loss'):  # 求误差
            # 使用平方误差
            self.loss = tf.reduce_mean(tf.squared_difference(self.q_target, self.q_eval))

        with tf.variable_scope("train"):
            optimizer = tf.train.RMSPropOptimizer(self.lr)
            self._train_op = optimizer.minimize(self.loss)



    def choose_action(self, observation):
        # 根据observation（state）选行为
        # 使用eval network选出state下的行为估计
        # 将observation的shape变为(1, size_of_observation)，行向量变为列向量才能与NN维度统一
        observation = observation[np.newaxis, :]

        if np.random.uniform() < self.epslion:
            action_value = self.session.run(self.q_eval, feed_dict = {self.s: observation})
            action = np.argmax(action_value)

        else:
            action = np.random.randint(0, self.n_actions)

        return action

    def learn(self):
        if self.learn_step_counter% self.replace_target_iter ==0:
            self.session.run(self.replace_target_op)
            print("\n target params replaced")

        if self.memory_counter > self.memory_size:
            sample_index = np.random.choice(self.memory_size, size = self.batch_size)
        else:
            sample_index = np.random.choice(self.memory_counter, size = self.batch_size)

        batch_memory = self.memory[sample_index,:]
        # 获取q_next即q现实(target_net产生的q)和q_eval(eval_net产生的q)
        # q_next和q_eval都是一个向量，包含了对应状态下所有动作的q值
        # 实际上feature为状态的维度，batch_memory[:, -self.n_features:]为s_,即状态s采取动作action后的状态s_, batch_memory[:, :self.n_features]为s
        q_next, q_eval = self.session.run([self.q_next, self.q_eval],feed_dict={self.s_: batch_memory[:, -self.n_features:],self.s: batch_memory[:, :self.n_features]})

        q_target = q_eval.copy()
        # 每个样本下标
        batch_index = np.arange(self.batch_size, dtype=np.int32)
        # 记录每个样本在st时刻执行的动作
        eval_act_index = batch_memory[:, self.n_features].astype(int)
        # 记录每个样本动作的奖励
        reward = batch_memory[:, self.n_features + 1]

        # 生成每个样本中q值对应动作的更新，即生成的q现实，
        q_target[batch_index, eval_act_index] = reward + self.gamma * np.max(q_next, axis=1)

        # 假如在这个 batch 中, 我们有2个提取的记忆, 根据每个记忆可以生产3个 action 的值:
        # q_eval =[[1, 2, 3],[4, 5, 6]]， 另q_target = q_eval.copy()

        # 然后根据memory当中的具体action位置来修改 q_target 对应 action 上的值:
        # 比如在:记忆 0的q_target计算值是 -1,而且我用了action0;忆1的q_targe 计算值是-2, 而且我用了action2:
        # q_target =[[-1, 2, 3],[4, 5, -2]]
        # 所以 (q_target - q_eval) 就变成了:[[(-1)-(1), 0, 0],[0, 0, (-2)-(6)]]
        # 最后我们将这个 (q_target - q_eval) 当成误差, 反向传递会神经网络
        # 所有为 0 的 action 值是当时没有选择的 action, 之前有选择的 action 才有不为0的值.
        # 我们只反向传递之前选择的 action 的值,
        _, self.cost = self.session.run([self._train_op, self.loss],
                                     feed_dict={self.s: batch_memory[:, :self.n_features], self.q_target: q_target})

        self.cost_his.append(self.cost)  # 记录 cost 误差

        # 每调用一次learn，降低一次epsilon，即行为随机性
        self.epsilon = self.epsilon + self.epsilon_increment if self.epsilon < self.epsilon_max else self.epsilon_max

        self.learn_step_counter += 1

        def plot_cost(self):
            import matplotlib.pyplot as plt
            plt.plot(np.arange(len(self.cost_his)),self.cost_his)
            plt.ylabel("Cost")
            plt.xlabel("training steps")
            plt.show()


def run_maze():
    pass








