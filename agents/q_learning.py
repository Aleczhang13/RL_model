from environments.environments import Maze
import numpy as np
import pandas as pd


class RL(object):
    # 初始化
    # actions为可选动作， learning_rate为学习率，reward_decay为传递奖励是的递减系数gamma，1-e_greed为随机选择其他动作的概率
    def __init__(self,actions, learning_rate=0.001,reward_decy=0.9,e_greedy = 0.9):
        self.actions = actions
        self.lr = learning_rate
        self.gamma = reward_decy
        self.epsilon = e_greedy
        # 初始化qtable，行为observation的state， 列为当前状态可以选择的action（对于所有列，可以选择的action一样）
        self.q_table = pd.DataFrame(columns=self.actions, dtype=np.float64)

    def choose_action(self, observation):
        self.check_state_exist(observation)  # 检查当前状态是否存在，不存在就添加这个状态

        if np.random.uniform() < self.epsilon:
            state_action = self.q_table.loc[observation, :]  # 找到当前状态可以选择的动作
            # 由于初始化或更新后一个状态下的动作值可能是相同的，为了避免每次都选择相同动作，用random.choice在值最大的action中损及选择一个
            action = np.random.choice(state_action[state_action == np.max(state_action)].index)

        else:
            action = np.random.choice(self.actions)  # 0.1的几率随机选择动作
        return action

    def check_state_exist(self, state):
        if state not in self.q_table.index:
            # 若找不到该obversation的转态，则添加该状态到新的qtable
            # 新的state的动作的q初始值赋值为0，列名为dataframe的列名，index为state
            self.q_table = self.q_table.append(
                pd.Series([0] * len(self.actions), index=self.q_table.columns, name=state))
            a = 3
    # 不同方式的学习方法不同，用可变参数，直接pass
    def learning(self, *args):
        pass


# QLearning继承RL
class QLearningTable(RL):
    # 初始化
    # 参数自己定义，含义继承父类RL
    # 类方法choose_action、check_state_exist自动继承RL，参数不变
    def __init__(self, actions, learning_rate=0.01, reward_decay=0.9, e_greedy=0.9):
        super(QLearningTable, self).__init__(actions, learning_rate, reward_decay, e_greedy)

    # 根绝当前观察状态s，选择动作a，选择动作后的奖励r，和执行动作后的状态s_，来更新qtable
    def learning(self, s, a, r, s_):
        self.check_state_exist(s_)  # 检查动作后状态s_是否存在

        q_old = self.q_table.loc[s, a]  # 旧的q[s,a]值

        if s_ != 'terminal':
            # 下个状态下最大的值
            max_s_ = self.q_table.loc[s_, :].max()
            q_new = r + self.gamma * max_s_  # 计算新的值
        else:
            q_new = r

        self.q_table.loc[s, a] = q_old - self.lr * (q_new - q_old)  # 根据更新公式更新，类似于梯度下降


def update():
    for episode in range(100):
        # 初始化 state 的观测值
        observation = env.reset()  # 每轮训练都要初始化观测值，即回到原点状态

        while True:
            env.render()

            # RL 根据 state 的观测值挑选 action
            action = RL.choose_action(str(observation))  # qlearning采用greeed方法，选择q值最大的action

            # 探索者在环境中实施这个 action, 并得到环境返回的下一个 state 观测值, reward 和 done (是否是掉下地狱或者升上天堂)
            # 是根据当前选择动作，观察到的采取动作后的状态和奖励
            observation_, reward, done = env.step(action)

            # RL 从这个序列 (state, action, reward, state_) 中学习
            # 根绝旧observation的q值，和采取动作，以及奖励和采取动作后的observation_的最大q值进行更新
            RL.learning(str(observation), action, reward, str(observation_))

            # 将下一个 state 的值传到下一次循环
            observation = observation_

            if done:
                break

    # 结束游戏并关闭窗口
    print('game over')
    env.destroy()


if __name__ == "__main__":
    # 定义环境 env 和 RL 方式
    env = Maze()
    RL = QLearningTable(actions=list(range(env.n_actions)))
    # 开始可视化环境 env
    env.after(100, update)
    env.mainloop()
