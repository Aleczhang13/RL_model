import pandas as pd
import numpy as np
from environments.maze import Maze

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


class SarasaTable(RL):
    def __init__(self,actions, learning_rate=0.01, reward_decay=0.9,e_greedy=0.9):
        super(SarasaTable,self).__init__(actions, learning_rate, reward_decay, e_greedy)

    def learning(self, s, a,r, s_, a_):
        self.check_state_exist(s_)

        q_old = self.q_table.loc[s,a]

        if s_ !="terminal":
            q_predict = self.q_table.loc[s_, a_]
            q_new = r+self.gamma*q_predict
        else:
            q_new = r

        self.q_table.loc[s,a] = q_old- self.lr * (q_new-q_old)


def update():
    for episode in range(100):
        # 初始化环境
        observation = env.reset()

        # 根据当前状态选行为
        action = RL.choose_action(str(observation))

        while True:
            # 刷新环境
            env.render()

            # 在环境中采取行为, 获得下一个 state_ (obervation_), reward, 和是否终止
            observation_, reward, done = env.step(action)

            # 根据observation_选择observation_下应该选择的动作action_
            action_ = RL.choose_action(str(observation_))

            # 从当前状态state，当前动作action，奖励r，执行动作后state_，state_下的action_,(s,a,r,s,a)
            RL.learning(str(observation), action, reward, str(observation_), action_)

            # 将下一个当成下一步的 state (observation) and action。
            # 与qlearning的却别是sarsa在observation_下真正执行了动作action_，供下次使用
            # 而qlearning中下次状态observation_时还要重新选择action_
            observation = observation_
            action = action_

            # 终止时跳出循环
            if done:
                break


    # 大循环完毕
    print('game over')
    env.destroy()

if __name__ == '__main__':
    env = Maze()

    # Sarsa和SarsaLambda的调用方式一模一样
    RL = SarasaTable(actions=list(range(env.n_actions)))

    env.after(100, update)
    env.mainloop()

