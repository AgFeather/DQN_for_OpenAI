import gym
from RL_Model import DQN
import matplotlib.pyplot as plt
import tensorflow as tf
import numpy as np

env = gym.make('Enduro-ram-v0') #导入环境

env = env.unwrapped
action_space = env.action_space
observation_space = env.observation_space
observation_space_high = env.observation_space.high
observation_space_low = env.observation_space.low

print(action_space)
print(observation_space)

env.seed(21)
MEMORY_SIZE = 10000

sess = tf.Session()

with tf.variable_scope('DQN_with_prioritized_replay'):
    RL_Model = DQN(n_actions=action_space.n, n_features=observation_space.shape[0], memory_size=MEMORY_SIZE,
        e_greedy_increment=0.00005, sess=sess, prioritized=True, output_graph=True,)
sess.run(tf.global_variables_initializer())


def train(RL):
    total_steps = 0
    steps = []
    episodes = []
    for i_episode in range(30):
        observation = env.reset() #获取环境初始state对应的observation
        while True:
            if total_steps % 10 == 0: env.render() #刷新环境并显示

            action = RL.choose_action(observation)

            observation_, reward, done, info = env.step(action) #获取下一个state

            if done: reward = 10

            RL.store_transition(observation, action, reward, observation_)

            if total_steps > MEMORY_SIZE:
                RL.learn()

            if done:
                print('episode ', i_episode, ' finished')
                steps.append(total_steps)
                episodes.append(i_episode)
                break

            observation = observation_
            total_steps += 1
    return np.vstack((episodes, steps))


if __name__ == '__main__':

    train(RL_Model)
# plt.plot(his_prio[0, :], his_prio[1, :] - his_prio[1, 0], c='r', label='DQN with prioritized replay')
# plt.legend(loc='best')
# plt.ylabel('total training time')
# plt.xlabel('episode')
# plt.grid()
# plt.show()
    pass
