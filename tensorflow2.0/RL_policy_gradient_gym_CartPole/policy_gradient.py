# -*- coding: utf-8 -*-

"""
ref: https://www.mdeditor.tw/pl/2e4d/zh-tw :TensorFlow 2.0 (九) – 強化學習 70行程式碼實戰 Policy Gradient

實測發現用上文的code非常容易訓練失敗，
查資料說policy gradient的loss函數通常用「-log(prob)*adventage」計算，
而非用'mean_squared_error'，
另可能可以參考另一篇文章改進: https://adventuresinmachinelearning.com/policy-gradient-tensorflow-2/
"""
import matplotlib.pyplot as plt
import gym
import numpy as np
from tensorflow.keras import models, layers, optimizers

def bulid_model():
    STATE_DIM, ACTION_DIM = 4, 2
    model = models.Sequential([
        layers.Dense(100, input_dim=STATE_DIM, activation='relu'),
        layers.Dropout(0.2), # 測試發現DropOut層設0.2似乎較容易訓練成功 
        layers.Dense(ACTION_DIM, activation="softmax")
    ])
    model.compile(loss='mean_squared_error',
                  optimizer=optimizers.Adam(0.001))
    print(model.summary())
    return model


def choose_action(s):
    """预测动作"""
    prob = model.predict(np.array([s]))[0]
    return np.random.choice(len(prob), p=prob)


def discount_rewards(rewards, gamma=0.95):
    """计算衰减reward的累加期望，并中心化和标准化处理"""
    prior = 0
    out = np.zeros_like(rewards)
    for i in reversed(range(len(rewards))):
        prior = prior * gamma + rewards[i]
        out[i] = prior
    return out / np.std(out - np.mean(out))


def train(records):
    s_batch = np.array([record[0] for record in records])
    # action 独热编码处理，方便求动作概率，即 prob_batch
    a_batch = np.array([[1 if record[1] == i else 0 for i in range(ACTION_DIM)]
                        for record in records])
    # 假设predict的概率是 [0.3, 0.7]，选择的动作是 [0, 1]
    # 则动作[0, 1]的概率等于 [0, 0.7] = [0.3, 0.7] * [0, 1]
    prob_batch = model.predict(s_batch) * a_batch
    r_batch = discount_rewards([record[2] for record in records])

    model.fit(s_batch, prob_batch, sample_weight=r_batch, verbose=0)

env = gym.make('CartPole-v0')
model = bulid_model()
episodes = 2000  # 至多2000次
score_list = []  # 记录所有分数
for i in range(episodes):
    s = env.reset()
    score = 0
    replay_records = []
    while True:
        a = choose_action(s)
        next_s, r, done, _ = env.step(a)
        replay_records.append((s, a, r))

        score += r
        s = next_s
        if done:
            train(replay_records)
            score_list.append(score)
            print('episode:', i, 'score:', score, 'max:', max(score_list))
            break
    # 最后10次的平均分大于 195 时，停止并保存模型
    if np.mean(score_list[-10:]) > 195:
        model.save('CartPole-v0-pg.h5')
        break
env.close()


# 画图
plt.plot(score_list)
x = np.array(range(len(score_list)))
smooth_func = np.poly1d(np.polyfit(x, score_list, 3))
plt.plot(x, smooth_func(x), label='Mean', linestyle='--')
plt.show()