# -*- coding: utf-8 -*-

""" ref: https://www.youtube.com/watch?v=cO5g5qLrLSo"""

import gym
import random
import numpy as np

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten
from tensorflow.keras.optimizers import Adam

"""
To use the package, use
pip install keras-rl2
"""
from rl.agents import DQNAgent
from rl.policy import BoltzmannQPolicy
from rl.memory import SequentialMemory


def random_play_game(episodes:int):
    for i in range(1,episodes+1):
        env.reset()
        done = False
        score = 0
        
        while not done:
            env.render()
            action = random.choice([0,1])
            n_state, reward, done, info = env.step(action)
            score += reward
            print(score)
        print(f"Episode {i} score: {score}")
    
    env.close()

def bulid_model(states:int, actions:int):
    model = Sequential() #建立空的神經網路
    model.add(Flatten(input_shape = (1,states)))
    model.add(Dense(24, activation = 'relu'))
    model.add(Dense(24, activation = 'relu'))    
    model.add(Dense(actions, activation = 'linear'))
    return model

def bulid_agent(model, actions:int):
    policy = BoltzmannQPolicy()
    memory = SequentialMemory(limit = 50000, window_length =1)
    dqn = DQNAgent(model=model, memory=memory, policy=policy,
                   nb_actions=actions, nb_steps_warmup=10, target_model_update=0.01)
    return dqn

if __name__=='__main__':
    env = gym.make('CartPole-v0')
    states = env.observation_space.shape[0]
    actions = env.action_space.n
    
    model = bulid_model(states, actions)
    print(model.summary())
    dqn = bulid_agent(model, actions)
    dqn.compile(Adam(lr=0.001), metrics=['mae'])
    dqn.fit(env, nb_steps=50000, visualize=False, verbose=1)
    
    scores = dqn.test(env, nb_episodes=100, visualize=True)
    print(np.mean(scores.history['episode_reward']))
