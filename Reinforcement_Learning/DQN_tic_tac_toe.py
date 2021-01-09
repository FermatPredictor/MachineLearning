# -*- coding: utf-8 -*-
""" 
ref: https://mahowald.github.io/deep-tictactoe/
實際運行發現效果不佳
"""
import gym
from gym import spaces
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
from rl.policy import EpsGreedyQPolicy
from rl.memory import SequentialMemory

from Multi_agent import MultiAgent

class TicTacToe(gym.Env):
    reward_range = (-np.inf, np.inf)
    observation_space = spaces.MultiDiscrete([2 for _ in range(0, 3*3*3)])
    action_space = spaces.Discrete(9)
    
    """
    Board indexing:
    [ 0, 1, 2,
      3, 4, 5,
      6, 7, 8 ]
    """
    
    def _one_hot_board(self):
        one_hot_bd = np.zeros(27, dtype='int')
        for i in range(9):
            one_hot_bd[9*self.board[i]+i] = 1
        return one_hot_bd

#    def seed(self, seed=None):
#        pass
    
    def reset(self):
        self.current_player = 0
        self.board = np.zeros(9, dtype='int')
        self.winning_streaks = [[0,1,2],
                                [3,4,5],
                                [6,7,8],
                                [0,3,6],
                                [1,4,7],
                                [2,5,8],
                                [0,4,8],
                                [2,4,6]]
        return self._one_hot_board()
    
    def _rtn_state(self, reward, done, exp):
        # move to the next player
        self.current_player = 1 - self.current_player
        return self._one_hot_board(), reward, done, exp
        
    
    def step(self, action):
        exp = {"state": "in progress", "reason":""}
        
        reward = 0
        done = False

        # handle illegal moves
        if self.board[action] != 0:
            reward = -10 # illegal moves are really bad
            exp = {"state": "done", 
                   "reason":"Illegal move"}
            done = True
            return self._rtn_state(reward, done, exp)
        
        self.board[action] = self.current_player + 1
        
                 
        # check if we won
        #print('現在玩家',self.current_player)
        for streak in self.winning_streaks:
            #print(streak, self.board[streak])
            if (self.board[streak] == self.current_player + 1).all():
                reward = 1 # player wins!
                exp = {"state": "in progress", 
                       "reason": "{} has won".format(self.current_player)}
                done = True
                return self._rtn_state(reward, done, exp)
                
        # check if we tied
        if (self.board != 0).all():
            reward = 0
            exp = {"state": "in progress", 
                   "reason": "{} has tied".format(self.current_player)}
            done = True
            return self._rtn_state(reward, done, exp)
        
        # check if the other player can win on the next turn:
        for streak in self.winning_streaks:
            if ((self.board[streak] == 2 - self.current_player).sum() >= 2) \
                 and (self.board[streak] == 0).any():
                reward = -1
                exp = {
                "state": "in progress", 
                "reason": "{} can lose next turn".format(self.current_player)
                    }
                return self._rtn_state(reward, done, exp)
        
        return self._rtn_state(reward, done, exp)
    
    def render(self, mode="human"):
        print(self.board[0:3])
        print(self.board[3:6])
        print(self.board[6:9])
    
def bulid_model(states:int, actions:int):
    model = Sequential() #建立空的神經網路
    model.add(Flatten(input_shape = (1,states)))
    model.add(Dense(24, activation = 'relu'))
    model.add(Dense(24, activation = 'relu')) 
    model.add(Dense(24, activation = 'relu'))
    model.add(Dense(actions, activation = 'linear'))
    return model

def bulid_agent(model, actions:int):
    policy = EpsGreedyQPolicy(eps=0.2)
    memory = SequentialMemory(limit = 50000, window_length =1)
    dqn = DQNAgent(model=model, memory=memory, policy=policy,
                   nb_actions=actions, nb_steps_warmup=10, target_model_update=0.01)
    return dqn

if __name__=='__main__':
    env = TicTacToe()
    states = env.observation_space.shape[0]
    actions = env.action_space.n
    
    model = bulid_model(states, actions)
    print(model.summary())
    dqn = bulid_agent(model, actions)
    dqn.compile(Adam(lr=0.001), metrics=['mae'])
    agent = MultiAgent([dqn, dqn])
    agent.compile(Adam(lr=0.001), metrics=['mae'])
    agent.fit(env, nb_steps=100000, visualize=False, verbose=1)
    
    """
    Below is for testing
    """
    test_env = TicTacToe()
    done = False
    observation = test_env.reset()
    agent.training = False
    
    step = 0
    while not done:
        print("Turn: {}".format(step))
        action = agent.forward(observation)
        observation, reward, done, exp = test_env.step(action)
        test_env.render()
        print(exp['reason'])
        print("\n")
        step += 1
    
    print("A strange game. The only winning move is not to play.")



