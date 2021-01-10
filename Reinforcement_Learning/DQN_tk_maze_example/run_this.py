# -*- coding: utf-8 -*-

from maze_env import Maze
from DQN import DeepQNetwork

import time


def run_maze():
    step = 0
    for episode in range(300):
        # initial observation
        observation = env.reset()
        done = False

        while not done:
            # fresh env
            env.render()
                    

            # RL choose action based on observation
            action = RL.choose_action(observation)

            # RL take action and get next observation and reward
            observation_, reward, done = env.step(action)

            RL.store_transition(observation, action, reward, observation_)
            
            step += 1

            if (step > 200) and (step % 5 == 0):
                RL.learn()

            # swap observation
            observation = observation_

    # end of game
    print('game over')
    env.destroy()


if __name__ == "__main__":
    # maze game
    env = Maze()
    RL = DeepQNetwork(env.n_actions, env.n_features,
                      learning_rate=0.01,
                      reward_decay=0.9,
                      e_greedy=0.9,
                      replace_target_iter=200,
                      memory_size=2000,
                      # output_graph=True
                      )
    env.after(100, run_maze)
    env.mainloop()
    RL.plot_cost()