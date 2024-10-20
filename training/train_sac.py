import sys
import os

import numpy as np
# 获取当前脚本文件夹的上一级目录并将项目根目录添加到系统路径
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(project_root)
import time
import gymnasium as gym
import matplotlib.pyplot as plt
from agents.sac_agent import SACAgent
from agents.ddpg_agent import DDPGAgent
import env  # 确保导入 env 模块以便注册环境

if __name__ == "__main__":
    # 训练参数：
    # np.seterr(over='warn')  # 或 over='raise' 以抛出异常

    num_episodes=150
    max_steps=1000
    render=False
    env = gym.make("BRC-v0")

    start_time = time.time() 
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0]
    # agent = SACAgent(state_dim, action_dim)
    agent = DDPGAgent(state_dim, action_dim)


    rewards = []

    for episode in range(num_episodes):
        state, info = env.reset()
        episode_reward = 0
        envdam = 0

        for t in range(max_steps):

            action = agent.select_action(state)
            next_state, reward, done, truncated,_ = env.step(action)
            if episode_reward >=40:
                envdam = 1
                break
            agent.store_transition(state, action, reward, next_state, done)
            agent.update()

            state = next_state
            episode_reward += reward

            if done or truncated:
                break

        if render:
            env.render()

        if envdam == 0:
            rewards.append(min(1000,episode_reward))
            print(f"Episode {episode + 1}, Reward: {episode_reward}")

    env.close()

    end_time = time.time() 
    elapsed_time = end_time - start_time  
    print(f"训练耗时: {elapsed_time:.2f} 秒")

    plt.plot(rewards)
    plt.xlabel("Episode")
    plt.ylabel("Reward")
    plt.show()