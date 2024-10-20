import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import random
from collections import deque
import gymnasium as gym
import matplotlib.pyplot as plt

# Actor 网络
class Actor(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_dim):
        super(Actor, self).__init__()
        self.fc1 = nn.Linear(state_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, action_dim)

    def forward(self, state):
        x = torch.relu(self.fc1(state))
        x = torch.relu(self.fc2(x))
        return torch.tanh(self.fc3(x))

# Critic 网络
class Critic(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_dim):
        super(Critic, self).__init__()
        self.fc1 = nn.Linear(state_dim + action_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, 1)

    def forward(self, state, action):
        x = torch.cat([state, action], 1)
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        return self.fc3(x)

# 回放池
class ReplayBuffer:
    def __init__(self, capacity):
        self.buffer = deque(maxlen=capacity)

    def add(self, state, action, reward, next_state, done):
        self.buffer.append((state, action, reward, next_state, done))

    def sample(self, batch_size):
        batch = random.sample(self.buffer, batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)
        return (
            torch.tensor(states, dtype=torch.float),
            torch.tensor(actions, dtype=torch.float),
            torch.tensor(rewards, dtype=torch.float),
            torch.tensor(next_states, dtype=torch.float),
            torch.tensor(dones, dtype=torch.float),
        )

    def size(self):
        return len(self.buffer)

# SAC Agent
class SACAgent:
    def __init__(self, state_dim, action_dim, hidden_dim=512, gamma=0.99, tau=0.005, alpha=0.2, buffer_size=1000000, batch_size=256, lr=3e-4):
        self.actor = Actor(state_dim, action_dim, hidden_dim).cuda()
        self.critic1 = Critic(state_dim, action_dim, hidden_dim).cuda()
        self.critic2 = Critic(state_dim, action_dim, hidden_dim).cuda()
        self.target_critic1 = Critic(state_dim, action_dim, hidden_dim).cuda()
        self.target_critic2 = Critic(state_dim, action_dim, hidden_dim).cuda()
        
        self.target_critic1.load_state_dict(self.critic1.state_dict())
        self.target_critic2.load_state_dict(self.critic2.state_dict())

        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=lr)
        self.critic_optimizer = optim.Adam(
            list(self.critic1.parameters()) + list(self.critic2.parameters()), lr=lr
        )

        self.buffer = ReplayBuffer(buffer_size)
        self.batch_size = batch_size
        self.gamma = gamma
        self.tau = tau
        self.alpha = alpha

    def select_action(self, state):
        state = torch.tensor(state, dtype=torch.float).cuda()  # 确保状态在GPU上
        action = self.actor(state).cpu().detach().numpy()  # 将动作移到CPU以进行numpy操作
        return action

    def update(self):
        if self.buffer.size() < self.batch_size:
            return

        states, actions, rewards, next_states, dones = self.buffer.sample(self.batch_size)

        # 移动到GPU
        states = states.cuda()
        actions = actions.cuda()
        rewards = rewards.cuda()
        next_states = next_states.cuda()
        dones = dones.cuda()

        # 计算目标值
        with torch.no_grad():
            next_actions = self.actor(next_states)
            next_Q1 = self.target_critic1(next_states, next_actions)
            next_Q2 = self.target_critic2(next_states, next_actions)
            next_Q = torch.min(next_Q1, next_Q2) - self.alpha * next_actions
            target_Q = rewards + (1 - dones) * self.gamma * next_Q

        # 更新 Critic
        current_Q1 = self.critic1(states, actions)
        current_Q2 = self.critic2(states, actions)
        critic_loss = ((current_Q1 - target_Q).pow(2) + (current_Q2 - target_Q).pow(2)).mean()
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()

        # 更新 Actor
        new_actions = self.actor(states)
        actor_loss = -(self.critic1(states, new_actions) - self.alpha * new_actions).mean()
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()

        # 软更新目标网络
        for target_param, param in zip(self.target_critic1.parameters(), self.critic1.parameters()):
            target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)
        for target_param, param in zip(self.target_critic2.parameters(), self.critic2.parameters()):
            target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)

    def store_transition(self, state, action, reward, next_state, done):
        self.buffer.add(state, action, reward, next_state, done)

# 主程序
if __name__ == '__main__':
    print(torch.cuda.is_available())
