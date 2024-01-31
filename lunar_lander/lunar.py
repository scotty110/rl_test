# https://github.com/openai/gym/issues/3201
import ale_py
import gymnasium as gym
import torch
from collections import deque


class env():
    def __init__(self, env_str='CartPole-v1', stack_size=4):
        self.env = gym.make(env_str)
        self.stack_size = stack_size
        self.obs = deque(maxlen=stack_size)
        self.total_reward = 0
        self.reset()
        self.n_actions = self.env.action_space.n

    def random_action(self):
        return self.env.action_space.sample()
    
    def reset(self):
        self.total_reward = 0
        obs, _ = self.env.reset()
        obs = torch.tensor(obs)
        for _ in range(self.stack_size):
            self.obs.append(obs)
        return self.get_stacked_obs()
    
    def step(self, action):
        obs, reward, done, _, _ = self.env.step(action)
        self.obs.append(torch.tensor(obs))
        stacked_obs = self.get_stacked_obs()
        self.total_reward += reward
        if done:
            self.reset()
        return stacked_obs, action, reward, done

    def get_stacked_obs(self):
        # Assuming the observations are images, stack along a new dimension
        return torch.stack(list(self.obs), dim=0) 

'''
if __name__ == '__main__':
    env = env()
    obs = env.reset()
    for i in range(10):
        obs, action, reward, done = env.step(env.random_action())
        print(obs.shape, action, reward, done)
        print(reward, done)
        if done:
            break
'''