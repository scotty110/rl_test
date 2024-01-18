# https://github.com/openai/gym/issues/3201
import ale_py
# if using gymnasium
#import shimmy

import gymnasium as gym
import torch
import numpy as np
import torch.nn.functional as F
    
def convert_gray(obs:np.array) -> np.array: 
    return np.dot(obs[...,:3], [0.299, 0.587, 0.114])

class env():
    def __init__(self, env_str:str='ALE/SpaceInvaders-v5'):
        self.env = gym.make(env_str)
        self.obs = None
        self.reset()
        self.n_actions = self.env.action_space.n

    def random_action(self):
        return self.env.action_space.sample()
    
    def reset(self):
        obs = self.env.reset()
        self.obs = torch.tensor(convert_gray(obs))
        return

    def step(self, action):
        obs, reward, stop, _ = self.env.step(action)
        self.obs = torch.tensor(convert_gray(obs))
        to_return = (self.obs, action, reward)

        # All Tensor
        if stop:
            self.reset
        return to_return
    