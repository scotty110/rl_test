# https://github.com/openai/gym/issues/3201
import ale_py
# if using gymnasium
#import shimmy

import gymnasium as gym
#env = gym.make("CartPole-v1")
#print(gym.envs.registry.keys())
import torch
import numpy as np
import torch.nn.functional as F
    
def convert_gray(obs:np.array) -> np.array: 
    return np.dot(obs[...,:3], [0.299, 0.587, 0.114])

def resize(obs:np.array) -> torch.Tensor:
    return (F.interpolate((torch.from_numpy(convert_gray(obs))).reshape([1,1,210,160]), size=(110,84) ))[:,:,15:99,:]

def save(name, np_array):
    from PIL import Image
    img = Image.fromarray(np.uint8(np_array), 'L')
    img.save(name+'.png')


class env():
    def __init__(self, tstep:int=5, gamma:float=0.9, env_str:str='ALE/SpaceInvaders-v5'):
        self.env = gym.make(env_str)
        self.gamma = gamma
        self.tstep = tstep
        self.obs = None
        self.next_obs = None
        self.reward = [0 for i in range(self.tstep)]
        self.reset()
        self.n_actions = self.env.action_space.n

    def random_action(self):
        return self.env.action_space.sample()
    
    def reset(self):
        obs, _ = self.env.reset()
        self.obs = torch.zeros(self.tstep, *(84,84) ) 
        cropped = resize(obs)
        self.obs[0] = cropped.squeeze(0)
        self.reward_window = [0. for i in range(self.tstep)] 
        self.reward = 0.

    def step(self, action):
        obs, reward, stop, _, _ = self.env.step(action)
        print(f'Reward: {reward}')
        f = lambda x: sum([v*(self.gamma**(i)) for i,v in enumerate(x)])
        to_return = (self.obs, action, f(self.reward_window))

        # All Tensor
        #aTensor = torch.cat((torch.from_numpy(convert_gray(obs)).unsqueeze(0), self.obs), dim=0 )
        aTensor = torch.cat(((resize(obs)).squeeze(0), self.obs), dim=0 )
        self.obs = aTensor[:-1,...]

        # Fix reward r_1 = r + gamma 
        #Will need to add gamma, but this might be right: V_t = R_t + V_(t+1)
        self.reqard = self.reward + reward
        self.reward_window = [*self.reward_window, self.reward][1:]
        if stop:
            self.reset
        return to_return
    

'''
if __name__ == '__main__':
    tester = env()

    t_list = []
    for i in range(10):
        action = tester.env.action_space.sample()
        t = tester.step(action)
        t_list.append(t)
#'''