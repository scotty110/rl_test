# https://github.com/openai/gym/issues/3201
import ale_py
# if using gymnasium
#import shimmy

import gymnasium as gym
#env = gym.make("CartPole-v1")
#print(gym.envs.registry.keys())
import torch
import numpy as np
    
def convert_gray(obs:np.array) -> np.array: 
    return np.dot(obs[...,:3], [0.299, 0.587, 0.114])

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
        #print(f'init reward: {self.reward}')

    def reset(self):
        obs, _ = self.env.reset()
        obs = convert_gray(obs)
        self.obs = torch.zeros(self.tstep, *obs.shape ) 
        self.obs[0] = torch.from_numpy(obs) 

        self.reward = [0 for i in range(self.tstep)] 
        #save('test_1', self.obs[0])

    def step(self, action):
        obs, reward, stop, _, _ = self.env.step(action)
        f = lambda x: sum([v*(self.gamma**(i)) for i,v in enumerate(x)])
        #print(f'First step reward: {self.reward}')
        to_return = (self.obs, action, f(self.reward))

        # All Tensor
        aTensor = torch.cat((torch.from_numpy(convert_gray(obs)).unsqueeze(0), self.obs), dim=0 )
        self.obs = aTensor[:-1,...]
        #save('test_2', self.obs[0])

        # Fix reward r_1 = r + gamma 
        #Will need to add gamma, but this might be right: V_t = R_t + V_(t+1)
        self.reward = [*self.reward, reward][1:]
        #print(f'last step reward: {self.reward}')
        if stop:
            self.reset
        return to_return


if __name__ == '__main__':
    tester = env()

    for i in range(1000):
        action = tester.env.action_space.sample()
        t = tester.step(action)
        print(f'Obs size: {t[0].size()},\tReward: {t[2]}')