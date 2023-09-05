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
    def __init__(self, tstep:int=2, env_str:str='ALE/SpaceInvaders-v5'):
        self.env = gym.make(env_str)
        self.tstep = tstep
        self.obs = None
        self.next_obs = None
        self.reward = 0 
        self.reset()

    def reset(self):
        obs, info = self.env.reset()
        obs = convert_gray(obs)
        self.obs = torch.zeros(self.tstep, *obs.shape ) 
        self.obs[0] = torch.from_numpy(obs) 
        save('test_1', self.obs[0])
    
    def step(self, action, gamma:float=0.9):
        obs, reward, stop, _, _ = self.env.step(action)
        to_return = (self.obs, action, self.reward)

        # All Tensor
        #oTensor = torch.from_numpy(convert_gray(obs)).unsqueeze(0)
        #print(oTensor.shape)
        aTensor = torch.cat((torch.from_numpy(convert_gray(obs)).unsqueeze(0), self.obs), dim=0 )
        self.obs = aTensor[:-1,...]
        save('test_2', self.obs[0])

        # House Keeping
        self.reward = self.reward + gamma*reward # This might be wrong
        if stop:
            self.reset
        return to_return


if __name__ == '__main__':
    tester = env()

    for i in range(1000):
        action = tester.env.action_space.sample()
        tester.step(action)