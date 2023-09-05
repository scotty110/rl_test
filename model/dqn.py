import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

import random
from collections import deque

import sys
sys.path.insert(0,'../atari')
from main import env

'''
Try and recreate paper: 
    https://arxiv.org/pdf/1312.5602v1.pdf
'''

class Memory():
    '''
    Store (state, action value), tuple pairs. 
    '''
    def __init__(self, capacity):
        self.memory = deque([], maxlen=capacity) 
    
    def push(self, x):
        self.memory.append(x)
    
    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)
    
    def __len__(self):
        return len(self.memory)


class DQN(nn.Module):
    def __init__(self, t_step, n_actions):
        super(DQN, self).__init__()
        self.conv_1 = nn.Conv2d(t_step,32,8,4)
        self.conv_2 = nn.Conv2d(32,1,4,2)
        self.linear_1 = nn.Linear(9, 1)
        self.linear_2 = nn.Linear(9, n_actions)

    # Called with either one element to determine next action, or a batch
    # during optimization. Returns tensor([[left0exp,right0exp]...]).
    def forward(self, x):
        x = F.relu(self.conv_1(x))
        x = F.relu(self.conv_2(x))
        x = x.squeeze(1)
        x = F.relu(self.linear_1(x))
        x = x.squeeze(2)
        return F.softmax(self.linear_2(x), dim=1)


def select_action(action:int, prob:float, action_space:int):
    '''
    Select random action with probability (1-p)
    '''
    r = random.random()
    if r > prob:
        with torch.no_grad():
            return torch.tensor([random.randint(0,action_space)], dtype=torch.float)
    else:
        return action  


def training_step(policy:nn.Module, ) -> Memory:

    pass


def optimize(policy:nn.Module, memory:Memory) -> nn.Module:
    if len(memory) < BATCH_SIZE:
        return
    
    # Get Data (This might not be right)
    batch_data = memory.sample(BATCH_SIZE)
    state_batch = torch.cat( batch_data[:][0], dim=0)
    action_batch = torch.cat( batch_data[:][1], dim=0)
    reward_batch = torch.cat( batch_data[:][2], dim=0)


    return 




if __name__ == '__main__':
    BATCH_SIZE = 32
    '''
    #ex = torch.zeros(5,1,210,160)
    ex = torch.zeros(5,1,84,84)
    #print(ex.shape)

    model = DQN(5)
    x = model(ex)
    print(x.shape)
    print(x[0][:])
    '''
    tstep = 4
    env = env(tstep)
    n_actions = env.env.action_space.n
    model = DQN(tstep, n_actions)
    model = model.float()

    action = env.env.action_space.sample()
    obs = env.step(action)[0]
    obs = obs.unsqueeze(0)
    #print(obs.shape)
    for i in range(1):
        mout = model(obs.float())
        action = torch.argmax(mout)
        #print(action)
        t = env.step(action)
        obs = t[0]
        obs = obs.unsqueeze(0)
