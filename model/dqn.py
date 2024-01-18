import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

import random
from collections import deque

import sys
sys.path.insert(0,'../atari')
from main import env

if not torch.cuda.is_available():
    raise Exception('CUDA not available, code requires cuda')

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
        if len(self.memory) == self.memory.maxlen:
            self.memory.popleft()
        self.memory.append(x)
    
    def sample(self, batch_size):
        return list(random.sample(self.memory, batch_size))
    
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
    epsilon-greedy
    '''
    r = random.random()
    if r > prob:
        with torch.no_grad():
            return random.randint(0,action_space-1)
    else:
        return action 


def training_step(policy:nn.Module, env:env, memory:Memory, prob:float, steps_done:int):
    # Env has obs that is reset at the begining of each episode
    obs = env.obs
    obs = obs.to('cuda', dtype=torch.float)
    
    action = torch.argmax(policy(obs))
    action = select_action(action, prob, env.n_actions) 
    v = env.step(action)
    memory.push(v)

    if steps_done % 100 == 0:
        optimize(policy, memory, env.n_actions) 
    pass


def optimize(policy:nn.Module, memory:Memory, n_actions:int) -> nn.Module:
    if len(memory) < BATCH_SIZE:
        return
    
    # Get Data (This might not be right)
    batch_data = list(zip(*(memory.sample(BATCH_SIZE))))

    state_batch = torch.stack( (batch_data[0]), dim=0).to('cuda', dtype=torch.float)
    #action_batch = torch.stack( torch.tensor((batch_data[1])), dim=0) # actions are ints
    action_batch = torch.tensor(batch_data[1]).unsqueeze(0).to('cuda', dtype=torch.int)
    reward_batch = (torch.tensor(batch_data[2])).unsqueeze(1).to('cuda', dtype=torch.float)
    value_batch = F.one_hot(action_batch, n_actions) * reward_batch

    '''
    DQN predicts Values, and then we select the action with the highest possible value 
    '''
    # Feed to policy
    p_values = policy(state_batch)
    criterion = nn.SmoothL1Loss()
    loss = criterion(value_batch, p_values) 

    # Optimize the model
    optimizer = optim.AdamW(policy.parameters(), lr=1e-4, amsgrad=True)
    optimizer.zero_grad()
    loss.backward()
    # In-place gradient clipping
    torch.nn.utils.clip_grad_value_(policy.parameters(), 100)
    optimizer.step()
    return 


if __name__ == '__main__':
    BATCH_SIZE = 64 

    # Initialize 
    memory = Memory(10000)
    tstep = 4
    env = env(tstep)
    n_actions = env.n_actions
    print(f'Number of Actions: {n_actions}')
    model = DQN(tstep, n_actions)
    model = model.to('cuda', dtype=torch.float)

    # Training Loop
    for i in range(100000):
        training_step(model, env, memory, 0.1, i)
