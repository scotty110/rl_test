'''
Want to base off of:
https://pytorch.org/tutorials/intermediate/reinforcement_q_learning.html
'''
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

import random
import math
from collections import deque

import sys
sys.path.insert(0,'./atari')
from main import env

if not torch.cuda.is_available():
    raise Exception('CUDA not available, code requires cuda')

DEVICE = torch.device('cuda:0')
DTYPE = torch.float32
CPU = torch.device('cpu')


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


class Network(nn.Module):
    def __init__(self, in_channels:int=10, n_actions:int=6):
        super(Network, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=in_channels, out_channels=32, kernel_size=8, stride=4, padding=0)
        self.conv2 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=4, stride=2, padding=0)
        self.conv3 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=0)
        self.fc1 = nn.Linear(in_features=64 * 22 * 16, out_features=512)  # Adjust the in_features based on input image size
        self.fc2 = nn.Linear(in_features=512, out_features=n_actions)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))

        x = x.view(x.size(0), -1)  # Flatten the tensor
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x


class Agent():
    def __init__(self, 
            env, 
            memory_len:int=int(1e4), 
            batch_size:int=256, 
            gamma:float=0.999, 
            eps_start:float=1.0, 
            eps_end:float=0.01, 
            eps_decay:float=0.999, 
            tau:float=1e-2,
            target_update:int=10):

        # Setup env and stuff
        self.env = env()
        self.eval_env = env()
        self.memory = Memory(memory_len) 

        # Setup networks
        self.policy = Network().to(DEVICE, DTYPE) 
        self.target = Network().to(DEVICE, DTYPE)
        self.target.load_state_dict(self.policy.state_dict())
        
        self.optimizer = optim.RMSprop(self.policy.parameters())
        # optim.AdamW(policy_net.parameters(), lr=LR, amsgrad=True)

        self.batch_size = batch_size
        self.gamma = gamma
        self.eps_start = eps_start
        self.eps_end = eps_end
        self.eps_decay = eps_decay
        self.tau = tau
        self.target_update = target_update

        self.steps_done = 0


    def select_action(self, state):
        sample = random.random()
        eps_threshold = self.eps_end + (self.eps_start - self.eps_end) * math.exp(-1. * self.steps_done / self.eps_decay)
        self.steps_done += 1
        if sample > eps_threshold:
            with torch.no_grad():
                return torch.argmax( self.policy(state) ).item()
        else:
            return self.env.random_action()


    def optimize_model(self):
        if len(self.memory) < self.batch_size:
            return

        experiences = self.memory.sample(self.batch_size)

        states, actions, rewards, next_states, dones = zip(*experiences)

        states = torch.stack(states).to(device=DEVICE, dtype=DTYPE)
        actions = torch.tensor(actions).to(device=DEVICE, dtype=torch.int64)
        rewards = torch.tensor(rewards).to(device=DEVICE, dtype=DTYPE)
        next_states = torch.stack(next_states).to(device=DEVICE, dtype=DTYPE)
        dones = torch.tensor(dones, device=DEVICE, dtype=torch.float32)

        # Q values for current states
        curr_Q = self.policy(states).gather(1, actions.unsqueeze(1)).squeeze(1)
        
        # Q values for next states
        next_Q = self.target(next_states).max(1)[0]
        expected_Q = rewards + self.gamma * next_Q * (1 - dones)
        expected_Q = expected_Q.detach()

        # Loss
        loss = F.smooth_l1_loss(curr_Q, expected_Q)

        # Optimize the model
        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_value_(self.policy.parameters(), 10)
        self.optimizer.step()

        # Update target? 
        if self.steps_done % self.target_update == 0:
            for key in self.policy.state_dict():
                self.target.state_dict()[key] = self.tau * self.policy.state_dict()[key] + (1 - self.tau) * self.target.state_dict()[key]

    def episode(self, i:int):
        obs = self.env.reset()
        done = False
        self.steps_done = 0 # Reset at the start of a new episode
        while not done:
            obs = obs.unsqueeze(0).to(device=DEVICE, dtype=DTYPE)
            action = self.select_action(obs)
            next_obs, reward, done, _ = self.env.step(action)
            self.memory.push((obs.squeeze(0).to(device=CPU), action, reward, next_obs, done))
            obs = next_obs

            if self.steps_done > 10000:
                break

            if self.steps_done % 100 == 0:
                self.optimize_model()
        
        self.eval(i)
        return

    def eval(self, episode:int):
        obs = self.eval_env.reset()
        done = False
        total_reward = 0
        while not done:
            obs = obs.unsqueeze(0).to(device=DEVICE, dtype=DTYPE)
            action = self.select_action(obs)
            next_obs, reward, done, _ = self.eval_env.step(action)
            total_reward += reward
            obs = next_obs

        print(f'Episode {episode} Eval reward: {total_reward}')
        return


if __name__ == '__main__':
    agent = Agent(env)
    for i in range(int(1e4)):
        agent.episode(i)