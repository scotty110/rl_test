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

from lunar import env

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
    '''
    Simple MLP for Lunar Lander/Cart Pole
    '''
    def __init__(self, input_dim:int=4, history:int=4, hidden_dim:int=128, n_actions:int=4):
        super(Network, self).__init__()
        self.fc1 = nn.Linear(input_dim * history, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, n_actions)

    def forward(self, x):
        x = x.view(x.size(0), -1) # Flatten
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = self.fc3(x)
        return x


class Agent():
    def __init__(self, 
            env, 
            memory_len:int=int(1e4), 
            batch_size:int=128,
            gamma:float=0.99, 
            eps_start:float=0.9, 
            eps_end:float=0.005, 
            eps_decay:float= 1000, 
            tau:float=0.005,
            target_update:int=10):

        # Setup env and stuff
        self.env = env()
        self.eval_env = env()
        self.memory = Memory(memory_len) 

        # Setup networks
        self.policy = Network(n_actions=self.env.n_actions).to(DEVICE, DTYPE) 
        self.target = Network(n_actions=self.env.n_actions).to(DEVICE, DTYPE)
        self.target.load_state_dict(self.policy.state_dict())
        
        #self.optimizer = optim.RMSprop(self.policy.parameters(), lr=1e-4)
        self.optimizer = optim.AdamW(self.policy.parameters(), lr=1e-3, amsgrad=True)

        self.loss = nn.SmoothL1Loss()

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
                return torch.argmax( self.policy(state) ).item() # Different from tutorial
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
        curr_Q = self.policy(states).gather(1, actions.unsqueeze(1)) #.squeeze(1)
        
        # Q values for next states
        next_Q = self.target(next_states).max(1)[0]
        expected_Q = rewards + self.gamma * next_Q * (1 - dones) # Different from tutorial, but I think Im masking
        expected_Q = expected_Q.unsqueeze(1).detach()

        # Loss
        #loss = F.smooth_l1_loss(curr_Q, expected_Q)
        #print(f'curr_Q: {curr_Q.shape}, expected_Q: {expected_Q.shape}')
        loss = self.loss(curr_Q, expected_Q)


        # Optimize the model
        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.policy.parameters(), 1) # Different than tutorial
        self.optimizer.step()

        # Update target? 
        # This is different than the tutorial, but in a similar order
        if self.steps_done % self.target_update == 0:

            target_net_state_dict = self.target.state_dict()
            policy_net_state_dict = self.policy.state_dict()
            for key in policy_net_state_dict: 
                target_net_state_dict[key] = self.tau * policy_net_state_dict[key] + (1 - self.tau) * target_net_state_dict[key]
            self.target.load_state_dict(target_net_state_dict)

        return

    def episode(self, i:int):
        obs = self.env.reset()
        done = False
        c = 0
        while not done:
            obs = obs.unsqueeze(0).to(device=DEVICE, dtype=DTYPE)
            action = self.select_action(obs)
            next_obs, action, reward, done = self.env.step(action)
            self.memory.push((obs.squeeze(0).to(device=CPU), action, reward, next_obs, done))
            obs = next_obs

            self.optimize_model()

            if c > int(1e5):
                done = True

        self.eval(i)
        return

    def eval(self, episode:int):
        obs = self.eval_env.reset()
        done = False
        total_reward = 0
        dur = 0
        while not done:
            dur += 1
            obs = obs.unsqueeze(0).to(device=DEVICE, dtype=DTYPE)
            action = torch.argmax( self.policy(obs) ).item() 
            next_obs, action, reward, done = self.eval_env.step(action)
            total_reward += reward
            obs = next_obs

            if dur > int(1e5):
                done = True

        print(f'Episode {episode} Eval reward: {total_reward} with a duration of {dur}')
        return


if __name__ == '__main__':
    agent = Agent(env)
    for i in range(int(1000)):
        agent.episode(i)