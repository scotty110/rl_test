'''
Just want simple, why is that so much to ask for?
'''

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

import random
import math
from collections import deque

import sys
sys.path.insert(0,'../atari')
from main import env

if not torch.cuda.is_available():
    raise Exception('CUDA not available, code requires cuda')


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
    def __init__(self, in_channels:int=10, n_actions:int=6):
        super(DQN, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=in_channels, out_channels=32, kernel_size=8, stride=4, padding=0)
        self.conv2 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=4, stride=2, padding=0)
        self.conv3 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=0)
        self.fc1 = nn.Linear(in_features=64 * 7 * 7, out_features=512)  # Adjust the in_features based on input image size
        self.fc2 = nn.Linear(in_features=512, out_features=n_actions)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = x.view(x.size(0), -1)  # Flatten the tensor
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x


def update_model(model, optimizer, experiences, gamma=0.99):
    states, actions, rewards, next_states, dones = zip(*experiences)

    states = torch.stack(states)
    actions = torch.tensor(actions)
    rewards = torch.tensor(rewards)
    next_states = torch.stack(next_states)
    dones = torch.tensor(dones, dtype=torch.float32)

    # Q values for current states
    curr_Q = model(states).gather(1, actions.unsqueeze(1)).squeeze(1)

    # Q values for next states
    next_Q = model(next_states).max(1)[0]
    expected_Q = rewards + gamma * next_Q * (1 - dones)

    # Loss
    #loss = F.mse_loss(curr_Q, expected_Q)
    loss = F.smooth_l1_loss(curr_Q, expected_Q)

    # Optimize the model
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()


if __name__ == '__main__': 
    # Simple DQN, please for the love of god work
    batch_size = 128 
    mem = Memory(int(1e3))

    train_env = env()  # Replace with your environment
    n_actions = train_env.n_actions  # Replace with the number of actions in your environment
    model = DQN(in_channels=10, n_actions=n_actions)  # Adjust in_channels based on your input
    optimizer = optim.Adam(model.parameters())

    epsilon = 0.2
    n_episodes = int(1e6)

    for episode in range(n_episodes):
        print(f'Episode {episode}')
        state = train_env.reset()
        done = False
        c = 0

        while not done:
            if random.random() < epsilon:
                action = train_env.random_action()
            else:
                with torch.no_grad():
                    q_values = model(torch.from_numpy(state).unsqueeze(0))
                    action = torch.argmax(q_values).item()

            next_state, reward, done, _ = train_env.step(action)
            mem.push((state, action, reward, next_state, done))

            state = next_state

            if c > 0 and c % 100 == 0 and len(mem) > batch_size:
                experiences = mem.sample(batch_size)
                update_model(model, optimizer, experiences)

            c += 1
            if c > 1000:
                done = True

        if episode > 0 and episode % 10 == 0:
            eval_env = env()
            state = eval_env.reset()
            total_reward = 0
            done = False
            while not done:
                with torch.no_grad():
                    q_values = model(torch.from_numpy(state).unsqueeze(0))
                    action = torch.argmax(q_values).item()

                state, reward, done, _ = eval_env.step(action)
                total_reward += reward

            print(f'Episode finished with reward {total_reward}')