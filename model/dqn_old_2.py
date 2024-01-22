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
    def __init__(self, n_actions, history:int=4):
        super(DQN, self).__init__()
        # Assuming the input image is grayscale, thus 1 channel
        self.conv_1 = nn.Conv2d(history, 32, kernel_size=8, stride=4)
        self.conv_2 = nn.Conv2d(32, 64, kernel_size=4, stride=2)
        self.conv_3 = nn.Conv2d(64, 64, kernel_size=3, stride=1)

        # Calculate the size of the output from the last conv layer
        def conv2d_size_out(size, kernel_size=5, stride=2):
            return (size - (kernel_size - 1) - 1) // stride  + 1

        # Example input dimensions (e.g., 160x210 for an Atari game)
        # Adjust these values based on the actual size of your input images
        input_width, input_height = 160, 210

        convw = conv2d_size_out(conv2d_size_out(conv2d_size_out(input_width, 8, 4), 4, 2), 3, 1)
        convh = conv2d_size_out(conv2d_size_out(conv2d_size_out(input_height, 8, 4), 4, 2), 3, 1)
        linear_input_size = convw * convh * 64        

        self.linear_1 = nn.Linear(linear_input_size, 512)
        self.linear_2 = nn.Linear(512, n_actions)

    def forward(self, x):
        if x.dim() == 3:
            x = x.unsqueeze(0)  # Add a channel dimension

        x = F.relu(self.conv_1(x))
        x = F.relu(self.conv_2(x))
        x = F.relu(self.conv_3(x))
        x = x.view(x.size(0), -1)  # Flatten the tensor
        x = F.relu(self.linear_1(x))
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
        return action.item() 


def training_step(policy: nn.Module, env: env, memory: Memory, prob: float, steps_done: int):
    # Get the current observation from the environment
    obs = torch.stack(list(env.obs), dim=0).to('cuda', dtype=torch.float)

    # Forward pass through the policy network
    action_probs = policy(obs)
    action = torch.argmax(action_probs)

    # Select an action using your defined strategy (e.g., epsilon-greedy)
    action = select_action(action, prob, env.n_actions)

    # Perform a step in the environment with the selected action
    new_obs, action, reward, done = env.step(action)

    # Push this experience to memory
    memory.push((obs, action, reward, new_obs, done))

    if steps_done % 100 == 0:
        optimize(policy, memory, env.n_actions)


def optimize(policy: nn.Module, memory: Memory, n_actions: int, gamma: float = 0.99):
    if len(memory) < BATCH_SIZE:
        return

    # Get a batch of experiences
    batch_data = memory.sample(BATCH_SIZE)
    states, actions, rewards, next_states, dones = zip(*batch_data)

    states = torch.stack(states).to('cuda', dtype=torch.float)
    actions = torch.tensor(actions).to('cuda', dtype=torch.long)
    rewards = torch.tensor(rewards).to('cuda', dtype=torch.float)
    next_states = torch.stack(next_states).to('cuda', dtype=torch.float)
    dones = torch.tensor(dones).to('cuda', dtype=torch.long)

    # Compute Q values for current states
    current_q_values = policy(states).gather(1, actions.unsqueeze(-1)).squeeze(-1)

    # Compute Q values for next states
    next_q_values = policy(next_states).max(1)[0]
    next_q_values[dones] = 0  # Zero out Q-values for terminal states

    # Calculate target Q values
    target_q_values = rewards + (gamma * next_q_values)

    # Compute loss
    criterion = nn.SmoothL1Loss()
    loss = criterion(current_q_values, target_q_values)

    # Optimize the model
    optimizer = optim.AdamW(policy.parameters(), lr=1e-4, amsgrad=True)
    optimizer.zero_grad()
    (-loss).backward()

    # Gradient clipping
    torch.nn.utils.clip_grad_value_(policy.parameters(), 100)
    optimizer.step()
    return 

if __name__ == '__main__':
    BATCH_SIZE = 256 

    # Initialize 
    memory = Memory(1000)
    tstep = 4
    env = env()
    n_actions = env.n_actions
    print(f'Number of Actions: {n_actions}')
    model = DQN(n_actions)
    model = model.to('cuda', dtype=torch.float)

    #f = lambda x: sum([v*(self.gamma**(i)) for i,v in enumerate(x)])

    # Training Loop
    for i in range(int(10e6)):
        training_step(model, env, memory, 0.9, i)
