import jax
import jax.numpy as jnp
from collections import deque
import gymnasium as gym
import random

GPU = jax.devices('gpu')[0]


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


def convert_gray(obs):
    obs = obs / 255.0
    weights = jnp.array([0.299, 0.587, 0.114], dtype=jnp.float32).reshape((1, 1, 3))
    grayscale = jnp.sum(obs * weights, axis=2)
    return grayscale


class env():
    def __init__(self, env_str='ALE/SpaceInvaders-v5', stack_size=10):
        self.env = gym.make(env_str)
        self.stack_size = stack_size
        self.obs = deque(maxlen=stack_size)
        self.total_reward = 0
        self.reset()
        self.n_actions = self.env.action_space.n

    def random_action(self):
        return self.env.action_space.sample()
    
    def reset(self):
        self.total_reward = 0
        obs, _ = self.env.reset()
        # Ensure that the observation is on CPU before processing
        obs = jax.device_put(jnp.array(obs, dtype=jnp.float32), device=GPU)
        obs = convert_gray(obs)
        for _ in range(self.stack_size):
            self.obs.append(obs)
        return self.get_stacked_obs()
    
    def step(self, action:int, evaluate=False):
        obs, reward, done, _, _ = self.env.step(action)
        # Ensure that the observation is on CPU before processing
        obs = jax.device_put(jnp.array(obs, dtype=jnp.float32), device=GPU)
        obs = convert_gray(obs)
        self.obs.append(obs)
        stacked_obs = self.get_stacked_obs()
        self.total_reward += reward
        if done:
            if evaluate:
                print(f'Episode finished with reward {self.total_reward}')
            self.reset()
        return stacked_obs, action, reward, done

    def get_stacked_obs(self):
        return jnp.stack(list(self.obs), axis=0)
