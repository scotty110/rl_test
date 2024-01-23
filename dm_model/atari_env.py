import jax.numpy as jnp
from collections import deque
import gymnasium as gym

def convert_gray(obs):
    weights = jnp.array([0.299, 0.587, 0.114]).reshape((1, 1, 3))
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
        obs = convert_gray(jnp.array(obs))
        for _ in range(self.stack_size):
            self.obs.append(obs)
        return self.get_stacked_obs()
    
    def step(self, action):
        obs, reward, done, _ = self.env.step(action)
        obs = convert_gray(jnp.array(obs))
        self.obs.append(obs)
        stacked_obs = self.get_stacked_obs()
        self.total_reward += reward
        if done:
            print(f'Episode finished with reward {self.total_reward}')
            self.reset()
        return stacked_obs, action, reward, done

    def get_stacked_obs(self):
        # Assuming the observations are images, stack along a new dimension
        return jnp.stack(list(self.obs), axis=0)