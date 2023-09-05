# https://github.com/openai/gym/issues/3201
import ale_py
# if using gymnasium
#import shimmy

import gymnasium as gym
#env = gym.make("CartPole-v1")
#print(gym.envs.registry.keys())

env = gym.make("ALE/SpaceInvaders-v5")

observation, info = env.reset(seed=42)
for _ in range(1000):
    action = env.action_space.sample()
    observation, reward, terminated, truncated, info = env.step(action)
    print(observation.shape)
    print(f'reward: {reward}')

    if terminated or truncated:
        observation, info = env.reset()
env.close()