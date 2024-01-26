import jax
import jax.numpy as jnp
import optax
import haiku as hk

from collections import deque

from atari_env import env
from network import dqn_atari_network 
from agent import DQN
from utils import Memory

'''
Using: https://github.com/google-deepmind/dqn_zoo/tree/master/dqn_zoo/dqn
As our baseline to compare against. Want to simplify the code and make it more
readable (for me at least).
'''

GPU = jax.devices('gpu')[0]


if __name__ == '__main__':
    # Init the environment and the model
    train_env = env()
    eval_env = env()
    mem = Memory(int(1e3))
    model = dqn_atari_network(train_env.n_actions)

    # Initialize the basic obs
    obs = train_env.reset()
    obs = jax.device_put(obs, device=GPU)

    # Init the random key
    rng = jax.random.PRNGKey(1)  # RNG for the apply function, still don't understand why I need a random number generator

    agent = DQN(
        sample_network_input=obs,
        network=model,
        optimizer=optax.adam(1e-3),
        memory=mem,
        batch_size=64,
        epsilon=0.1,
        update_period=50,
        rng_key=rng,
        device=GPU,
        target_update_period=100,
        gamma=0.99
    )

    # Train the model
    for i in range(int(1e6)):
        obs = train_env.get_stacked_obs()
        new_obs, action, reward, done = train_env.step(agent.select_action(obs))
        agent.store_transition(obs, action, reward, new_obs, done)

        # Has logic to update the target network
        agent.update()

        if i>0 and i % 200 == 0:
            c = 0
            obs = eval_env.reset()
            done = False
            while not done:
                c += 1
                action = agent.select_action(obs)
                obs, _, _, done = eval_env.step(action, True) 
                if c == 1000:
                    print(f'Eval episode reward: {eval_env.total_reward}')
                    done = True




