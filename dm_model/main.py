import jax
import jax.numpy as jnp
import optax
import haiku as hk

from collections import deque

from atari_env import env
from network import dqn_atari_network 
from agent import DQN

'''
Using: https://github.com/google-deepmind/dqn_zoo/tree/master/dqn_zoo/dqn
As our baseline to compare against. Want to simplify the code and make it more
readable (for me at least).
'''

GPU = jax.devices('gpu')[0]


if __name__ == '__main__':
    # Init the environment and the model
    env = env()
    model = dqn_atari_network(env.n_actions)

    # Initialize the basic obs
    obs = env.reset()
    obs = jax.device_put(obs, device=GPU)

    # Init the random key
    rng = jax.random.PRNGKey(1)  # RNG for the apply function, still don't understand why I need a random number generator

    agent = DQN(
        sample_network_input=obs,
        network=model,
        optimizer=optax.adam(1e-4),
        memory=deque(maxlen=int(1e4)),
        batch_size=32,
        epsilon=0.01,
        update_period=4,
        rng_key=rng,
        device=GPU,
        target_update_period=1000,
        gamma=0.99
    )

    # Train the model
    for i in range(int(1e6)):
        # 
        env_step = env.step(agent.select_action(obs))
        agent.memory.append(env_step)
        agent.update()




