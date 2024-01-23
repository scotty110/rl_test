import jax
import optax
import haiku as hk

from atari_env import env

if __name__ == '__main__':
    # Load Tensor to GPU
    device = jax.devices()[0]
    print(device)
    test = jax.random.normal(jax.random.PRNGKey(42), (3, 3))

    env = env()
    print(env.n_actions)