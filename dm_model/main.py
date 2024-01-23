import jax
import jax.numpy as jnp
import optax
import haiku as hk

from atari_env import env
from dqn import dqn_atari_network 

'''
Using: https://github.com/google-deepmind/dqn_zoo/tree/master/dqn_zoo/dqn
As our baseline to compare against. Want to simplify the code and make it more
readable (for me at least).
'''

GPU = jax.devices('gpu')[0]

if __name__ == '__main__':
    # Load Tensor to GPU
    env = env()
    model = dqn_atari_network(env.n_actions)
    
    print('here')
    obs = env.reset()
    obs = jnp.expand_dims(obs, axis=0)
    print(obs.devices())

    # Push forward
    obs = jax.device_put(obs, device=GPU)
    print(obs.devices())

    # Initialize and apply the model inside an hk.transform
    model_init, model_apply = hk.transform(model)
    params = model_init(jax.random.PRNGKey(0), obs)

    for module_name, module_params in params.items():
        for param_name, param_value in module_params.items():
            print(f"{module_name}.{param_name}: shape = {param_value.shape}")


    rng = jax.random.PRNGKey(1)  # RNG for the apply function
    r = model_apply(params, rng, obs)

    print("After model_apply")
    print(r.q_values)  # Assuming your model returns q_values
