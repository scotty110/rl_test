import jax
import optax
import jax.numpy as jnp
from jax import grad
import haiku as hk

from atari_env import env, Memory

def create_dqn_network(n_actions:int=6):
    def _forward(x):
        net = hk.Sequential([
            #lambda x: jnp.transpose(x, (0, 2, 3, 1)),
            hk.Conv2D(output_channels=32, kernel_shape=8, stride=4, padding='VALID', data_format='NCHW'),
            jax.nn.relu,
            hk.Conv2D(output_channels=64, kernel_shape=4, stride=2, padding='VALID', data_format='NCHW'),
            jax.nn.relu,
            hk.Conv2D(output_channels=64, kernel_shape=3, stride=1, padding='VALID', data_format='NCHW'),
            jax.nn.relu,
            hk.Flatten(),
            hk.Linear(512),
            jax.nn.relu,
            hk.Linear(n_actions)
        ])
        return net(x)
    return _forward


def create_optimizer(learning_rate=1e-3):
    return optax.adam(learning_rate)


def update_model(params, opt_state, experiences, optimizer, network, rng, gamma=0.99, n_actions:int=6):
    states, actions, rewards, next_states, dones = zip(*experiences)
    # Convert to arrays
    states = jnp.array(states)
    actions = jnp.array(actions)
    rewards = jnp.array(rewards)
    next_states = jnp.array(next_states)
    dones = jnp.array(dones)

    '''
    print(f'states.shape: {states.shape}')
    print(f'actions.shape: {actions.shape}')
    print(f'rewards.shape: {rewards.shape}')
    print(f'next_states.shape: {next_states.shape}')
    print(f'dones.shape: {dones.shape}')
    '''

    def loss_fn(params):
        #q_values = network.apply(params, states)
        q_values = network(params, rng, states)
        #next_q_values = network.apply(params, next_states)
        next_q_values = network(params, rng, next_states)
        max_next_q_values = jnp.max(next_q_values, axis=1)
        target_q_values = rewards + gamma * max_next_q_values * (1 - dones)
        actions_one_hot = jax.nn.one_hot(actions, num_classes=n_actions)
        predicted_q_values = jnp.sum(q_values * actions_one_hot, axis=1)
        return jnp.mean((predicted_q_values - target_q_values) ** 2)

    grads = grad(loss_fn)(params)
    updates, opt_state = optimizer.update(grads, opt_state)
    new_params = optax.apply_updates(params, updates)
    return new_params, opt_state


if __name__ == '__main__':
    batch_size = 128 
    mem = Memory(int(1e3))

    train_env = env()
    obs = train_env.reset()
    obs = jnp.expand_dims(obs, axis=0)

    network = create_dqn_network(train_env.n_actions)
    model_init, model = hk.transform(network)
    params = model_init(jax.random.PRNGKey(1), obs)

    optimizer = create_optimizer()
    opt_state = optimizer.init(params)

    rng = jax.random.PRNGKey(0)
    epsilon = 0.2

    n_episodes = int(1e6)
    state = train_env.reset()
    for episode in range(n_episodes):
        # Environment is reset at the start of each episode internally
        print(f'Episode {episode}')
        done = False
        c = 0
        while not done:
            # Epsilon-greedy policy for exploration
            rng, new_key = jax.random.split(rng)
            state = jnp.expand_dims(state, axis=0)
            if jax.random.uniform(rng) < epsilon:
                action = train_env.random_action()
            else:
                q_values = model(params, rng, state)
                action = jnp.argmax(q_values)

            next_state, action, reward, done = train_env.step(action)
            mem.push((jnp.squeeze(state, axis=0), action, reward, next_state, done))

            state = next_state

            if c > 0 and c % 100 == 0:
                if len(mem) > batch_size:
                    experiences = mem.sample(batch_size)
                    rng, _ = jax.random.split(rng)
                    #params, opt_state = update_model(params, opt_state, experiences)
                    params, opt_state = update_model(params, opt_state, experiences, optimizer, model, rng)
            c += 1
            if c > 1000:
                done = True

        if episode>0 and episode % 10 == 0:
            eval_env = env()
            state = eval_env.reset()
            while not done:
                state = jnp.expand_dims(state, axis=0)
                q_values = model(params, rng, state)
                action = jnp.argmax(q_values)

                state, action, reward, done = eval_env.step(action, evaluate=True)
            # Done
            print(f'Episode finished with reward {eval_env.total_reward}')


            state = next_state