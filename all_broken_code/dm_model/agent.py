import jax
import jax.numpy as jnp
import optax
import haiku as hk

from utils import Memory

class DQN():
    def __init__(self,
                 sample_network_input: jnp.ndarray,
                 network: hk.Transformed,
                 optimizer: optax.GradientTransformation,
                 memory: Memory,
                 batch_size: int,
                 epsilon: float,
                 update_period: int,
                 rng_key: jax.random.PRNGKey,
                 device: jax.devices,
                 target_update_period: int = 1000,
                 gamma: float = 0.99,
                 n_actions: int=6):
        '''
        Build a simple agent to explore and learn.
        Inputs:
            - sample_network_input (jpn.ndarray): A sample input to the network to get the shape, does not include batch size
            - network (hk.Transformed): A Haiku network (this is the policy network)
            - optimizer (optax.GradientTransformation): An optimizer to train the network
            - memory (Memory): A memory buffer to store (state, action, reward, next_state, done) tuples
            - batch_size (int): The batch size to sample from the memory buffer
            - epsilon (float): The epsilon value for the epsilon-greedy policy
            - update_period (int): The number of steps before updating the network
            - rng_key (jax.random.PRNGKey): A random number generator key
            - device (jax.devices): The device to run the network on
            - target_update_period (int): The number of steps before updating the target network
            - gamma (float): Discount factor for future rewards
            - n_actions (int): The number of actions in the environment
        '''
        # Jax is strange
        model_init, model_apply = hk.transform(network)
        self.network = model_init(jax.random.PRNGKey(0), jnp.expand_dims(sample_network_input, axis=0))
        self.model = model_apply

        self.target_network = model_init(jax.random.PRNGKey(1), jnp.expand_dims(sample_network_input, axis=0))
        self.target_model = model_apply

        self.optimizer = optimizer
        self.optimizer_state = self.optimizer.init(self.network)

        self.memory = memory
        self.batch_size = batch_size
        self.epsilon = epsilon
        self.update_period = update_period
        self.target_update_period = target_update_period
        self.gamma = gamma
        self.rng_key, _ = jax.random.split(rng_key)
        self.device = device
        self.n_actions = n_actions

        # Counter to keep track of steps
        self.step_count = 0

    def select_action(self, state: jnp.ndarray) -> int:
        '''
        Select an action using epsilon-greedy policy.
        Inputs:
            - state (jnp.ndarray): The current state
        Returns:
            - int: The selected action
        '''
        self.rng_key, _ = jax.random.split(self.rng_key)

        state = jnp.expand_dims(state, axis=0)
        key1, key2 = jax.random.split(self.rng_key)
        if jax.random.uniform(key1, shape=()) < self.epsilon:
            # Explore: Choose a random action
            return jax.random.randint(key2, shape=(), minval=0, maxval=self.n_actions)
        else:
            # Exploit: Choose the action with the highest Q-value
            q_values = self.model(self.network, self.rng_key, state)
            return jnp.argmax(q_values.q_values)

    def store_transition(self, state: jnp.ndarray, action: int, reward: float, next_state: jnp.ndarray, done: bool):
        '''
        Store a transition (state, action, reward, next_state, done) in the memory buffer.
        Inputs:
            - state (jnp.ndarray): The current state
            - action (int): The chosen action
            - reward (float): The received reward
            - next_state (jnp.ndarray): The next state
            - done (bool): Whether the episode is done
        '''
        self.memory.push((state, action, reward, next_state, done))

    def loss_fn(self, params, states, actions, target_q_values):
        q_values = self.model(params, self.rng_key, states).q_values
        return jnp.mean(jnp.sum(jnp.square(q_values[jnp.arange(self.batch_size), actions] - target_q_values)))

    def update(self):
        '''
        Update the Q-network based on a batch of experiences from the memory buffer.
        '''
        self.rng_key, _ = jax.random.split(self.rng_key)
        if len(self.memory) < self.batch_size:
            # Not enough experiences in the memory buffer
            return

        if self.step_count % self.update_period == 0:
            # Sample a batch of experiences from the memory
            batch = self.memory.sample(self.batch_size)

            # Extract the components of the batch
            states, actions, rewards, next_states, dones = zip(*batch)

            # Convert to arrays
            states = jnp.array(states)
            actions = jnp.array(actions)
            rewards = jnp.array(rewards)
            next_states = jnp.array(next_states)
            dones = jnp.array(dones)

            # Compute the Q-values for the current and next states
            q_values = self.model(self.network, self.rng_key, states)
            q_values = q_values.q_values
            next_q_values = self.target_model(self.target_network, self.rng_key, next_states)
            next_q_values = next_q_values.q_values

            # Compute the target Q-values
            target_q_values = rewards + (1 - dones) * self.gamma * jnp.max(next_q_values, axis=-1)

            # Compute the loss and gradients
            grads = jax.grad(self.loss_fn)(self.network, states, actions, target_q_values)

            # Update the network using the optimizer
            updates, self.optimizer_state = self.optimizer.update(grads, self.optimizer_state)
            self.network_params = optax.apply_updates(self.network, updates)

        if self.step_count % self.target_update_period == 0:
            # Update the target network parameters
            self.target_network = self.network

        self.step_count += 1