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
                 gamma: float = 0.99):
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
        '''
        # Jax is strange
        model_init, model_apply = hk.transform(network)
        self.network = model_init(jax.random.PRNGKey(0), jnp.expand_dims(sample_network_input, axis=0))
        self.model = model_apply

        self.target_network = model_init(jax.random.PRNGKey(1), jnp.expand_dims(sample_network_input, axis=0))
        self.target_model = model_apply

        self.optimizer = optimizer
        self.memory = memory
        self.batch_size = batch_size
        self.epsilon = epsilon
        self.update_period = update_period
        self.target_update_period = target_update_period
        self.gamma = gamma
        self.rng_key = rng_key
        self.device = device

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
        if jax.random.uniform(self.rng_key, shape=()) < self.epsilon:
            # Explore: Choose a random action
            return jax.random.randint(self.rng_key, shape=(), minval=0, maxval=self.network.output_size)
        else:
            # Exploit: Choose the action with the highest Q-value
            q_values = self.model(self.network, state)
            return jnp.argmax(q_values)

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
        self.memory.append(state, action, reward, next_state, done)

    def update(self):
        '''
        Update the Q-network based on a batch of experiences from the memory buffer.
        '''
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
            q_values = self.model(self.network, states)
            next_q_values = self.target_model(self.target_network, next_states)

            # Compute the target Q-values
            target_q_values = rewards + (1 - dones) * self.gamma * jnp.max(next_q_values, axis=-1)

            # Compute the loss and gradients
            loss = jnp.mean(jax.nn.l2_loss(q_values[jnp.arange(self.batch_size), actions] - target_q_values))
            grads = jax.grad(loss)(self.network.init_params, states)

            # Update the network using the optimizer
            updates, optimizer_state = self.optimizer.update(grads, self.optimizer_state)
            self.network_params = optax.apply_updates(self.network_params, updates)

        if self.step_count % self.target_update_period == 0:
            # Update the target network parameters
            self.target_network_params = self.network_params

        self.step_count += 1