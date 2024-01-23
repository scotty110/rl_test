'''
Re Implement DQN from Deepmind
https://github.com/google-deepmind/dqn_zoo/blob/45061f4bbbcfa87d11bbba3cfc2305a650a41c26/dqn_zoo/networks.py#L352
'''
import typing
from typing import Any, Callable, Tuple, Union

import haiku as hk
import jax
import jax.numpy as jnp

NetworkFn = Callable[..., Any]

class QNetworkOutputs(typing.NamedTuple):
  q_values: jnp.ndarray


def _dqn_default_initializer(
    num_input_units: int,
) -> hk.initializers.Initializer:
  """Default initialization scheme inherited from past implementations of DQN.

  This scheme was historically used to initialize all weights and biases
  in convolutional and linear layers of DQN-type agents' networks.
  It initializes each weight as an independent uniform sample from [`-c`, `c`],
  where `c = 1 / np.sqrt(num_input_units)`, and `num_input_units` is the number
  of input units affecting a single output unit in the given layer, i.e. the
  total number of inputs in the case of linear (dense) layers, and
  `num_input_channels * kernel_width * kernel_height` in the case of
  convolutional layers.

  Args:
    num_input_units: number of input units to a single output unit of the layer.

  Returns:
    Haiku weight initializer.
  """
  max_val = jnp.sqrt(1 / num_input_units)
  return hk.initializers.RandomUniform(-max_val, max_val)

'''
def conv(
    num_features: int,
    kernel_shape: Union[int, Tuple[int, int]],
    stride: Union[int, Tuple[int, int]],
) -> NetworkFn:
  """Convolutional layer with DQN's legacy weight initialization scheme."""

  def net_fn(inputs):
    """Function representing conv layer with DQN's legacy initialization."""
    """ (batch_size, height, width, features) """
    num_input_units = inputs.shape[-1] * kernel_shape[0] * kernel_shape[1]
    initializer = _dqn_default_initializer(num_input_units)
    layer = hk.Conv2D(
        num_features,
        kernel_shape=kernel_shape,
        stride=stride,
        w_init=initializer,
        b_init=initializer,
        padding='VALID',
    )
    return layer(inputs)

  return net_fn
'''
def conv(
    num_features: int,
    kernel_shape: Union[int, Tuple[int, int]],
    stride: Union[int, Tuple[int, int]],
) -> NetworkFn:
  """Convolutional layer with DQN's legacy weight initialization scheme."""

  def net_fn(inputs):
    """Function representing conv layer with DQN's legacy initialization."""
    num_input_units = inputs.shape[1] * kernel_shape[0] * kernel_shape[1]
    initializer = _dqn_default_initializer(num_input_units)
    layer = hk.Conv2D(
        num_features,
        kernel_shape=kernel_shape,
        stride=stride,
        w_init=initializer,
        b_init=initializer,
        padding='VALID',
        data_format='NCHW',  # Add this line to change the data format
    )
    return layer(inputs)

  return net_fn


def linear(num_outputs: int, with_bias=True) -> NetworkFn:
  """Linear layer with DQN's legacy weight initialization scheme."""

  def net_fn(inputs):
    """Function representing linear layer with DQN's legacy initialization."""
    initializer = _dqn_default_initializer(inputs.shape[0])
    layer = hk.Linear(
        num_outputs, with_bias=with_bias, w_init=initializer, b_init=initializer
    )
    return layer(inputs)

  return net_fn


def linear_with_shared_bias(num_outputs: int) -> NetworkFn:
  """Linear layer with single shared bias instead of one bias per output."""

  def layer_fn(inputs):
    """Function representing a linear layer with single shared bias."""
    initializer = _dqn_default_initializer(inputs.shape[-1])
    bias_free_linear = hk.Linear(
        num_outputs, with_bias=False, w_init=initializer
    )
    linear_output = bias_free_linear(inputs)
    bias = hk.get_parameter('b', [1], inputs.dtype, init=initializer)
    bias = jnp.broadcast_to(bias, linear_output.shape)
    return linear_output + bias

  return layer_fn


def dqn_torso() -> NetworkFn:
  """DQN convolutional torso.

  Includes scaling from [`0`, `255`] (`uint8`) to [`0`, `1`] (`float32`)`.

  Returns:
    Network function that `haiku.transform` can be called on.
  """

  def net_fn(inputs):
    """Function representing convolutional torso for a DQN Q-network."""
    network = hk.Sequential([
        #lambda x: x.astype(jnp.float32) / 255.0, # Already scaled and converted to grayscale
        conv(32, kernel_shape=(8, 8), stride=(4, 4)),
        jax.nn.relu,
        conv(64, kernel_shape=(4, 4), stride=(2, 2)),
        jax.nn.relu,
        conv(64, kernel_shape=(3, 3), stride=(1, 1)),
        jax.nn.relu,
        hk.Flatten(),
    ])
    print(f'net output: {network(inputs).shape}')
    return network(inputs)

  return net_fn 


def dqn_value_head(num_actions: int, shared_bias: bool = False) -> NetworkFn:
  """Regular DQN Q-value head with single hidden layer."""

  last_layer = linear_with_shared_bias if shared_bias else linear

  def net_fn(inputs):
    """Function representing value head for a DQN Q-network."""
    print(f'linear inputs: {inputs.shape}')
    network = hk.Sequential([
        linear(512),
        jax.nn.relu,
        last_layer(num_actions),
    ])
    return network(inputs)

  return net_fn


def dqn_atari_network(num_actions: int) -> NetworkFn:
  """DQN network, expects `uint8` input."""

  def net_fn(inputs):
    """Function representing DQN Q-network."""
    print(f'Input shape: {inputs.shape}')
    network = hk.Sequential([
        dqn_torso(),
        dqn_value_head(num_actions),
    ])
    print(f'Output shape: {network(inputs).shape}')
    return QNetworkOutputs(q_values=network(inputs))

  return net_fn