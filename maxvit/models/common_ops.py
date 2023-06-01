# Copyright 2023 Google LLC.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Common operations."""
import functools
from absl import logging
import numpy as np
import tensorflow as tf
from tensorflow.python.tpu import tpu_function  # pylint:disable=g-direct-tensorflow-import


def activation_fn(features: tf.Tensor, act_fn: str):
  """Customized non-linear activation type."""
  if act_fn in ('silu', 'swish'):
    return tf.nn.swish(features)
  elif act_fn == 'relu':
    return tf.nn.relu(features)
  elif act_fn == 'gelu':
    return 0.5 * features * (1 + tf.tanh(np.sqrt(2 / np.pi) * (
        features + 0.044715 * tf.pow(features, 3))))
  else:
    raise ValueError('Unsupported act_fn {}'.format(act_fn))


def get_act_fn(act_fn):
  if act_fn is None:
    act_fn = 'gelu'
  if isinstance(act_fn, str):
    return functools.partial(activation_fn, act_fn=act_fn)
  elif callable(act_fn):
    return act_fn
  else:
    raise ValueError('Unsupported act_fn %s.' % act_fn)


def pooling_2d(inputs, pool_type, stride, **kwargs):
  """Perform 2D pooling."""
  if stride > 1:
    if pool_type == 'max':
      pool_op = tf.keras.layers.MaxPool2D
    elif pool_type == 'avg':
      pool_op = tf.keras.layers.AveragePooling2D
    else:
      raise ValueError('Unsurpported pool_type %s' % pool_type)
    output = pool_op(
        pool_size=(stride, stride),
        strides=(stride, stride),
        **kwargs)(inputs)
  else:
    output = inputs
  return output


def drop_connect(inputs, training, survival_prob):
  """Drop the entire conv with given survival probability."""
  # "Deep Networks with Stochastic Depth", https://arxiv.org/pdf/1603.09382.pdf
  if not training:
    return inputs

  # Compute tensor.
  batch_size = tf.shape(inputs)[0]
  random_tensor = survival_prob
  random_tensor += tf.random.uniform([batch_size], dtype=inputs.dtype)
  for _ in range(inputs.shape.rank - 1):
    random_tensor = tf.expand_dims(random_tensor, axis=-1)
  binary_tensor = tf.floor(random_tensor)
  # Unlike conventional way that multiply survival_prob at test time, here we
  # divide survival_prob at training time, such that no addition compute is
  # needed at test time.
  output = inputs / survival_prob * binary_tensor
  return output


def residual_add(residual, shortcut, survival_prob, training):
  """Combine residual and shortcut."""
  if survival_prob is not None and 0 < survival_prob < 1:
    residual = drop_connect(residual, training, survival_prob)
  return shortcut + residual


def cross_replica_mean(t, num_shards_per_group=None):
  """Calculates the average value of input tensor across TPU replicas."""
  num_shards = tpu_function.get_tpu_context().number_of_shards
  if not num_shards_per_group:
    return tf.compat.v1.tpu.cross_replica_sum(t) / tf.cast(num_shards, t.dtype)

  group_assignment = None
  if num_shards_per_group > 1:
    if num_shards % num_shards_per_group != 0:
      raise ValueError('num_shards: %d mod shards_per_group: %d, should be 0' %
                       (num_shards, num_shards_per_group))
    num_groups = num_shards // num_shards_per_group
    group_assignment = [[
        x for x in range(num_shards) if x // num_shards_per_group == y
    ] for y in range(num_groups)]
  return tf.compat.v1.tpu.cross_replica_sum(t, group_assignment) / tf.cast(
      num_shards_per_group, t.dtype)


def maybe_reshape_to_2d(x, height=None):
  """Reshape tensor to 2d if not already 2d."""
  if x.shape.rank == 3:
    _, length, num_channel = x.shape.as_list()
    if height is None:
      height = int(np.sqrt(length))
    else:
      assert length % height == 0
    width = length // height
    logging.debug('Reshape %s -> %s', [length, num_channel],
                  [height, width, num_channel])
    return tf.reshape(x, [-1, height, width, num_channel])
  elif x.shape.rank == 4:
    return x
  else:
    raise ValueError('Unsupport shape {}'.format(x.shape))


def maybe_reshape_to_1d(x):
  """Reshape tensor to 1d if not already 1d."""
  if x.shape.rank == 4:
    _, h, w, num_channel = x.shape.as_list()
    logging.debug('Reshape %s -> %s', [h, w, num_channel],
                  [h * w, num_channel])
    return tf.reshape(x, [-1, h * w, num_channel])
  elif x.shape.rank == 3:
    return x
  else:
    raise ValueError('Unsupport shape {}'.format(x.shape))
