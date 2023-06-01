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

"""Attention Utils."""
import numpy as np
import tensorflow as tf

_CACHE = {}


def maybe_reset_cache():
  """Resets the constants if the default graph changes."""
  global _CACHE
  def _get_tensor(t):
    if isinstance(t, tf.Tensor):
      return t
    elif isinstance(t, dict):
      return _get_tensor(list(t.values())[0])
    elif isinstance(t, (tuple, list)):
      return _get_tensor(t[0])
    else:
      raise ValueError('Unsupport cache type %d' % type(t))

  if _CACHE:
    _CACHE = {}


def generate_lookup_tensor(length,
                           max_relative_position=None,
                           clamp_out_of_range=False,
                           dtype=tf.float32):
  """Generate a one_hot lookup tensor to reindex embeddings along one dimension.

  Args:
    length: the length to reindex to.
    max_relative_position: the maximum relative position to consider.
      Relative position embeddings for distances above this threshold
      are zeroed out.
    clamp_out_of_range: bool. Whether to clamp out of range locations to the
      maximum relative distance. If False, the out of range locations will be
      filled with all-zero vectors.
    dtype: dtype for the returned lookup tensor.
  Returns:
    a lookup Tensor of size [length, length, vocab_size] that satisfies
      ret[n,m,v] = 1{m - n + max_relative_position = v}.
  """
  maybe_reset_cache()
  if max_relative_position is None:
    max_relative_position = length - 1
  lookup_key = ('lookup_matrix', length, max_relative_position)
  # Return the cached lookup tensor, otherwise compute it and cache it.
  if lookup_key not in _CACHE:
    vocab_size = 2 * max_relative_position + 1
    ret = np.zeros((length, length, vocab_size))
    for i in range(length):
      for x in range(length):
        v = x - i + max_relative_position
        if abs(x - i) > max_relative_position:
          if clamp_out_of_range:
            v = np.clip(v, 0, vocab_size - 1)
          else:
            continue
        ret[i, x, v] = 1
    _CACHE[lookup_key] = tf.constant(ret, dtype)
  return _CACHE[lookup_key]


def reindex_2d_einsum_lookup(relative_position_tensor,
                             height, width,
                             max_relative_height=None,
                             max_relative_width=None,
                             h_axis=None):
  """Reindex 2d relative position bias with 2 independent einsum lookups.

  Args:
    relative_position_tensor: tensor of shape
      [..., vocab_height, vocab_width, ...].
    height: height to reindex to.
    width: width to reindex to.
    max_relative_height: maximum relative height.
      Position embeddings corresponding to vertical distances larger
      than max_relative_height are zeroed out. None to disable.
    max_relative_width: maximum relative width.
      Position embeddings corresponding to horizontal distances larger
      than max_relative_width are zeroed out. None to disable.
    h_axis: Axis corresponding to vocab_height. Default to 0 if None.

  Returns:
    reindexed_tensor: a Tensor of shape
      [..., height * width, height * width, ...]
  """
  height_lookup = generate_lookup_tensor(
      height, max_relative_position=max_relative_height,
      dtype=relative_position_tensor.dtype)
  width_lookup = generate_lookup_tensor(
      width, max_relative_position=max_relative_width,
      dtype=relative_position_tensor.dtype)

  if h_axis is None:
    h_axis = 0

  non_spatial_rank = relative_position_tensor.shape.rank - 2
  non_spatial_expr = ''.join(chr(ord('n') + i) for i in range(non_spatial_rank))
  prefix = non_spatial_expr[:h_axis]
  suffix = non_spatial_expr[h_axis:]

  reindexed_tensor = tf.einsum(
      '{0}hw{1},ixh->{0}ixw{1}'.format(prefix, suffix),
      relative_position_tensor, height_lookup, name='height_lookup')
  reindexed_tensor = tf.einsum(
      '{0}ixw{1},jyw->{0}ijxy{1}'.format(prefix, suffix),
      reindexed_tensor, width_lookup, name='width_lookup')

  ret_shape = relative_position_tensor.shape.as_list()
  ret_shape[h_axis] = height * width
  ret_shape[h_axis + 1] = height * width
  reindexed_tensor = tf.reshape(reindexed_tensor, ret_shape)

  return reindexed_tensor


def float32_softmax(x, *args, **kwargs):
  y = tf.cast(tf.nn.softmax(tf.cast(x, tf.float32), *args, **kwargs), x.dtype)
  return y
