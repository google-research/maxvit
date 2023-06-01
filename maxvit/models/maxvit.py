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

"""Layers and Model class for MaxViT."""
import copy
import functools
import math
import re
import string

from absl import logging
from maxvit.models import attention_utils as attn_utils
from maxvit.models import common_ops as ops
from maxvit.models import hparam_configs
import tensorflow as tf


def float32_softmax(x, *args, **kwargs):
  y = tf.cast(tf.nn.softmax(tf.cast(x, tf.float32), *args, **kwargs), x.dtype)
  return y


def create_config_from_dict(config_dict, required_keys, optional_keys):
  """Create hparam config from dictionary."""
  config = hparam_configs.Config()
  for key in required_keys:
    if key not in config_dict:
      raise ValueError('Required key %s missed in config dict.' % key)
    config[key] = config_dict[key]
  for key in optional_keys:
    if key not in config_dict:
      config[key] = optional_keys[key]
    else:
      config[key] = config_dict[key]
  return config


class TrailDense(tf.keras.layers.Layer):
  """Dense module that projects multiple trailing dimensions."""

  def __init__(self,
               output_trailing_dims,
               begin_axis=-1,
               use_bias=True,
               kernel_initializer=tf.random_normal_initializer(stddev=0.02),
               bias_initializer=tf.zeros_initializer,
               name='dense'):
    super(TrailDense, self).__init__(name=name)

    if isinstance(output_trailing_dims, int):
      self._output_trailing_dims = [output_trailing_dims]
    else:
      assert isinstance(output_trailing_dims, (list, tuple)) and all(
          isinstance(i, int) for i in output_trailing_dims), (
              'Invalid output shape: {}.'.format(output_trailing_dims))
      self._output_trailing_dims = list(output_trailing_dims)
    self.begin_axis = begin_axis
    self.use_bias = use_bias

    self.kernel_initializer = kernel_initializer
    self.bias_initializer = bias_initializer

  def build(self, input_shape):
    """Create variables and einsum expression based on input shape."""
    # Create variables
    weight_shape = input_shape[self.begin_axis:] + self._output_trailing_dims
    self.weight = self.add_weight(
        name='weight',
        shape=weight_shape,
        initializer=self.kernel_initializer,
        trainable=True)
    if self.use_bias:
      self.bias = self.add_weight(
          name='bias',
          shape=self._output_trailing_dims,
          initializer=self.bias_initializer,
          trainable=True)

    # Create einsum expression
    input_rank = input_shape.rank
    shared_size = self.begin_axis % input_rank
    i_only_size = input_rank - shared_size
    o_only_size = len(self._output_trailing_dims)

    assert input_rank + o_only_size < len(string.ascii_uppercase), (
        'Cannot use einsum as input rank + output rank > 26.')
    einsum_str = string.ascii_uppercase[:input_rank + o_only_size]

    offset = 0
    shared_str = einsum_str[offset:offset+shared_size]
    offset += shared_size
    i_only_str = einsum_str[offset:offset+i_only_size]
    offset += i_only_size
    o_only_str = einsum_str[offset:offset+o_only_size]

    input_str = '{}{}'.format(shared_str, i_only_str)
    output_str = '{}{}'.format(shared_str, o_only_str)
    weight_str = '{}{}'.format(i_only_str, o_only_str)

    self.einsum_expr = '{},{}->{}'.format(input_str, weight_str, output_str)

  def call(self, inputs):
    output = tf.einsum(self.einsum_expr, inputs, self.weight)
    if self.use_bias:
      output += self.bias
    return output


class Attention(tf.keras.layers.Layer):
  """Multi-headed attention module."""

  def __init__(self,
               hidden_size,
               head_size,
               num_heads=None,
               dropatt=0.0,
               attn_axis=0,
               rel_attn_type=None,
               scale_ratio=None,
               kernel_initializer=tf.random_normal_initializer(stddev=0.02),
               bias_initializer=tf.zeros_initializer,
               name='attention'):
    super(Attention, self).__init__(name=name)

    self.hidden_size = hidden_size
    self.head_size = head_size
    self.num_heads = num_heads or hidden_size // head_size
    self.dropatt = dropatt
    self.attn_axis = attn_axis
    self.rel_attn_type = rel_attn_type
    self.scale_ratio = scale_ratio

    self.kernel_initializer = kernel_initializer
    self.bias_initializer = bias_initializer

    self._q_proj = TrailDense(
        output_trailing_dims=[self.num_heads, self.head_size],
        kernel_initializer=kernel_initializer,
        bias_initializer=bias_initializer,
        name='q')
    self._k_proj = TrailDense(
        output_trailing_dims=[self.num_heads, self.head_size],
        kernel_initializer=kernel_initializer,
        bias_initializer=bias_initializer,
        name='k')
    self._v_proj = TrailDense(
        output_trailing_dims=[self.num_heads, self.head_size],
        kernel_initializer=kernel_initializer,
        bias_initializer=bias_initializer,
        name='v')
    self._o_proj = TrailDense(
        output_trailing_dims=self.hidden_size, begin_axis=-2,
        kernel_initializer=kernel_initializer,
        bias_initializer=bias_initializer,
        name='o')

    self.q_scale = self.head_size ** -0.5

  def build(self, query_shape):
    num_attn_dims = query_shape.rank - 2   # -2 to account for bsz, hidden size
    assert num_attn_dims < 6, 'Only support at most 6 attention dims.'
    symbols = ''.join([chr(ord('U') + i) for i in range(num_attn_dims - 1)])
    insert = lambda s, i, c: s[:i] + c + s[i:]
    create_expr = lambda s, prefix='B', suffix='NK': prefix + s + suffix
    self.q_expr = create_expr(insert(symbols, self.attn_axis, 'S'))
    self.k_expr = create_expr(insert(symbols, self.attn_axis, 'T'))
    self.v_expr = create_expr(insert(symbols, self.attn_axis, 'T'))
    self.a_expr = create_expr(symbols, suffix='NST')

    ##### Relative attention
    if self.rel_attn_type in ['2d_multi_head', '2d_single_head']:
      query_shape_list = query_shape.as_list()
      if query_shape.rank == 4:
        height, width = query_shape_list[1:3]
      elif query_shape.rank == 3:
        seq_len = query_shape_list[1]
        height = int(seq_len ** 0.5)
        width = height
        if height * width != seq_len:
          raise ValueError('Does not support 2D relative attentive for '
                           'non-square inputs.')
      else:
        raise ValueError(
            'Does not support relative attention for query shape: %s.'
            % query_shape_list)

      if self.scale_ratio is not None:
        scale_ratio = eval(self.scale_ratio)
        vocab_height = 2 * round(height / scale_ratio) - 1
        vocab_width = 2 * round(width / scale_ratio) - 1
      else:
        vocab_height = 2 * height - 1
        vocab_width = 2 * width - 1

      if self.rel_attn_type == '2d_multi_head':
        h_axis = 1
        rel_bias_shape = [self.num_heads, vocab_height, vocab_width]
      elif self.rel_attn_type == '2d_single_head':
        h_axis = 0
        rel_bias_shape = [vocab_height, vocab_width]
      else:
        raise NotImplementedError('rel_attn_type %s not implemented yet.' %
                                  self.rel_attn_type)

      self.relative_bias = self.add_weight(
          'relative_bias',
          rel_bias_shape,
          initializer=self.kernel_initializer,
          trainable=True)

      if self.scale_ratio is not None:
        src_shape = self.relative_bias.shape.as_list()
        relative_bias = tf.expand_dims(self.relative_bias, axis=-1)
        relative_bias = tf.cast(
            tf.image.resize(relative_bias, [2 * height - 1, 2 * width - 1]),
            self.compute_dtype)
        relative_bias = tf.squeeze(relative_bias, axis=-1)
        tgt_shape = relative_bias.shape.as_list()
        logging.info('Bilinear resize relative position bias %s -> %s.',
                     src_shape, tgt_shape)
      else:
        relative_bias = tf.cast(self.relative_bias, self.compute_dtype)

      self.reindexed_bias = attn_utils.reindex_2d_einsum_lookup(
          relative_bias, height, width, height - 1, width - 1,
          h_axis=h_axis)
    else:
      self.reindexed_bias = None

  def call(self, query, training, context=None, attn_mask=None):
    if context is None:
      context = query

    q_heads = self._q_proj(query)
    k_heads = self._k_proj(context)
    v_heads = self._v_proj(context)
    q_heads *= self.q_scale

    # attention
    attn_logits = tf.einsum(
        f'{self.q_expr},{self.k_expr}->{self.a_expr}',
        q_heads, k_heads)

    if self.reindexed_bias is not None:
      attn_logits += self.reindexed_bias

    if attn_mask is not None:
      attn_logits += (1.0 - attn_mask) * attn_logits.dtype.min

    attn_probs = float32_softmax(attn_logits, axis=-1)
    if self.dropatt:
      attn_probs = tf.keras.layers.Dropout(self.dropatt, 'attn_prob_drop')(
          attn_probs, training=training)

    attn_out = tf.einsum(
        f'{self.a_expr},{self.v_expr}->{self.q_expr}',
        attn_probs, v_heads)
    output = self._o_proj(attn_out)

    return output


class FFN(tf.keras.layers.Layer):
  """Positionwise feed-forward network."""

  def __init__(self,
               hidden_size,
               dropout=0.0,
               expansion_rate=4,
               activation='gelu',
               kernel_initializer=tf.random_normal_initializer(stddev=0.02),
               bias_initializer=tf.zeros_initializer,
               name='ffn'):
    super(FFN, self).__init__(name=name)

    self.hidden_size = hidden_size
    self.expansion_rate = expansion_rate
    self.expanded_size = self.hidden_size * self.expansion_rate
    self.dropout = dropout
    self.activation = activation

    self._expand_dense = TrailDense(
        output_trailing_dims=self.expanded_size,
        kernel_initializer=kernel_initializer,
        bias_initializer=bias_initializer,
        name='expand_dense')
    self._shrink_dense = TrailDense(
        output_trailing_dims=self.hidden_size,
        kernel_initializer=kernel_initializer,
        bias_initializer=bias_initializer,
        name='shrink_dense')
    self._activation_fn = ops.get_act_fn(self.activation)

  def call(self, inputs, training):
    output = inputs
    output = self._expand_dense(output)
    output = self._activation_fn(output)
    if self.dropout:
      output = tf.keras.layers.Dropout(self.dropout, name='nonlinearity_drop')(
          output, training=training)
    output = self._shrink_dense(output)

    return output


class TransformerBlock(tf.keras.layers.Layer):
  """Transformer block = Attention + FFN."""

  def _retrieve_config(self, config):
    required_keys = ['hidden_size', 'head_size']
    optional_keys = {
        'num_heads': None,
        'expansion_rate': 4,
        'activation': 'gelu',
        'pool_type': 'avg',
        'pool_stride': 1,
        'dropatt': None,
        'dropout': None,
        'rel_attn_type': '2d_multi_head',
        'scale_ratio': None,
        'survival_prob': None,
        'ln_epsilon': 1e-5,
        'ln_dtype': None,
        'kernel_initializer': tf.random_normal_initializer(stddev=0.02),
        'bias_initializer': tf.zeros_initializer,
    }
    config = create_config_from_dict(config, required_keys, optional_keys)
    return config

  def __init__(self, config, name='transformer'):
    super().__init__(name=name)

    self._config = self._retrieve_config(config)

  def build(self, input_shape):
    config = self._config

    input_size = input_shape.as_list()[-1]

    if input_size != config.hidden_size:
      self._shortcut_proj = TrailDense(
          config.hidden_size,
          kernel_initializer=config.kernel_initializer,
          bias_initializer=config.bias_initializer,
          name='shortcut_proj')
    else:
      self._shortcut_proj = None

    self._attn_layer_norm = tf.keras.layers.LayerNormalization(
        axis=-1,
        epsilon=config.ln_epsilon,
        dtype=config.ln_dtype,
        name='attn_layer_norm')

    self._attention = Attention(
        config.hidden_size,
        config.head_size,
        num_heads=config.num_heads,
        dropatt=config.dropatt,
        rel_attn_type=config.rel_attn_type,
        scale_ratio=config.scale_ratio,
        kernel_initializer=config.kernel_initializer,
        bias_initializer=config.bias_initializer)

    self._ffn_layer_norm = tf.keras.layers.LayerNormalization(
        axis=-1,
        epsilon=config.ln_epsilon,
        dtype=config.ln_dtype,
        name='ffn_layer_norm')

    self._ffn = FFN(
        config.hidden_size,
        dropout=config.dropout,
        expansion_rate=config.expansion_rate,
        activation=config.activation,
        kernel_initializer=config.kernel_initializer,
        bias_initializer=config.bias_initializer)

  def downsample(self, inputs, name):
    config = self._config
    output = inputs
    if config.pool_stride > 1:
      output = ops.maybe_reshape_to_2d(output)
      output = ops.pooling_2d(output,
                              config.pool_type,
                              config.pool_stride,
                              padding='same',
                              data_format='channels_last',
                              name=name)
    return output

  def shortcut_branch(self, shortcut):
    shortcut = self.downsample(shortcut, 'shortcut_pool')
    shortcut = ops.maybe_reshape_to_1d(shortcut)
    if self._shortcut_proj:
      shortcut = self._shortcut_proj(shortcut)

    return shortcut

  def attn_branch(self, inputs, training, attn_mask):
    output = self._attn_layer_norm(inputs)
    output = self.downsample(output, 'residual_pool')
    output = ops.maybe_reshape_to_1d(output)
    output = self._attention(output, training, attn_mask=attn_mask)
    return output

  def ffn_branch(self, inputs, training):
    output = self._ffn_layer_norm(inputs)
    output = self._ffn(output, training)
    return output

  def call(self, inputs, training, attn_mask=None):
    logging.debug('Block %s input shape: %s (%s)', self.name, inputs.shape,
                  inputs.dtype)

    config = self._config

    shortcut = self.shortcut_branch(inputs)
    output = self.attn_branch(inputs, training, attn_mask)
    if config.dropout:
      output = tf.keras.layers.Dropout(config.dropout, name='after_attn_drop')(
          output, training=training)
    output = ops.residual_add(output, shortcut, config.survival_prob, training)

    shortcut = output
    output = self.ffn_branch(output, training)
    if config.dropout:
      output = tf.keras.layers.Dropout(config.dropout, name='after_ffn_drop')(
          output, training=training)
    output = ops.residual_add(output, shortcut, config.survival_prob, training)

    return output


class SE(tf.keras.layers.Layer):
  """Squeeze-and-excitation layer."""

  def __init__(self,
               se_filters,
               output_filters,
               local_pooling=False,
               data_format='channels_last',
               activation='swish',
               kernel_initializer=tf.random_normal_initializer(stddev=0.02),
               bias_initializer=tf.zeros_initializer,
               name='se'):
    super().__init__(name=name)

    self._local_pooling = local_pooling
    self._data_format = data_format
    self._activation_fn = ops.get_act_fn(activation)

    # Squeeze and Excitation layer.
    self._se_reduce = tf.keras.layers.Conv2D(
        se_filters,
        kernel_size=[1, 1],
        strides=[1, 1],
        padding='same',
        data_format=self._data_format,
        use_bias=True,
        kernel_initializer=kernel_initializer,
        bias_initializer=bias_initializer,
        name='reduce_conv2d')
    self._se_expand = tf.keras.layers.Conv2D(
        output_filters,
        kernel_size=[1, 1],
        strides=[1, 1],
        padding='same',
        data_format=self._data_format,
        use_bias=True,
        kernel_initializer=kernel_initializer,
        bias_initializer=bias_initializer,
        name='expand_conv2d')

  def call(self, inputs):
    h_axis, w_axis = [2, 3] if self._data_format == 'channels_first' else [1, 2]
    if self._local_pooling:
      se_tensor = tf.nn.avg_pool(
          inputs,
          ksize=[1, inputs.shape[h_axis], inputs.shape[w_axis], 1],
          strides=[1, 1, 1, 1],
          padding='VALID')
    else:
      se_tensor = tf.reduce_mean(inputs, [h_axis, w_axis], keepdims=True)
    se_tensor = self._se_expand(self._activation_fn(self._se_reduce(se_tensor)))
    return tf.sigmoid(se_tensor) * inputs


class MBConvBlock(tf.keras.layers.Layer):
  """Mobile Inverted Residual Bottleneck."""

  def _retrieve_config(self, config):
    required_keys = ['hidden_size']
    optional_keys = {
        'downsample_loc': 'depth_conv',
        'data_format': 'channels_last',
        'kernel_size': 3,
        'expansion_rate': 4,
        'se_ratio': 0.25,
        'activation': 'gelu',
        'pool_type': 'avg',
        'pool_stride': 1,
        'dropcnn': None,
        'survival_prob': None,
        'norm_type': 'tpu_batch_norm',
        'bn_epsilon': 1e-3,
        'bn_momentum': 0.99,
        'bn_group_size': None,
        'kernel_initializer': tf.random_normal_initializer(stddev=0.02),
        'bias_initializer': tf.zeros_initializer,
    }
    config = create_config_from_dict(config, required_keys, optional_keys)
    return config

  def __init__(self, config, name='mbconv'):
    super().__init__(name=name)

    self._config = self._retrieve_config(config)

    self._activation_fn = ops.get_act_fn(self._config.activation)

    self.endpoints = None

  def build(self, input_shape):
    """Builds block according to the arguments."""
    config = self._config

    channel_axis = 3 if config.data_format == 'channels_last' else 1
    input_size = input_shape[channel_axis]
    inner_size = config.hidden_size * config.expansion_rate

    bsz_per_shard = input_shape.as_list()[0]
    norm_class = config_batch_norm(config, bsz_per_shard)

    # Shortcut projection.
    if input_size != config.hidden_size:
      self._shortcut_conv = tf.keras.layers.Conv2D(
          filters=config.hidden_size,
          kernel_size=1,
          strides=1,
          padding='same',
          data_format=config.data_format,
          kernel_initializer=config.kernel_initializer,
          bias_initializer=config.bias_initializer,
          use_bias=True,
          name='shortcut_conv')
    else:
      self._shortcut_conv = None

    # Pre-Activation norm
    self._pre_norm = norm_class(name='pre_norm')

    # Expansion phase.
    if self._config.expansion_rate != 1:
      self._expand_conv = tf.keras.layers.Conv2D(
          filters=inner_size,
          kernel_size=1,
          strides=(config.pool_stride if config.downsample_loc == 'expand_conv'
                   else 1),
          kernel_initializer=config.kernel_initializer,
          padding='same',
          data_format=config.data_format,
          use_bias=False,
          name='expand_conv')
      self._expand_norm = norm_class(name='expand_norm')

    # Depth-wise convolution phase.
    self._depthwise_conv = tf.keras.layers.DepthwiseConv2D(
        kernel_size=config.kernel_size,
        strides=(config.pool_stride if config.downsample_loc == 'depth_conv'
                 else 1),
        depthwise_initializer=config.kernel_initializer,
        padding='same',
        data_format=config.data_format,
        use_bias=False,
        name='depthwise_conv')
    self._depthwise_norm = norm_class(name='depthwise_norm')

    if config.se_ratio is not None and 0 < config.se_ratio <= 1:
      se_filters = max(1, int(config.hidden_size * config.se_ratio))
      self._se = SE(se_filters=se_filters,
                    output_filters=inner_size,
                    data_format=config.data_format,
                    kernel_initializer=config.kernel_initializer,
                    bias_initializer=config.bias_initializer,
                    name='se')
    else:
      self._se = None

    # Output phase.
    self._shrink_conv = tf.keras.layers.Conv2D(
        filters=config.hidden_size,
        kernel_size=1,
        strides=1,
        padding='same',
        data_format=config.data_format,
        kernel_initializer=config.kernel_initializer,
        bias_initializer=config.bias_initializer,
        use_bias=True,
        name='shrink_conv')

  def downsample(self, inputs, name):
    config = self._config
    output = inputs
    if config.pool_stride > 1:
      output = ops.pooling_2d(output,
                              config.pool_type,
                              config.pool_stride,
                              padding='same',
                              data_format=config.data_format,
                              name=name)
    return output

  def shortcut_branch(self, shortcut):
    shortcut = self.downsample(shortcut, name='shortcut_pool')
    if self._shortcut_conv:
      shortcut = self._shortcut_conv(shortcut)

    return shortcut

  def residual_branch(self, inputs, training):
    config = self._config
    output = self._pre_norm(inputs, training=training)
    if config.downsample_loc == 'inputs':
      output = self.downsample(output, name='residual_pool')
    if config.expansion_rate != 1:
      output = self._expand_conv(output)
      output = self._expand_norm(output, training=training)
      output = self._activation_fn(output)
      logging.debug('Expand shape: %s', output.shape)

    output = self._depthwise_conv(output)
    output = self._depthwise_norm(output, training=training)
    output = self._activation_fn(output)
    logging.debug('DConv shape: %s', output.shape)

    if config.dropcnn:
      output = tf.keras.layers.Dropout(config.dropcnn, 'after_dconv_drop')(
          output, training=training)

    if self._se:
      output = self._se(output)
    self.endpoints = {'expansion_output': output}

    output = self._shrink_conv(output)
    logging.debug('Shrink shape: %s', output.shape)

    return output

  def call(self, inputs, training, survival_prob=None):
    """Implementation of call().

    Args:
      inputs: the inputs tensor.
      training: boolean, whether the model is constructed for training.
      survival_prob: float, between 0 to 1, drop connect rate.

    Returns:
      A output tensor.
    """
    logging.debug('Block %s input shape: %s (%s)', self.name, inputs.shape,
                  inputs.dtype)

    residual = self.residual_branch(inputs, training)
    shortcut = self.shortcut_branch(inputs)
    survival_prob = survival_prob or self._config.survival_prob
    output = ops.residual_add(residual, shortcut, survival_prob, training)

    return output


class MaxViTBlock(tf.keras.layers.Layer):
  """MaxViT block = MBConv + Block-Attention + FFN + Grid-Attention + FFN."""

  def _retrieve_config(self, config):
    required_keys = ['hidden_size', 'head_size', 'window_size', 'grid_size']
    optional_keys = {
        'num_heads': None,
        'expansion_rate': 4,
        'activation': 'gelu',
        'pool_type': 'avg',
        'pool_stride': 1,
        'dropatt': None,
        'dropout': None,
        'rel_attn_type': '2d_multi_head',
        'scale_ratio': None,
        'survival_prob': None,
        'ln_epsilon': 1e-5,
        'ln_dtype': None,
        'kernel_initializer': tf.random_normal_initializer(stddev=0.02),
        'bias_initializer': tf.zeros_initializer,
    }
    config = create_config_from_dict(config, required_keys, optional_keys)
    return config

  def __init__(self, config, name='transformer'):
    super().__init__(name=name)

    self._config = self._retrieve_config(config)

  def build(self, input_shape):
    config = self._config

    input_size = input_shape.as_list()[-1]

    if input_size != config.hidden_size:
      self._shortcut_proj = TrailDense(
          config.hidden_size,
          kernel_initializer=config.kernel_initializer,
          bias_initializer=config.bias_initializer,
          name='shortcut_proj')
    else:
      self._shortcut_proj = None

    self._block_attn_layer_norm = tf.keras.layers.LayerNormalization(
        axis=-1,
        epsilon=config.ln_epsilon,
        dtype=config.ln_dtype,
        name='attn_layer_norm')

    self._grid_attn_layer_norm = tf.keras.layers.LayerNormalization(
        axis=-1,
        epsilon=config.ln_epsilon,
        dtype=config.ln_dtype,
        name='attn_layer_norm_1')

    self._block_attention = Attention(
        config.hidden_size,
        config.head_size,
        num_heads=config.num_heads,
        dropatt=config.dropatt,
        rel_attn_type=config.rel_attn_type,
        scale_ratio=config.scale_ratio,
        kernel_initializer=config.kernel_initializer,
        bias_initializer=config.bias_initializer,
        name='attention')

    self._grid_attention = Attention(
        config.hidden_size,
        config.head_size,
        num_heads=config.num_heads,
        dropatt=config.dropatt,
        rel_attn_type=config.rel_attn_type,
        scale_ratio=config.scale_ratio,
        kernel_initializer=config.kernel_initializer,
        bias_initializer=config.bias_initializer,
        name='attention_1')

    self._block_ffn_layer_norm = tf.keras.layers.LayerNormalization(
        axis=-1,
        epsilon=config.ln_epsilon,
        dtype=config.ln_dtype,
        name='ffn_layer_norm')

    self._grid_ffn_layer_norm = tf.keras.layers.LayerNormalization(
        axis=-1,
        epsilon=config.ln_epsilon,
        dtype=config.ln_dtype,
        name='ffn_layer_norm_1')

    self._block_ffn = FFN(
        config.hidden_size,
        dropout=config.dropout,
        expansion_rate=config.expansion_rate,
        activation=config.activation,
        kernel_initializer=config.kernel_initializer,
        bias_initializer=config.bias_initializer,
        name='ffn')

    self._grid_ffn = FFN(
        config.hidden_size,
        dropout=config.dropout,
        expansion_rate=config.expansion_rate,
        activation=config.activation,
        kernel_initializer=config.kernel_initializer,
        bias_initializer=config.bias_initializer,
        name='ffn_1')

    self._mbconv = MBConvBlock(config)

  def downsample(self, inputs, name):
    config = self._config
    output = inputs
    if config.pool_stride > 1:
      output = ops.maybe_reshape_to_2d(output)
      output = ops.pooling_2d(output,
                              config.pool_type,
                              config.pool_stride,
                              padding='same',
                              data_format='channels_last',
                              name=name)
    return output

  def window_partition(self, features):
    """Partition the input feature maps into non-overlapping windows.

    Args:
      features: [B, H, W, C] feature maps.

    Returns:
      Partitioned features: [B, nH, nW, wSize, wSize, c].

    Raises:
      ValueError: If the feature map sizes are not divisible by window sizes.
    """
    config = self._config
    _, h, w, c = features.shape
    window_size = config.window_size

    if h % window_size != 0 or w % window_size != 0:
      raise ValueError(f'Feature map sizes {(h, w)} '
                       f'not divisible by window size ({window_size}).')

    features = tf.reshape(features, (-1,
                                     h // window_size, window_size,
                                     w // window_size, window_size, c))
    features = tf.transpose(features, (0, 1, 3, 2, 4, 5))
    features = tf.reshape(features, (-1, window_size, window_size, c))
    return features

  def window_stitch_back(self, features, window_size, h, w):
    """Reverse window_partition."""
    features = tf.reshape(features, [
        -1, h // window_size, w // window_size, window_size, window_size,
        features.shape[-1]
    ])
    return tf.reshape(
        tf.transpose(features, (0, 1, 3, 2, 4, 5)),
        [-1, h, w, features.shape[-1]])

  def grid_partition(self, features):
    """Partition the input feature maps into non-overlapping windows.

    Args:
      features: [B, H, W, C] feature maps.

    Returns:
      Partitioned features: [B, nH, nW, wSize, wSize, c].

    Raises:
      ValueError: If the feature map sizes are not divisible by window sizes.
    """
    config = self._config
    _, h, w, c = features.shape
    grid_size = config.grid_size
    if h % grid_size != 0 or w % grid_size != 0:
      raise ValueError(f'Feature map sizes {(h, w)} '
                       f'not divisible by window size ({grid_size}).')
    features = tf.reshape(features, (-1,
                                     grid_size, h // grid_size,
                                     grid_size, w // grid_size, c))
    features = tf.transpose(features, (0, 2, 4, 1, 3, 5))
    features = tf.reshape(features, (-1, grid_size, grid_size, c))
    return features

  def grid_stitch_back(self, features, grid_size, h, w):
    """Reverse window_partition."""
    features = tf.reshape(features, [
        -1, h // grid_size, w // grid_size, grid_size,
        grid_size, features.shape[-1]
    ])
    return tf.reshape(
        tf.transpose(features, (0, 3, 1, 4, 2, 5)),
        [-1, h, w, features.shape[-1]])

  def block_shortcut_branch(self, shortcut):
    return shortcut

  def grid_shortcut_branch(self, shortcut):
    return shortcut

  def mbconv_shortcut_branch(self, shortcut):
    shortcut = self.downsample(shortcut, 'shortcut_pool')
    if self._shortcut_proj:
      shortcut = self._shortcut_proj(shortcut)

  def block_attn_branch(self, inputs, training, attn_mask):
    config = self._config
    output = self._block_attn_layer_norm(inputs)
    # If put grid-attention in front, we don't need to downsample.
    # Apply local block-attention
    _, h, w, _ = output.shape
    output = self.window_partition(output)
    output = ops.maybe_reshape_to_1d(output)
    output = self._block_attention(output, training, attn_mask=attn_mask)
    output = self.window_stitch_back(output, config.window_size, h, w)
    return output

  def grid_attn_branch(self, inputs, training, attn_mask):
    config = self._config
    output = self._grid_attn_layer_norm(inputs)
    # Apply global grid
    _, h, w, _ = output.shape
    output = self.grid_partition(output)
    output = ops.maybe_reshape_to_1d(output)
    output = self._grid_attention(output, training, attn_mask=attn_mask)
    output = self.grid_stitch_back(output, config.grid_size, h, w)
    return output

  def block_ffn_branch(self, inputs, training):
    output = self._block_ffn_layer_norm(inputs)
    output = self._block_ffn(output, training)
    return output

  def grid_ffn_branch(self, inputs, training):
    output = self._grid_ffn_layer_norm(inputs)
    output = self._grid_ffn(output, training)
    return output

  def mbconv_branch(self, inputs, training):
    output = self._mbconv(inputs, training=training)
    return output

  def call(self, inputs, training, attn_mask=None):
    logging.debug('Block %s input shape: %s (%s)', self.name, inputs.shape,
                  inputs.dtype)

    config = self._config

    # MBConv
    output = self.mbconv_branch(inputs, training)

    # block self-attention
    shortcut = output
    output = self.block_attn_branch(output, training, attn_mask)
    if config.dropout:
      output = tf.keras.layers.Dropout(
          config.dropout, name='after_block_attn_drop')(
              output, training=training)
    output = ops.residual_add(output, shortcut, config.survival_prob, training)

    shortcut = output
    output = self.block_ffn_branch(output, training)
    if config.dropout:
      output = tf.keras.layers.Dropout(
          config.dropout, name='after_block_ffn_drop_1')(
              output, training=training)
    output = ops.residual_add(output, shortcut, config.survival_prob, training)

    # grid self-attention
    shortcut = output
    output = self.grid_attn_branch(output, training, attn_mask)
    if config.dropout:
      output = tf.keras.layers.Dropout(
          config.dropout, name='after_grid_attn_drop')(
              output, training=training)
    output = ops.residual_add(output, shortcut, config.survival_prob, training)

    shortcut = output
    output = self.grid_ffn_branch(output, training)
    if config.dropout:
      output = tf.keras.layers.Dropout(
          config.dropout, name='after_grid_ffn_drop')(
              output, training=training)
    output = ops.residual_add(output, shortcut, config.survival_prob, training)

    return output


def config_batch_norm(config, bsz_per_shard=None):
  """Get the actual class for batch normalization."""
  channel_axis = 3 if config.data_format == 'channels_last' else 1
  num_shards = ops.tpu_function.get_tpu_context().number_of_shards or 1
  if bsz_per_shard is not None and config.bn_group_size is not None:
    bn_shards_per_group = min(
        int(math.ceil(config.bn_group_size / bsz_per_shard)),
        num_shards)
  else:
    bn_shards_per_group = 8
  if (config.norm_type == 'tpu_batch_norm' and
      bn_shards_per_group < num_shards):
    norm_class = functools.partial(
        TpuBatchNormalization,
        num_shards_per_group=bn_shards_per_group,
        axis=channel_axis,
        momentum=config.bn_momentum,
        epsilon=config.bn_epsilon)
  else:
    norm_class = functools.partial(
        BatchNormalization,
        axis=channel_axis,
        momentum=config.bn_momentum,
        epsilon=config.bn_epsilon)

  return norm_class


class TpuBatchNormalization(tf.keras.layers.BatchNormalization):
  """Cross replica batch normalization."""

  def __init__(self, num_shards_per_group=None, fused=False, **kwargs):
    self.num_shards_per_group = num_shards_per_group or 8
    if not kwargs.get('name', None):
      kwargs['name'] = 'tpu_batch_normalization'
    if fused in (True, None):
      raise ValueError('TpuBatchNormalization does not support fused=True.')
    super().__init__(fused=fused, **kwargs)

  def _moments(self, inputs, reduction_axes, keep_dims):
    """Compute the mean and variance: it overrides the original _moments."""
    shard_mean, shard_variance = super()._moments(
        inputs, reduction_axes, keep_dims=keep_dims)

    num_shards = ops.tpu_function.get_tpu_context().number_of_shards or 1
    num_shards_per_group = min(self.num_shards_per_group, num_shards)
    if num_shards_per_group > 1:
      logging.info('TpuBatchNormalization with num_shards_per_group %d',
                   num_shards_per_group)
      shard_square_of_mean = tf.math.square(shard_mean)
      shard_mean_of_square = shard_variance + shard_square_of_mean
      group_mean = ops.cross_replica_mean(shard_mean, num_shards_per_group)
      group_mean_of_square = ops.cross_replica_mean(shard_mean_of_square,
                                                    num_shards_per_group)
      group_variance = group_mean_of_square - tf.math.square(group_mean)
      return (group_mean, group_variance)
    else:
      return (shard_mean, shard_variance)

  def call(self, inputs, training=None):
    outputs = super().call(inputs, training)
    for u in self.updates:
      tf.compat.v1.add_to_collection(tf.compat.v1.GraphKeys.UPDATE_OPS, u)
    return outputs


class BatchNormalization(tf.keras.layers.BatchNormalization):
  """Fixed default name of BatchNormalization to match TpuBatchNormalization."""

  def __init__(self, **kwargs):
    if not kwargs.get('name', None):
      kwargs['name'] = 'tpu_batch_normalization'
    super().__init__(**kwargs)

  def call(self, inputs, training=None):
    outputs = super().call(inputs, training)
    if training and not tf.executing_eagerly():
      for u in self.updates:
        tf.compat.v1.add_to_collection(tf.compat.v1.GraphKeys.UPDATE_OPS, u)
    return outputs


class MaxViT(tf.keras.layers.Layer):
  """MaxViT layer definition."""

  def _retrieve_config(self, config):
    required_keys = ['block_type', 'num_blocks', 'hidden_size', 'stem_hsize']
    optional_keys = {
        # cls head
        'dropcls': None,
        'cls_hsize': -1,
        'cls_bias_init': 0,
        'num_classes': 1000,
        # tfm specific
        'head_size': 32,
        'dropatt': None,
        'dropout': None,
        'rel_attn_type': '2d_multi_head',
        'window_size': 7,
        'grid_size': 7,
        # A string for finetuning at different size, e.g. '384/224'
        'scale_ratio': None,
        'ln_epsilon': 1e-5,
        'ln_dtype': None,
        # conv specific
        'downsample_loc': 'depth_conv',
        'kernel_size': 3,
        'se_ratio': 0.25,
        'dropcnn': None,
        # Only channels_last is supported for now.
        'data_format': 'channels_last',
        'norm_type': 'tpu_batch_norm',
        'bn_epsilon': 1e-3,
        'bn_momentum': 0.99,
        'bn_group_size': None,
        # shared
        'add_pos_enc': False,
        'pool_type': 'avg',
        'pool_stride': 2,
        'expansion_rate': 4,
        'activation': 'gelu',
        'survival_prob': None,
        'survival_prob_anneal': True,
        'kernel_initializer': tf.random_normal_initializer(stddev=0.02),
        'bias_initializer': tf.zeros_initializer,
    }
    config = create_config_from_dict(config, required_keys, optional_keys)

    if isinstance(config.block_type, str):
      config.block_type = [config.block_type]
    if isinstance(config.hidden_size, int):
      config.hidden_size = [config.hidden_size]
    if isinstance(config.stem_hsize, int):
      config.stem_hsize = [config.stem_hsize]

    return config

  def _local_config(self, config, idx, exclude_regex=None):
    config = copy.deepcopy(config)
    for key in config.__dict__:
      if isinstance(config[key], (list, tuple)):
        if not exclude_regex or not re.match(exclude_regex, key):
          config[key] = config[key][idx]
    return config

  def __init__(self, config, name='maxvit'):
    super().__init__(name=name)

    self._config = self._retrieve_config(config)

    self.endpoints = {}

  def build(self, input_shape):
    bsz_per_shard = input_shape.as_list()[0]
    bn_class = config_batch_norm(self._config, bsz_per_shard)

    # Stem
    stem_layers = []
    for i in range(len(self._config.stem_hsize)):
      conv_layer = tf.keras.layers.Conv2D(
          filters=self._config.stem_hsize[i],
          kernel_size=self._config.kernel_size,
          strides=2 if i == 0 else 1,
          padding='same',
          data_format=self._config.data_format,
          kernel_initializer=self._config.kernel_initializer,
          bias_initializer=self._config.bias_initializer,
          use_bias=True,
          name='conv_{}'.format(i))
      stem_layers.append(conv_layer)
      if i < len(self._config.stem_hsize) - 1:
        stem_layers.append(bn_class(name='norm_{}'.format(i)))
        stem_layers.append(tf.keras.layers.Activation(
            ops.get_act_fn(self._config.activation), name='act_{}'.format(i)))
    self._stem = tf.keras.Sequential(
        layers=stem_layers,
        name='stem')

    # Backbone
    self._blocks = []
    total_num_blocks = sum(self._config.num_blocks)
    bid = 0
    for i in range(len(self._config.block_type)):
      self._blocks.append([])
      config_s = self._local_config(self._config, i, '^stem.*')
      for j in range(config_s.num_blocks):
        # block name
        block_name = 'block_{:0>2d}_{:0>2d}'.format(i, j)

        ##### Update per-block config
        # No pooling if not the first block in the stage
        config = copy.deepcopy(config_s)
        if j > 0:
          config = config.replace(pool_stride=1)

        # anneal the survival prob
        survival_prob = self._config.survival_prob
        if survival_prob and self._config.survival_prob_anneal:
          drop_rate = 1.0 - survival_prob
          survival_prob = 1.0 - drop_rate * bid / total_num_blocks
          logging.info('[%02d/%02d] %s survival_prob: %.4f', bid,
                       total_num_blocks, block_name, survival_prob)
          config = config.replace(survival_prob=survival_prob)

        ##### Init block
        if config.block_type == 'maxvit':
          block = MaxViTBlock(config, name=block_name)
        else:
          raise ValueError('Unsupported block_type %s' % config.block_type)
        self._blocks[-1].append(block)

        bid += 1

    # Pre-classification layer norm
    self._final_layer_norm = tf.keras.layers.LayerNormalization(
        axis=-1,
        epsilon=config.ln_epsilon,
        dtype=config.ln_dtype,
        name='final_layer_norm')

    # Classification head
    cls_layers = []
    if self._config.cls_hsize:
      assert isinstance(self._config.cls_hsize, int)
      if self._config.cls_hsize == -1:
        cls_hsize = self._config.hidden_size[-1]
      else:
        cls_hsize = self._config.cls_hsize
      inner_dense = TrailDense(
          cls_hsize,
          kernel_initializer=self._config.kernel_initializer,
          bias_initializer=self._config.bias_initializer,
          name='inner_dense')
      cls_layers.append(inner_dense)
      cls_layers.append(tf.keras.layers.Activation(tf.nn.tanh, name='tanh'))

    logit_dense = TrailDense(
        self._config.num_classes,
        kernel_initializer=self._config.kernel_initializer,
        bias_initializer=tf.constant_initializer(self._config.cls_bias_init),
        name='logit_dense')
    cls_layers.append(logit_dense)

    self._cls_head = tf.keras.Sequential(
        layers=cls_layers,
        name='cls_head')

  def call(self, inputs, training):
    logging.info('Network inputs: shape %s, dtype %s.',
                 inputs.shape, inputs.dtype)
    output = self._stem(inputs, training=training)
    logging.info('Stage 0 (stem) output: shape %s, dtype %s.',
                 output.shape, output.dtype)
    self.endpoints['stage_0'] = output
    self.endpoints['stem'] = output

    for idx, stage_blocks in enumerate(self._blocks):

      # Blocks forward
      for block in stage_blocks:
        output = block(output, training=training)
      logging.info('Stage %d output: shape %s, dtype %s.',
                   idx + 1, output.shape, output.dtype)
      self.endpoints['stage_{}'.format(idx + 1)] = output

    # global average pooling
    reduce_axes = list(range(1, output.shape.rank - 1))
    output = tf.reduce_mean(output, axis=reduce_axes)
    self.endpoints['pooled_features'] = output

    # final layer normalization
    output = self._final_layer_norm(output)
    self.endpoints['normed_features'] = output

    # classification head
    output = self._cls_head(output, training=training)
    self.endpoints['logits'] = output

    return output
