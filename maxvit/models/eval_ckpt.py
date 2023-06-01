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

import json
import sys
import ast
import numpy as np
import tensorflow.compat.v1 as tf
import maxvit.models.utils as e_utils
import maxvit.models.maxvit as layers
import maxvit.models.hparams as hparams


_MEAN_RGB = [0.485 * 255, 0.456 * 255, 0.406 * 255]
_STDDEV_RGB = [0.229 * 255, 0.224 * 255, 0.225 * 255]


def set_precision_policy(policy_name=None):
  """Set precision policy according to the name.

  Args:
    policy_name: precision policy name, one of 'float32', 'mixed_float16',
      'mixed_bfloat16', or None.
  """
  if not policy_name:
    return

  assert policy_name in ('mixed_float16', 'mixed_bfloat16', 'float32')
  print('use mixed precision policy name %s', policy_name)
  tf.keras.layers.enable_v2_dtype_behavior()
  tf.keras.mixed_precision.set_global_policy(policy_name)


def build_model(images, model_name, training, image_size=224,
                use_bfloat16=False, output_type='logits', overrides=None):
  """Build MaxViT model."""
  config = hparams.lookup(model_name)

  if int(image_size) == 224:
    config.model.window_size = 7
    config.model.grid_size = 7
    config.model.scale_ratio = None
  elif int(image_size) == 384:
    config.model.window_size = 12
    config.model.grid_size = 12
    config.model.scale_ratio = '384/224'
  elif int(image_size) == 512:
    config.model.window_size = 16
    config.model.grid_size = 16
    config.model.scale_ratio = '512/224'

  if use_bfloat16:
    if images.dtype != tf.float16:
      images = tf.cast(images, tf.bfloat16)
    set_precision_policy('mixed_bfloat16')
    with tf.tpu.bfloat16_scope():
      model = layers.MaxViT(config.model)
      _ = model(images, training=training)
    set_precision_policy('float32')
  else:
    model = layers.MaxViT(config.model)
    _ = model(images, training=training)

  output = model.endpoints[output_type]

  return output


def get_ema_vars():
  """Get all exponential moving average (ema) variables."""
  ema_vars = tf.trainable_variables() + tf.get_collection('moving_vars')
  for v in tf.global_variables():
    # We maintain batch norm moving mean and variance as well.
    if 'moving_mean' in v.name or 'moving_variance' in v.name:
      ema_vars.append(v)
  return list(set(ema_vars))


class EvalCkptDriver(object):
  """A driver for running eval inference.
  Attributes:
    model_name: str. Model name to eval.
    batch_size: int. Eval batch size.
    image_size: int. Input image size, determined by model name.
    num_classes: int. Number of classes, default to 1000 for ImageNet.
    include_background_label: whether to include extra background label.
    advprop_preprocessing: whether to use advprop preprocessing.
    legacy_preprocessing: whether to use legacy preprocessing.
  """

  def __init__(self,
               model_name,
               model_input_size=224,
               batch_size=1,
               image_size=224,
               num_classes=1000,
               include_background_label=False,
               advprop_preprocessing=False,
               legacy_preprocessing=True):
    """Initialize internal variables."""
    self.model_name = model_name
    self.model_input_size = model_input_size
    self.batch_size = batch_size
    self.num_classes = num_classes
    self.include_background_label = include_background_label
    self.image_size = image_size
    self.advprop_preprocessing = advprop_preprocessing
    self.legacy_preprocessing = legacy_preprocessing

  def restore_model(self, sess, ckpt_dir, enable_ema=True, export_ckpt=None):
    """Restore variables from checkpoint dir."""
    sess.run(tf.global_variables_initializer())
    checkpoint = tf.train.latest_checkpoint(ckpt_dir)
    if enable_ema:
      ema = tf.train.ExponentialMovingAverage(decay=0.0)
      ema_vars = get_ema_vars()
      var_dict = ema.variables_to_restore(ema_vars)
      ema_assign_op = ema.apply(ema_vars)
    else:
      var_dict = get_ema_vars()
      ema_assign_op = None

    tf.train.get_or_create_global_step()
    sess.run(tf.global_variables_initializer())
    saver = tf.train.Saver(var_dict, max_to_keep=1)
    saver.restore(sess, checkpoint)

    if export_ckpt:
      if ema_assign_op is not None:
        sess.run(ema_assign_op)
      saver = tf.train.Saver(max_to_keep=1, save_relative_paths=True)
      saver.save(sess, export_ckpt)

  def build_model(self, features, is_training):
    """Build model with input features."""
    del features, is_training
    raise ValueError('Must be implemented by subclasses.')

  def get_preprocess_fn(self):
    raise ValueError('Must be implemented by subclsses.')

  def build_dataset(self, filenames, labels, is_training):
    """Build input dataset."""
    batch_drop_remainder = False
    if 'condconv' in self.model_name and not is_training:
      # CondConv layers can only be called with known batch dimension. Thus, we
      # must drop all remaining examples that do not make up one full batch.
      # To ensure all examples are evaluated, use a batch size that evenly
      # divides the number of files.
      batch_drop_remainder = True
      num_files = len(filenames)
      if num_files % self.batch_size != 0:
        tf.logging.warn('Remaining examples in last batch are not being '
                        'evaluated.')
    filenames = tf.constant(filenames)
    labels = tf.constant(labels)
    dataset = tf.data.Dataset.from_tensor_slices((filenames, labels))

    def _parse_function(filename, label):
      image_string = tf.read_file(filename)
      preprocess_fn = self.get_preprocess_fn()
      image_decoded = preprocess_fn(
          image_string, is_training, image_size=self.image_size)
      image = tf.cast(image_decoded, tf.float32)
      return image, label

    dataset = dataset.map(_parse_function)
    dataset = dataset.batch(self.batch_size,
                            drop_remainder=batch_drop_remainder)

    iterator = dataset.make_one_shot_iterator()
    images, labels = iterator.get_next()
    return images, labels

  def run_inference(self,
                    ckpt_dir,
                    image_files,
                    labels,
                    enable_ema=True,
                    export_ckpt=None):
    """Build and run inference on the target images and labels."""
    label_offset = 1 if self.include_background_label else 0
    with tf.Graph().as_default(), tf.Session() as sess:
      images, labels = self.build_dataset(image_files, labels, False)
      probs = self.build_model(images, is_training=False)
      if isinstance(probs, tuple):
        probs = probs[0]

      self.restore_model(sess, ckpt_dir, enable_ema, export_ckpt)

      prediction_idx = []
      prediction_prob = []
      for _ in range(len(image_files) // self.batch_size):
        out_probs = sess.run(probs)
        idx = np.argsort(out_probs)[::-1]
        prediction_idx.append(idx[:5] - label_offset)
        prediction_prob.append([out_probs[pid] for pid in idx[:5]])

      # Return the top 5 predictions (idx and prob) for each image.
      return prediction_idx, prediction_prob

  def eval_example_images(self,
                          ckpt_dir,
                          image_files,
                          labels_map_file,
                          enable_ema=True,
                          export_ckpt=None):
    """Eval a list of example images.
    Args:
      ckpt_dir: str. Checkpoint directory path.
      image_files: List[str]. A list of image file paths.
      labels_map_file: str. The labels map file path.
      enable_ema: enable expotential moving average.
      export_ckpt: export ckpt folder.
    Returns:
      A tuple (pred_idx, and pred_prob), where pred_idx is the top 5 prediction
      index and pred_prob is the top 5 prediction probability.
    """
    classes = json.loads(tf.gfile.Open(labels_map_file).read())
    pred_idx, pred_prob = self.run_inference(
        ckpt_dir, image_files, [0] * len(image_files), enable_ema, export_ckpt)
    for i in range(len(image_files)):
      print('predicted class for image {}: '.format(image_files[i]))
      for j, idx in enumerate(pred_idx[i]):
        print('  -> top_{} ({:4.2f}%): {}  '.format(j, pred_prob[i][j] * 100,
                                                    classes[str(idx)]))
    return pred_idx, pred_prob

  def eval_imagenet(self, ckpt_dir, imagenet_eval_glob,
                    imagenet_eval_label, num_images, enable_ema, export_ckpt):
    """Eval ImageNet images and report top1/top5 accuracy.
    Args:
      ckpt_dir: str. Checkpoint directory path.
      imagenet_eval_glob: str. File path glob for all eval images.
      imagenet_eval_label: str. File path for eval label.
      num_images: int. Number of images to eval: -1 means eval the whole
        dataset.
      enable_ema: enable expotential moving average.
      export_ckpt: export checkpoint folder.
    Returns:
      A tuple (top1, top5) for top1 and top5 accuracy.
    """
    imagenet_val_labels = [int(i) for i in tf.gfile.Open(imagenet_eval_label)]
    imagenet_filenames = sorted(tf.gfile.Glob(imagenet_eval_glob))
    if num_images < 0:
      num_images = len(imagenet_filenames)
    image_files = imagenet_filenames[:num_images]
    labels = imagenet_val_labels[:num_images]

    pred_idx, _ = self.run_inference(
        ckpt_dir, image_files, labels, enable_ema, export_ckpt)
    top1_cnt, top5_cnt = 0.0, 0.0
    for i, label in enumerate(labels):
      top1_cnt += label in pred_idx[i][:1]
      top5_cnt += label in pred_idx[i][:5]
      if i % 100 == 0:
        print('Step {}: top1_acc = {:4.2f}%  top5_acc = {:4.2f}%'.format(
            i, 100 * top1_cnt / (i + 1), 100 * top5_cnt / (i + 1)))
        sys.stdout.flush()
    top1, top5 = 100 * top1_cnt / num_images, 100 * top5_cnt / num_images
    print('Final: top1_acc = {:4.2f}%  top5_acc = {:4.2f}%'.format(top1, top5))
    return top1, top5


class MaxViTDriver(EvalCkptDriver):
  """A driver for running eval inference."""

  def build_model(self, features, is_training):
    """Build model with input features."""

    if self.legacy_preprocessing:
      features -= tf.constant(
          _MEAN_RGB, shape=[1, 1, 3], dtype=features.dtype)
      features /= tf.constant(
          _STDDEV_RGB, shape=[1, 1, 3], dtype=features.dtype)
    else:
      features = (features - 127.5) / 127.5  # normalize to [-1, 1]

    logits = build_model(
        features, self.model_name, is_training, self.model_input_size)
    probs = tf.nn.softmax(logits)
    probs = tf.squeeze(probs)
    return probs

  def get_preprocess_fn(self):
    """Build input dataset."""
    return e_utils.preprocess_image