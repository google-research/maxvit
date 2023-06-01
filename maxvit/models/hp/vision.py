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

"""Define configurable."""
import copy
from maxvit.models import hparam_configs
from maxvit.models.hparams_registry import register


@register
class Base():
  cfg = hparam_configs.Config(__maxvit__=1)  # This is for sanity check.


# input processing config
input_cfg = dict(
    augname=None,
    ra_num_layers=2,
    ra_magnitude=15,
    mixup_alpha=0.0,
    cutmix_alpha=0.0,
    legacy_preprocess=False)

# train/eval config
eval_cfg = dict(split=None,
                image_size=224,
                batch_size=16,
                steps=None)

train_cfg = dict(split=None,
                 image_size=224,
                 epochs=300,
                 steps=None,
                 batch_size=4096,
                 lr_schedule=dict(
                     type='cosine',
                     warmup_steps=10000,
                     warmup_epochs=None,
                     lr_max=1e-3,
                     lr_min=1e-5,
                 ),
                 weight_decay=0.1,
                 ema_decay=0,
                 grad_clip=1.0,
                 optimizer='adamw')

loss_cfg = dict(xent_type='softmax',
                label_smoothing=0.1)


# Base config where the "model" is not specified
# (1) Set up model field
# (2) Update the train field (models use different training settings)
class Vision(Base):
  """Default config for Vision."""
  cfg = copy.deepcopy(Base.cfg)

  # data
  cfg.dataset = 'imagenet'
  cfg.double_transpose = True

  # input preprocessing
  cfg.input = input_cfg

  # loss related
  cfg.loss = loss_cfg

  # train related params.
  cfg.train = train_cfg

  # eval related params.
  cfg.eval = eval_cfg

  # path related params.
  cfg.path = dict(ckpt_dir=None)

  # tpu related params.
  cfg.tpu = dict(iterations_per_loop=5000,
                 save_checkpoints_steps=5000,
                 keep_checkpoint_max=0,
                 keep_checkpoint_every_n_hours=4,
                 use_bfloat16=True)

  # initialization
  cfg.init = dict(warm_start_mode='restore_train',
                  warm_start_from=None)

maxvit_common = dict(
    block_type=['maxvit', 'maxvit', 'maxvit', 'maxvit'],
    add_pos_enc=[False, False, False, False],
    downsample_loc='depth_conv')

maxvit_t = dict(
    stem_hsize=[64, 64],
    num_blocks=[2, 2, 5, 2],
    hidden_size=[64, 128, 256, 512],
    window_size=7,
    grid_size=7)

maxvit_s = dict(
    stem_hsize=[64, 64],
    num_blocks=[2, 2, 5, 2],
    hidden_size=[96, 192, 384, 768],
    window_size=7,
    grid_size=7)

maxvit_b = dict(
    stem_hsize=[64, 64],
    num_blocks=[2, 6, 14, 2],
    hidden_size=[96, 192, 384, 768],
    window_size=7,
    grid_size=7)

maxvit_l = dict(
    stem_hsize=[128, 128],
    num_blocks=[2, 6, 14, 2],
    hidden_size=[128, 256, 512, 1024],
    window_size=7,
    grid_size=7)

maxvit_xl = dict(
    stem_hsize=[192, 192],
    num_blocks=[2, 6, 14, 2],
    hidden_size=[192, 384, 768, 1536],
    window_size=7,
    grid_size=7)

