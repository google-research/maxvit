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

"""ImageNet1K hparams."""
import copy
from maxvit.models.hp import vision
from maxvit.models.hparams_registry import register


i1k_pt_input = dict(
    augname='randaug',
    ra_num_layers=2,
    ra_magnitude=15,
    mixup_alpha=0.8,
    cutmix_alpha=0.0,
    legacy_preprocess=True)


class MaxViTI1KBase(vision.Vision):
  cfg = copy.deepcopy(vision.Vision.cfg)
  # model config
  cfg.model = vision.maxvit_common

  # input config
  cfg.dataset = 'imagenet'
  cfg.input.update(i1k_pt_input)

  # train config
  cfg.train = dict(split=None,
                   image_size=224,
                   epochs=300,
                   batch_size=4096,
                   optimizer='adamw',
                   lr_schedule=dict(
                       type='cosine',
                       warmup_steps=10000,
                       warmup_epochs=None,
                       lr_max=3e-3,
                       lr_min=1e-5,
                   ),
                   weight_decay=0.05,
                   weight_decay_exclude='.*(bias|scale|gain|gamma|beta).*',
                   ema_decay=0.9999,
                   grad_clip=1.0,
                   steps=None)


@register
class MaxViTTiny(MaxViTI1KBase):
  cfg = copy.deepcopy(MaxViTI1KBase.cfg)
  cfg.model.update(vision.maxvit_t)
  cfg.model.survival_prob = 0.8


@register
class MaxViTSmall(MaxViTI1KBase):
  cfg = copy.deepcopy(MaxViTI1KBase.cfg)
  cfg.model.update(vision.maxvit_s)
  cfg.model.survival_prob = 0.7


@register
class MaxViTBase(MaxViTI1KBase):
  cfg = copy.deepcopy(MaxViTI1KBase.cfg)
  cfg.model.update(vision.maxvit_b)
  cfg.model.survival_prob = 0.6


@register
class MaxViTLarge(MaxViTI1KBase):
  cfg = copy.deepcopy(MaxViTI1KBase.cfg)
  cfg.model.update(vision.maxvit_l)
  cfg.model.survival_prob = 0.4


############################## 1K Finetuning ##############################
i1k_ft_update = dict(
    dataset='imagenet',
    input=dict(augname='ft_randaug',
               ra_num_layers=2,
               ra_magnitude=15,
               mixup_alpha=0.8,
               cutmix_alpha=0.0,
               legacy_preprocess=False),
    eval=dict(image_size=384),
    train=dict(image_size=384,
               epochs=30,
               batch_size=512,
               lr_schedule=dict(
                   type=None,
                   lr_max=5e-5,
                   lr_min=5e-5,
                   warmup_steps=0,
                   warmup_epochs=None,
               ),
               weight_decay=1e-8),
    model=dict(scale_ratio='384/224'),
    tpu=dict(iterations_per_loop=1000,
             save_checkpoints_steps=1000),
    # Do NOT reuse the classification head
    init=dict(warm_start_mode='init_finetune'),
)


class MaxViTI1KFtBase(MaxViTI1KBase):
  cfg = copy.deepcopy(MaxViTI1KBase.cfg)
  cfg.update(i1k_ft_update)


@register
class MaxViTTinyI1KFt(MaxViTI1KFtBase):
  cfg = copy.deepcopy(MaxViTI1KFtBase.cfg)
  cfg.model.update(vision.maxvit_t)
  cfg.model.window_size = 12
  cfg.model.grid_size = 12
  cfg.model.survival_prob = 0.8
  cfg.init.warm_start_from = (
      ''
  )


@register
class MaxViTTinyI1KFt512(MaxViTI1KFtBase):
  cfg = copy.deepcopy(MaxViTI1KFtBase.cfg)
  cfg.train.image_size = 512
  cfg.eval.image_size = 512
  cfg.model.window_size = 16
  cfg.model.grid_size = 16
  cfg.model.scale_ratio = '512/224'
  cfg.model.survival_prob = 0.8
