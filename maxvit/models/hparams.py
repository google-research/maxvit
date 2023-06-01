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

"""Import all hyper-parameters."""
# pylint: disable=unused-import
from maxvit.models import hparams_registry
from maxvit.models.hp import vision
from maxvit.models.hp import vision_i1k
# pylint: enable=unused-import


def lookup(name, prefix='maxvit:'):
  name = prefix + name
  if name not in hparams_registry.registry_map:
    raise ValueError('{} not registered: {}'.format(
        name, hparams_registry.registry_map.keys()))
  return hparams_registry.registry_map[name].cfg