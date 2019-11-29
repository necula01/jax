# Copyright 2019 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import functools
from typing import Any, Dict, Callable, Tuple, Sequence, List, Optional, Union, TypeVar

from jax import pprint_util as ppu
from jax.pprint_util import PrettyPrint

TA = TypeVar('TA')
TB = TypeVar('TB')

def map_tuple(f: Callable[[TA], TB], s: Sequence[TA]) -> Tuple[TB]:
  return tuple(map(f, s))


def map_list(f: Callable[[TA], TB], s: Sequence[TA]) -> List[TB]:
  return list(map(f, s))


def unzip(lst):
  res1 = []
  res2 = []
  for e in lst:
    res1.append(e[0])
    res2.append(e[1])
  return res1, res2


def pp_str(s: str) -> PrettyPrint: return ppu.pp(s)


def pp_list(lst: List, vertical=False, hsep=" ") -> PrettyPrint:
  """Apply pprint_util.pp to each element of a list and then concatenate
  Args:
    lst: a list of PrettyPrinter, or string
    vertical: if True, concatenate vertically
    hsep: horizontal separator string
  """
  ppl = map_tuple(
    lambda x: x if isinstance(x, PrettyPrint) else pp_str(str(x)),
    lst)
  if vertical:
    return ppu.vcat(ppl)
  else:
    return functools.reduce(lambda x, y: x >> pp_str(hsep) >> y, ppl)