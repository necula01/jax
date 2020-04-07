# Copyright 2020 Google LLC
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
"""
Definitions for some custom operators
-------------------------------------

"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import itertools
import math
import numpy as np

from typing import Any, Dict, Callable, Tuple, Sequence, List

# Import some internals for defining custom ops
from jax.experimental.mini_jax.mini_jax import (
  CustomOperator, Expr, ExprType, Value, PrettyPrint, Shape,
  pp_str,
  const_like, zero_like, unzip)

class _CustomPowerOp(CustomOperator):
  """An implementation of exponentiation as a CustomOperator.

  This operator implements x ** exp.
  Usage: customPowerOp.invoke_single(x, exp=3)
  """
  def __init__(self):
    return super().__init__("cpow", "exp")

  def invoke(self, *args: Value, **params: Any) -> Sequence[Value]:
    res = super().invoke(*args, **params)  # For the checking
    # Try some simplifications
    pow = params["exp"]
    if pow == 0:
      return (1.,)
    elif pow == 1:
      return (args[0],)
    else:
      return res

  def type_check(self, params: Dict, args_t: Sequence[ExprType]) -> Sequence[ExprType]:
    if not(len(args_t) == 1 and args_t[0].dtype is float):
      raise TypeError(f"Unexpected argument types for {self}: {args_t}")
    return args_t  # type: ignore[return-value]

  def eval_concrete(self, params: Dict, args: Sequence[Value]) -> Sequence[Value]:
    return (args[0] ** params["exp"],)  # Use the Python implementation

  def compile_assigned(self, params: Dict, args_s: Sequence[str],
                       e_types: Sequence[ExprType],
                       name: str) -> PrettyPrint:
    return pp_str(f"{name} = ({args_s[0]} ** {params['exp']},)")

  def eval_jvp(self, params: Dict, args_v: Sequence[Value], args_tan: Sequence[Value]) -> Sequence[Value]:
    pow = params['exp']
    arg = args_v[0]
    primal_out = self.invoke_single(arg, exp=params["exp"])  # Use the same custom op in the primal
    assert pow > 0  # Special cases are handled in invoke
    # Use the same custom op even in the computation of the tangent
    return (primal_out,
            const_like(float(pow), arg) * self.invoke_single(arg, exp=params["exp"] - 1) * args_tan[0])

  def eval_vjp(self, params: Dict, args: Sequence['Expr'], out_adj: Sequence[Value],
               eval_std_expr: Callable[['Expr'], Value]) -> Sequence[Value]:
    pow = params['exp']
    assert pow > 0
    arg = args[0]
    arg_v = eval_std_expr(arg)  # We need the primal value of the argument
    # Use the same custom op even in the computation of the adjoint
    return (const_like(float(pow), out_adj[0]) * self.invoke_single(arg_v, exp=params["exp"] - 1) * out_adj[0],)


  def eval_count_flops(self, params: Dict, args: Sequence['Expr'],
                       eval_std_expr: Callable[['Expr'], Value]) -> Value:
    # We assume an implementation that repeatedly squares the argument.
    return float(math.ceil(math.log2(params["exp"])))

customPowerOp = _CustomPowerOp()

