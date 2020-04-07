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

  def eval_vmap(self, params: Dict, args_withb: Sequence[Tuple[Value, bool]],
                batch_size:int) -> Tuple[Sequence[Value], Sequence[bool]]:
    arg, arg_hasb = args_withb[0]
    assert arg_hasb
    return self.invoke(arg, **params), (arg_hasb,)

customPowerOp = _CustomPowerOp()


class _BroadcastInDim(CustomOperator):
  """Broadcasting in a given dimension.

  If
    res = bcast(a, dim, dim_sz)
  then
    * res.shape = a.shape with dim_sz in position dim
    * res[I,idim,J] = a[I, J] for all idim in range(dim_sz)
  """
  def __init__(self):
    super().__init__("bcast", "dim", "dim_sz")

  def _result_shape(self, arg_shape: Shape, dim: int, dim_sz: int) -> Shape:
    return arg_shape[0:dim] + (dim_sz,) + arg_shape[dim:]

  def type_check(self, params: Dict, args_t: Sequence[ExprType]) -> Sequence[ExprType]:
    if not(len(args_t) == 1 and args_t[0].dtype is float):
      raise TypeError(f"Unexpected argument types for {self}: {args_t}")
    a_shape = np.shape(args_t[0])
    dim = params["dim"]
    if dim < 0 or dim > len(a_shape):
      raise TypeError(f"Unexpected 'dim' ({dim}) for argument of shape {a_shape}")
    return (ExprType(self._result_shape(a_shape, dim, params["dim_sz"]), float),)

  def eval_concrete(self, params: Dict, args: Sequence[Value]) -> Sequence[Value]:
    arg, = args
    dim = params["dim"]
    arg_reshaped = np.reshape(arg, self._result_shape(np.shape(arg), dim, 1))
    return (np.broadcast_to(arg_reshaped, self._result_shape(np.shape(arg),
                                                             dim, params["dim_sz"])),)

  def compile_assigned(self, params: Dict, args_s: Sequence[str],
                       e_types: Sequence[ExprType],
                       name: str) -> PrettyPrint:
    arg_s, = args_s
    dim = params["dim"]
    result_shape = e_types[0].shape
    reshape_shape = result_shape[0:dim] + (1,) + result_shape[dim+1:]
    return pp_str(f"{name} = (np.broadcast_to(np.reshape({arg_s}, {reshape_shape}), {result_shape}),)")

  def eval_jvp(self, params: Dict, args_v: Sequence[Value], args_tan: Sequence[Value]) -> Sequence[Value]:
    return (self.invoke_single(args_v[0], **params), self.invoke_single(args_tan[0], **params))

  def eval_vjp(self, params: Dict, args: Sequence['Expr'], out_adj: Sequence[Value],
               eval_std_expr: Callable[['Expr'], Value]) -> Sequence[Value]:
    return sumDimOp.invoke(out_adj[0], dim=params["dim"])

  def eval_count_flops(self, params: Dict, args: Sequence['Expr'],
                       eval_std_expr: Callable[['Expr'], Value]) -> Value:
    # Pretend we copy everything
    orig_size = np.prod(args[0].etype.shape)
    return orig_size * (params["dim_sz"] - 1)

  def eval_vmap(self, params: Dict, args_withb: Sequence[Tuple[Value, bool]],
                batch_size:int) -> Tuple[Sequence[Value], Sequence[bool]]:
    arg, arg_hasb = args_withb[0]
    assert arg_hasb, "Should not get here unless the argument is batched"
    dim = params["dim"]
    return self.invoke(arg, dim=dim + 1, dim_sz=params["dim_sz"]), (True,)

broadcastInDimOp = _BroadcastInDim()


def broadcast_value(batch_size: int, v: Value) -> Value:
  """Broadcast a value on the leading axis."""
  return broadcastInDimOp.invoke_single(v, dim=0, dim_sz=batch_size)

def broadcast_values(batch_size: int,
                     args_b: Sequence[Value],
                     args_hasb: Sequence[bool]) -> Tuple[Sequence[Value], bool]:
  """Prepares the arguments to have the same batching."""
  if all(not a_hasb for a_hasb in args_hasb):
    return args_b, False
  return tuple([a if a_hasb else broadcast_value(batch_size, a)
                for a, a_hasb in zip(args_b, args_hasb)]), True

class _SumDim(CustomOperator):
  """Summing one dimension.

  If
    res = sumdim(a, dim)
  then
    * if a.shape = (I,dim_sz,J) then res.shape = (I, J)
    * res[I,J] = SUM a[I, i, J] for all i in range(dim_sz)
  """

  def __init__(self):
    super().__init__("sumdim", "dim")

  def _result_shape(self, arg_shape: Shape, dim: int) -> Shape:
    return arg_shape[0:dim] + arg_shape[dim+1:]

  def type_check(self, params: Dict, args_t: Sequence[ExprType]) -> Sequence[ExprType]:
    if not(len(args_t) == 1 and args_t[0].dtype is float):
      raise TypeError(f"Unexpected argument types for {self}: {args_t}")
    a_shape = args_t[0].shape
    dim = params["dim"]
    if dim < 0 or dim >= len(a_shape):
      raise TypeError(f"{self}: Unexpected 'dim' ({dim}) for argument of shape {a_shape}")
    return (ExprType(self._result_shape(a_shape, dim), float),)

  def eval_concrete(self, params: Dict, args: Sequence[Value]) -> Sequence[Value]:
    res = np.sum(args[0], axis=params["dim"])
    assert np.shape(res) == self._result_shape(np.shape(args[0]), params["dim"])
    return (res,)

  def compile_assigned(self, params: Dict, args_s: Sequence[str],
                       e_types: Sequence[ExprType],
                       name: str) -> PrettyPrint:
    return pp_str(f"{name} = (np.sum({args_s[0]}, axis={params['dim']}),)")

  def eval_jvp(self, params: Dict, args_v: Sequence[Value], args_tan: Sequence[Value]) -> Sequence[Value]:
    return (self.invoke_single(args_v[0], **params),
            self.invoke_single(args_tan[0], **params))

  def eval_vjp(self, params: Dict, args: Sequence['Expr'], out_adj: Sequence[Value],
               eval_std_expr: Callable[['Expr'], Value]) -> Sequence[Value]:
    a_shape = args[0].etype.shape
    dim = params["dim"]
    return broadcastInDimOp.invoke(out_adj[0], dim=dim, dim_sz=a_shape[dim])

  def eval_count_flops(self, params: Dict, args: Sequence['Expr'],
                       eval_std_expr: Callable[['Expr'], Value]) -> Value:
    return float(np.prod(args[0].etype.shape))

sumDimOp = _SumDim()

class _WhereGe(CustomOperator):
  """An implementation of numpy.where.

  For
    res = where_ge(x, t, f)
  then:
    * x, t, f must have the same shape, and have dtype float
    * res[I] = t[I] if x[I] >= 0 else f[I]
  """
  def __init__(self):
    super().__init__("where_ge")

  def type_check(self, params: Dict, args_t: Sequence[ExprType]) -> Sequence[ExprType]:
    if not(len(args_t) == 3 and all([at.dtype is float for at in args_t])):
      raise TypeError(f"Unexpected argument types for {self}: {args_t}")
    arg_shapes = {np.shape(at) for at in args_t}
    if len(arg_shapes) != 1:
      raise TypeError(f"All arguments must have same shape for {self}: {args_t}")
    return (args_t[1],)

  def eval_concrete(self, params: Dict, args: Sequence[Value]) -> Sequence[Value]:
    return (np.where(args[0] >= 0., args[1], args[2]),)

  def compile_assigned(self, params: Dict, args_s: Sequence[str],
                       e_types: Sequence[ExprType],
                       name: str) -> PrettyPrint:
    return pp_str(f"{name} = (np.where({args_s[0]} >= 0, {args_s[1]}, {args_s[2]}),)")

  def eval_jvp(self, params: Dict, args_v: Sequence[Value], args_tan: Sequence[Value]) -> Sequence[Value]:
    return (self.invoke_single(args_v[0], args_v[1], args_v[2]),
            self.invoke_single(args_v[0], args_tan[1], args_tan[2]))

  def eval_vjp(self, params: Dict, args: Sequence['Expr'], out_adj: Sequence[Value],
               eval_std_expr: Callable[['Expr'], Value]) -> Sequence[Value]:
    assert np.shape(out_adj[0]) == args[0].etype.shape
    arg0 = eval_std_expr(args[0])  # Need the primal value
    zeros = zero_like(arg0)
    return (zeros,
            self.invoke_single(arg0, out_adj[0], zeros),
            self.invoke_single(arg0, zeros, out_adj[0]))

  def eval_count_flops(self, params: Dict, args: Sequence['Expr'],
                       eval_std_expr: Callable[['Expr'], Value]) -> Value:
    cond_shape = args[0].etype.shape
    if not cond_shape: return 1.
    return cond_shape[0]

  def eval_vmap(self, params: Dict, args_withb: Sequence[Tuple[Value, bool]],
                batch_size:int) -> Tuple[Sequence[Value], Sequence[bool]]:
    args_b, args_hasb = unzip(args_withb)
    args_all_b, res_hasb = broadcast_values(batch_size, args_b, args_hasb)
    assert res_hasb
    return self.invoke(*args_all_b), (res_hasb,)

whereGeOp = _WhereGe()