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
Transformation: callbacks to Python user functions
---------------------------------------------------

See API and usage description below in `callback`.
Concrete examples are in `tests/mini_jax_callback_test.py`.

"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import itertools
from typing import Any, Dict, Callable, Tuple, Sequence, List, cast

from jax.experimental.mini_jax.mini_jax import (
  Expr, ExprType,
  Value, CustomOperator, PrettyPrint, pp_str, pp_seq,
  Globals, unzip
)
from jax.experimental.mini_jax.mini_jax_jit import Jit


class CallbackOp(CustomOperator):
  """We implement callbacks using a custom operator.

  The operator always has multiple results, and carries as parameters:
    * `func`: the Python callable, with signature (*args, transforms: Tuple[str]).
    * `transforms`: a tuple of transformations that have been applied, the last
       one being the first.
  The implementation is as for the identity function.
  """
  def __init__(self):
    super().__init__("callback", "func", "transforms",
                     defer_eval=True)

  def type_check(self, params: Dict, args_t: Sequence[ExprType]) -> Sequence[ExprType]:
    return tuple(args_t)

  def eval_concrete(self, params: Dict, args: Sequence[Value]) -> Sequence[Value]:
    # We only call the callback if we are not tracing
    assert Globals.scope_nesting_depth == 0
    params["func"].eval_concrete(args, transforms=params["transforms"])
    return tuple(args)

  def compile_assigned(self, params: Dict, args_s: Sequence[str],
                       e_types: Sequence[ExprType],
                       name: str) -> PrettyPrint:
    func = params['func']
    func_pp = func.register_func()
    pp_args = pp_seq(args_s, hsep=", ")
    return (func_pp >> pp_str("(") >> pp_args >>
            pp_str(f", transforms={params['transforms']}); ") >>
            pp_str(f"{name} = (") >> pp_args >> pp_str(",)"))

  def eval_jvp(self, params: Dict, args_v: Sequence[Value], args_tan: Sequence[Value]) -> Sequence[Value]:
    args_and_tan = (*args_v, *args_tan)
    # Use the callback even for the transformed code
    return self.invoke(*args_and_tan, func=params["func"],
                       transforms=("jvp",) + params["transforms"])

  def eval_vjp(self, params: Dict, args: Sequence['Expr'], out_adj: Sequence[Value],
               eval_std_expr: Callable[['Expr'], Value]) -> Sequence[Value]:
    # Use the callback even for the transformed code
    return self.invoke(*out_adj, func=params["func"],
                       transforms=("vjp",) + params["transforms"])

  def eval_count_flops(self, params: Dict, args: Sequence['Expr'],
                       eval_std_expr: Callable[['Expr'], Value]) -> Value:
    return 1.

  def eval_vmap(self, params: Dict, args_withb: Sequence[Tuple[Value, bool]],
                batch_size:int) -> Tuple[Sequence[Value], Sequence[bool]]:
    args_b, args_hasb = unzip(args_withb)
    return self.invoke(*args_b, func=params["func"],
                       transforms=("vmap",) + params["transforms"]), tuple(args_hasb)

_callback_op = CallbackOp()


class  _CallbackFunc(object):
  """Wrapper for Python callables used as operator parameters in "callback"."""
  _counter = itertools.count()
  _callback_dict: Dict[str, Any] = dict()  # This is leaking

  GLOBAL_CALLBACK_NAME = "__callback_dict"

  def __init__(self, func: Callable):
    self.func = func
    self.id = next(_CallbackFunc._counter)

  def __eq__(self, other):
    return self.func == other.func

  def __repr__(self):
    return f"cb{self.id}"
  __str__ = __repr__

  def register_func(self) -> PrettyPrint:
    """Register as a function in the JIT exec context.
    Returns the PrettyPrint for Python source that denotes the callaback to be invoked.
    """
    Jit.register_global_exec_context(_CallbackFunc.GLOBAL_CALLBACK_NAME, _CallbackFunc._callback_dict)
    _CallbackFunc._callback_dict[str(self.id)] = self.func
    return pp_str(f"{_CallbackFunc.GLOBAL_CALLBACK_NAME}['{self.id}']")

  def eval_concrete(self, args: Sequence[Value], transforms=()) -> None:
    self.func(*args, transforms=transforms)

def _reset_counters():
  """For deterministic tests"""
  _CallbackFunc._counter = itertools.count()


def callback(func: Callable) -> Callable:
  """Wrap a function into an identity with a callback.

  The returned function acts as the identity function, except that it
  calls the provided Python callable with the values of the arguments.
  The identity function always returns a tuple, even if it gets a single
  argument.

  The Python callable is called only on concrete arguments (after all
  transformations), even if inside a JIT, through a callback from the
  compiled code back into the user code. On transformations (jvp, grad),
  the behavior is as for the identity function, except that the result is
  passed through the same callback.

  This can be useful for debugging, for printing values or for breaking
  into a debugger::

    def print_callback(*args, transforms=()):
      print(f"args={args} (transforms={transforms})")
    print_callback = mj.callback(print_callback)
    def double_if_positive(x):
      return 3. * mj.Ops.cond_ge(x,
          lambda x: print_callback(x * 2.)[0], (x,),
          lambda _: 0., (0.,))

    >>> double_if_positive(3.)
    args=(6.0,)
    >>>double_if_positive(-3.)
    >>>mj.jit(double_if_positive)(4.)
    args=(8.0,)  # Even from within the JIT
    >>> mj.jit(double_if_positive)(-4.)
    # The branch is not taken, nothing is printed
    >>> mj.jvp(double_if_positive)(5., 0.1)
    args=(10.0, 0.2)  # Prints the arguments (x * 2), and the tangents
    >>> mj.grad(double_if_positive)(6.)
    args=(6.0,)  # The primal value
    args=(3.0)   # The adjoint propagated back to the argument of `print_callback`
  """
  def do_callback(*args: Value):
    assert len(args) >= 1
    return _callback_op.invoke(*args, func=_CallbackFunc(func),
                               transforms=())

  return do_callback
