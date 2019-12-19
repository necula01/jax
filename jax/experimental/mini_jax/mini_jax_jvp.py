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
"""
Transformation: Forward differentiation
---------------------------------------

For forward differentiation we write a custom evaluator that evaluates
expressions to a pair of values, the primal result and the tangent result,
given pairs of values for the variables.

Concrete examples are in `tests/mini_jax_jvp_test.py`.
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import itertools
from typing import Any, Dict, Callable, Tuple, Sequence, List

from jax.experimental.mini_jax.mini_jax import (
  Expr, ExprType, Operator, Function, Tracer,
  Value,
)
from jax.experimental.mini_jax.mini_jax_util import map_list, unzip


class Jvp(object):
  """Methods related to forward differentiation."""

  def _eval_jvp_func(func: Function,
                     *args_and_tan: Sequence[Value]
                     ) -> Sequence[Value]:
    """Evaluates the JVP given Values for invars followed by the corresponding
    tangents.

    Args:
      func: the function whose JVP to evaluate
      args_and_tan: a sequence of the arguments followed by as many tangents
    Returns:
       a sequence of all the results followed by as many tangents.
    """
    assert len(args_and_tan) == 2 * len(func.invars)
    nr_invars = len(func.invars)
    # Map variables to primal and tangent values
    env = {}  # type: Dict[Expr, Tuple[Value, Value]]
    for i, iv in enumerate(func.invars):
      env[iv.params["id"]] = (args_and_tan[i], args_and_tan[nr_invars + i])

    def visitor_jvp(e: Expr, args_v: Sequence[Tuple[Value, Value]]
                    ) -> Tuple[Value, Value]:
      return Jvp._eval_jvp_expr(e, args_v, env)

    res_and_tan = func.visit_bodies_memoized(visitor_jvp)
    res, res_tan = unzip(res_and_tan)
    return tuple(res + res_tan)

  @staticmethod
  def _eval_jvp_expr(e: Expr,
                     args_with_tan: Sequence[Tuple[Value, Value]],
                     env: Dict[int, Tuple[Value, Value]]
                     ) -> Tuple[Value, Value]:
    """Evaluates the JVP of an operator application.
    Args:
      e: the expression to evaluate
      args_with_tan: for each operator argument, a pair of primal `Value`
        and tangent `Value`.
      env: primal and tangents values for the variables (by var id).
    Returns:
      a pair of primal result `Value` and tangent result `Value`.
    """
    if e.operator == Operator.VAR:
      return env[e.params["id"]]
    if e.operator == Operator.LITERAL:
      return (e.params["val"], 0.)
    if e.operator == Operator.ADD:
      r = args_with_tan[0][0] + args_with_tan[1][0]
      r_tan = args_with_tan[0][1] + args_with_tan[1][1]
      return (r, r_tan)
    if e.operator == Operator.SUB:
      r = args_with_tan[0][0] - args_with_tan[1][0]
      r_tan = args_with_tan[0][1] - args_with_tan[1][1]
      return (r, r_tan)
    if e.operator == Operator.MUL:
      r = args_with_tan[0][0] * args_with_tan[1][0]
      r_tan = args_with_tan[0][1] * args_with_tan[1][0] + args_with_tan[0][0] * \
              args_with_tan[1][1]
      return (r, r_tan)
    if e.operator == Operator.POW:
      r = args_with_tan[0][0] ** e.params["pow"]
      r_tan = float(e.params["pow"]) * args_with_tan[0][0] ** (
          e.params["pow"] - 1) * args_with_tan[0][1]
      return (r, r_tan)
    if e.operator == Operator.PROJECTION:
      arg_with_tan = args_with_tan[0][e.params["idx"]]
      assert isinstance(arg_with_tan, tuple)
      return arg_with_tan

    if e.operator == Operator.JIT_CALL:
      # The JVP calculation has to also be under JIT
      func = e.params["func"]
      # For each invar we add also a tangent var, with the same ExprType
      args, args_tan = unzip(args_with_tan)
      jvp_func = func.trace_evaluator(
        Jvp._eval_jvp_func,
        extra_args_typ=[inv.etype for inv in func.invars])
      call_res = Expr.eval_std_operator(Operator.JIT_CALL,
                                        dict(func=jvp_func),
                                        args + args_tan)
      # Convert from (res1, res2, res1_tan, res2_tan) to ((res1, res1_tan), (res2, res2_tan))
      if len(func.bodies) > 1:
        return tuple(zip(call_res[0:len(func.bodies)], call_res[len(func.bodies):]))
      else:
        return call_res

    if e.operator == Operator.COND_GE:
      args, args_tan = unzip(args_with_tan)
      true_func_f = e.params["true_func"]
      true_func_jvp = true_func_f.trace_evaluator(
        Jvp._eval_jvp_func,
        extra_args_typ=[inv.etype for inv in true_func_f.invars])
      false_func_f = e.params["false_func"]
      false_func_jvp = false_func_f.trace_evaluator(
        Jvp._eval_jvp_func,
        extra_args_typ=[inv.etype for inv in true_func_f.invars])
      res_with_tan = Expr.eval_std_operator(
        Operator.COND_GE,
        dict(true_func=true_func_jvp,
             false_func=false_func_jvp),
        (args[0:1 + len(true_func_f.invars)] +  # pred and true_args
         args_tan[1:1 + len(true_func_f.invars)] +  # true_args_tan
         args[1 + len(true_func_f.invars):] +  # false_args
         args_tan[1 + len(true_func_f.invars):]))  # false_args_tan
      if len(true_func_f.bodies) > 1:
        # Convert from (res1, res2, res1_tan, res2_tan) to ((res1, res1_tan), (res2, res2_tan))
        return tuple(zip(res_with_tan[0:len(true_func_f.bodies)],
                         res_with_tan[len(true_func_f.bodies):]))
      else:
        return res_with_tan

    raise NotImplementedError


def jvp(func: Callable) -> Callable[..., Function]:
  """
  Args:
    func: a function of n-scalar arguments.
  Returns: a function that when applied to `func` arguments, followed by other
    `n` tangent arguments, returns a sequence of the function's results
    followed by their tangents.
  """

  def wrapped_jvp(*args_and_tangents: Sequence[Value]):
    assert len(args_and_tangents) % 2 == 0
    nr_args = len(args_and_tangents) // 2  # Arguments expected
    args = args_and_tangents[0:nr_args]
    # Trace the function on the arguments only
    func_f, func_f_env = Function.trace_callable(
      func, map_list(Tracer.val_to_type, args))
    if func_f_env:
      # Add also arguments for the freevars, with 0. tangents because for the point of view
      # of the current JVP we are only differentiating w.r.t. the arguments, not the freevars.
      args_and_tangents = itertools.chain(args, func_f_env,
                                          args_and_tangents[nr_args:],
                                          [0. for c in func_f_env])
    return Jvp._eval_jvp_func(func_f, *args_and_tangents)

  return wrapped_jvp
