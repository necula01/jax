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

import functools
import itertools
from typing import Any, Dict, Callable, Tuple, Sequence, List

from jax.experimental.mini_jax.mini_jax import (
  Expr, ExprType, Operator, Function, Tracer,
  Value
)
from jax.experimental.mini_jax.mini_jax_util import map_list, unzip


class Jvp(object):
  """Methods related to forward differentiation."""

  def eval_function(self,
                    func: Function,
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
    nr_invars = len(func.invars)
    args, args_tan = args_and_tan[0:nr_invars], args_and_tan[nr_invars:]
    assert len(args) == len(args_tan)

    # Map variables to primal and tangent values
    eval_jvp = func.make_evaluator(list(zip(args, args_tan)),
                                   eval_operator=self.eval_operator)
    res_and_tan = [eval_jvp(res) for res in func.results]
    res, res_tan = unzip(res_and_tan)
    return tuple(res + res_tan)

  def eval_operator(self,
                    op: Operator,
                    params: Dict,
                    args_with_tan: Sequence[Tuple[Value, Value]]
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
    args_v, args_tan = unzip(args_with_tan)
    if op == Operator.VAR:
      assert False
    if op == Operator.LITERAL:
      return (params["val"], 0.)
    if op == Operator.ADD:
      r = args_v[0] + args_v[1]
      r_tan = args_tan[0] + args_tan[1]
      return (r, r_tan)
    if op == Operator.SUB:
      r = args_v[0] - args_v[1]
      r_tan = args_tan[0] - args_tan[1]
      return (r, r_tan)
    if op == Operator.MUL:
      r = args_v[0] * args_v[1]
      r_tan = (args_tan[0] * args_v[1] +
               args_v[0] * args_tan[1])
      return (r, r_tan)
    if op == Operator.POW:
      pow = params["pow"]
      r = args_v[0] ** pow
      if pow == 0:
        r_tan = 0.
      else:
        r_tan = float(pow) * args_v[0] ** (pow - 1) * args_tan[0]
      return (r, r_tan)
    if op == Operator.PROJECTION:
      # We use PROJECTION when we have multiple-outputs. But in that
      # case the outputs are (out1, out2, ..., out_tan1, out_tan_2, ...)
      idx = params["idx"]
      args_v = args_with_tan[0]  # All args are in the first one
      assert len(args_v) % 2 == 0
      return (args_v[idx], args_v[idx + len(args_v) // 2])

    if op == Operator.JIT_CALL:
      # The JVP calculation has to also be under JIT
      func = params["func"]
      # For each invar we add also a tangent var, with the same ExprType
      jvp_func = func.transform_function(
        "jvp", Jvp().eval_function,
        extra_args_typ=[inv.etype for inv in func.invars])
      # Either build an Expr (if the args are tracers), or execute the JIT
      return Expr.eval_std_operator(Operator.JIT_CALL,
                                    dict(func=jvp_func),
                                    args_v + args_tan)

    if op == Operator.COND_GE:
      true_func_f = params["true_func"]
      true_func_jvp = true_func_f.transform_function(
        "jvp", Jvp().eval_function,
        extra_args_typ=[inv.etype for inv in true_func_f.invars])
      false_func_f = params["false_func"]
      false_func_jvp = false_func_f.transform_function(
        "jvp", Jvp().eval_function,
        extra_args_typ=[inv.etype for inv in false_func_f.invars])
      # Either build an Expr (if the args are tracers), or execute the conditional
      return Expr.eval_std_operator(
        Operator.COND_GE,
        dict(true_func=true_func_jvp,
             false_func=false_func_jvp),
        (args_v[0:1 + len(true_func_f.invars)] +  # pred and true_args
         args_tan[1:1 + len(true_func_f.invars)] +  # true_args_tan
         args_v[1 + len(true_func_f.invars):] +  # false_args
         args_tan[1 + len(true_func_f.invars):]))  # false_args_tan
      return res_with_tan

    raise NotImplementedError


def jvp(func: Callable, abstract: bool = True, cache: bool = True
        ) -> Callable[..., Function]:
  """
  Args:
    func: a function of n-scalar arguments.
    abstract: whether to force tracer arguments to be abstract.
    cache: whether to allow the caching of the result of tracing (only meaningful
      if `abstract`)
  Returns: a function that when applied to `func` arguments, followed by other
    `n` tangent arguments, returns a sequence of the function's results
    followed by their tangents.
  """

  def wrapped_jvp(*args_and_tangents: Sequence[Value]):
    assert len(args_and_tangents) % 2 == 0
    nr_args = len(args_and_tangents) // 2  # Arguments expected
    args = args_and_tangents[0:nr_args]
    # Trace the function on the arguments only
    func_f, func_f_env = Function.trace_user_function(func, args,
                                                      abstract=abstract,
                                                      cache=cache)
    if func_f_env:
      # Add also arguments for the freevars, with 0. tangents because for the point of view
      # of the current JVP we are only differentiating w.r.t. the arguments, not the freevars.
      args_and_tangents = itertools.chain(args, func_f_env,
                                          args_and_tangents[nr_args:],
                                          [0. for c in func_f_env])
    res = Jvp().eval_function(func_f, *args_and_tangents)
    return res

  return wrapped_jvp
