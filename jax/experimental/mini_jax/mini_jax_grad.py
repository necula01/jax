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
Backward differentiation
=========================

For backward differentiation we write a transformation that increments
the adjoints (cotangent) for the subexpressions, given the adjoints for each
expression. This effectively computes vector-Jacobian product.

To achieve the "backward differentiation" effect, we must process first
the top-level expression before processing the sub-expressions (i.e.,
top-down). Care is taken to ensure that for each unique `Expr` object only a
constant number of other `Expr` are constructed, and an `Expr` object is not
processed more than once.

For a `jit_call` operation, the VJP computation is pushed under the
`jit`. For example, considering a `Function` with two inputs `x1` ans `x2`,
and an invocation `func a1 a2`, where `a1` and `a2` are the actual argument
`Value`s. We first produce another `jit_call` that computes the value and the
VJP of `func`: takes as arguments `a1`, `a2`, and also the adjoint of the outputs,
and returns the adjoints for the inputs: `x1_adj` and `x2_adj`.
, i.e., returns three values when evaluated `a1` and `a2`.
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import itertools
from typing import Dict, Callable, Optional, Sequence

from jax.experimental.mini_jax.mini_jax import (
  Expr, ExprType, Operator, Function, Tracer,
  Value,
)
from jax.experimental.mini_jax.mini_jax_util import map_list


class Grad(object):
  """Methods related to backward differentiation."""

  @staticmethod
  def eval_vjp_func_tracer(func: Function,
                           *args_out_adj: Sequence[Value]) -> Sequence[Value]:
    """Evaluates the function and its vector-Jacobian product.

    Care must be taken bound the work by a constant-factor over the number
    of distinct `Expr` in the function. Must use memoization aggressively,
    watching for sharing among `Expr`.

    Args:
      func: the `Function` whose VJP to evaluate
      args_out_adj: the arguments to the function (func.invars) followed by
        the output adjoints (one for each function output).
    Returns:
      a tuple with the adjoints for the inputs (one for each `func.invars`).
    """
    nr_invars = len(func.invars)
    assert len(args_out_adj) == nr_invars + len(func.bodies)
    args = args_out_adj[0:nr_invars]
    out_adj = args_out_adj[nr_invars:]

    # First we build a graph with pointers to `Expr` parents
    # For each `Expr` (by `id`), the list of parents.
    parents = {}  # Map[id(e)] -> Sequence[Expr]
    var_leaves = []  # Sequence[Expr] (all the variables)

    def reverse_expr_graph(e: Expr, parent: Optional[Expr]):
      e_par = parents.get(id(e))
      if e_par is not None:
        if parent is not None:
          e_par.append(parent)  # Do not recurse if seen already
      else:  # First visit
        parents[id(e)] = [parent] if parent is not None else []
        if e.operator == Operator.VAR:
          var_leaves.append(e)
        else:
          [reverse_expr_graph(a, e) for a in e.args]

    adjoints = {}  # Dict[int, Value] - the accumulated adjoints
    for body, body_out_adj in zip(func.bodies, out_adj):
      reverse_expr_graph(body, None)
      # The body may occur more than once
      adjoints[id(body)] = adjoints.get(id(body), 0.) + body_out_adj

    # Build an environment for each variable used in the body of the function
    # Used to evaluate lazily values that are needed from the forward-pass.
    env = {iv.params["id"]: arg_v for iv, arg_v in zip(func.invars, args)}
    eval_expr = Expr.make_memoized_expr_evaluator(env)

    visited = {}  # By id
    def visit_expr_backwards(e: Expr):
      """Visit all expressions, to accumulate adjoints for all expressions."""
      id_e = id(e)
      if id_e in visited: return
      visited[id_e] = True
      [visit_expr_backwards(p) for p in parents[id_e]]
      # Now that all parents have been processed, our adjoint is ready
      e_adj = adjoints[id(e)]
      Grad.add_subexpr_adjoints(e, e_adj, adjoints, eval_expr)

    var_adjoints = {}
    for lvar in var_leaves:
      visit_expr_backwards(lvar)
      var_adjoints[lvar] = var_adjoints.get(lvar, 0.) + adjoints[id(lvar)]
    # The unused vars get 0.
    res = [var_adjoints.get(v, 0.) for v in func.invars]
    if len(func.invars) == 1:
      return res[0]
    else:
      return tuple(res)

  @staticmethod
  def add_subexpr_adjoints(e: Expr,
                           out_adj: Value,
                           adjoints: Dict[int, Value],
                           eval_expr: Callable[[Expr], Value]):
    """Increment adjoints for *immediate* subexpressions.

    Args:
      e: the expression whose subexpressions' adjoints to increment.
      out_adj: the output adjoint corresponding to `e`.
      adjoints: the dictionary of adjoints (by id(expr)).
      eval_expr: an expression evaluator.
    """

    def add_adjoint(sube: Expr, adj: Value):
      adjoints[id(sube)] = adjoints.get(id(sube), 0.) + adj

    if e.operator == Operator.VAR:
      return
    elif e.operator == Operator.LITERAL:
      return
    elif e.operator == Operator.ADD:
      add_adjoint(e.args[0], out_adj)
      add_adjoint(e.args[1], out_adj)
    elif e.operator == Operator.SUB:
      add_adjoint(e.args[0], out_adj)
      add_adjoint(e.args[1], out_adj * -1.)
    elif e.operator == Operator.MUL:
      add_adjoint(e.args[0], out_adj * eval_expr(e.args[1]))
      add_adjoint(e.args[1], eval_expr(e.args[0]) * out_adj)
    elif e.operator == Operator.POW:
      pow = e.params['pow']
      if pow == 1:
        add_adjoint(e.args[0], out_adj)
      elif pow != 0:
        add_adjoint(e.args[0],
                    out_adj * float(pow) * eval_expr(e.args[0]) ** (pow - 1))
    elif e.operator == Operator.PROJECTION:
      # We need to reach into the adjoints and increment only one element
      old_adj = adjoints.get(id(e.args[0]))
      if old_adj is None:
        old_adj = [0.] * len(e.args[0].etype)
        adjoints[id(e.args[0])] = old_adj
      old_adj[e.params["idx"]] += out_adj

    elif e.operator == Operator.JIT_CALL:
      # The GRAD calculation has to go under JIT
      func = e.params['func']
      vjp_func = Grad._prepare_function_vjp(func)
      if not isinstance(out_adj, (tuple, list)):
        out_adj = [out_adj]
      args_v = map_list(eval_expr, e.args) + out_adj
      arg_adj = Expr.eval_operator_tracer(Operator.JIT_CALL,
                                          dict(func=vjp_func),
                                          args_v)
      arg_adj = (arg_adj,) if len(func.invars) == 1 else arg_adj
      for a, a_adj in zip(e.args, arg_adj):
        add_adjoint(a, a_adj)

    elif e.operator == Operator.COND_GE:
      """Given the a conditional:
      
          if pred:
            out = true_func(true_arg)
          else:
            out = false_func(false_arg)
      
      (true_func, false_func are parameters, and [pred, true_arg, false_arg]
      are arguments). We compute the input adjoints (for the 3 arguments 
      in order) as follows:
      
          if pred:
            input_adj = 0, true_func_vjp(true_arg, out_adj), 0
          else:
            input_adj = 0, 0, false_func_vjp(false_arg, out_adj)    
      """
      true_func_f = e.params["true_func"]
      true_func_vjp = Grad._prepare_function_vjp(true_func_f)
      false_func_f = e.params["false_func"]
      false_func_vjp = Grad._prepare_function_vjp(false_func_f)
      # Expand the branches to return the adjoints for all the true_args and false_args
      # We actually do not compute the adjoint for the predicate; it is 0
      zero = Expr(Operator.LITERAL, (), etype=ExprType(float), val=0.)
      true_func_vjp_expanded = Function(
        invars=true_func_vjp.invars,
        bodies=true_func_vjp.bodies + [zero] * len(false_func_f.invars))
      false_func_vjp_expanded = Function(
        invars=false_func_vjp.invars,
        bodies=[zero] * len(true_func_f.invars) + false_func_vjp.bodies)
      if not isinstance(out_adj, (tuple, list)):
        out_adj = [out_adj]
      assert len(out_adj) == len(true_func_f.bodies) == len(false_func_f.bodies)
      # We need to evaluate the arguments
      args_v = map_list(eval_expr, e.args)
      # Insert the out_adj for both branches
      args_vjp_v = (args_v[0:1 + len(true_func_f.invars)] + out_adj +
                    args_v[1 + len(true_func_f.invars):] + out_adj)
      args_adj = Expr.eval_operator_tracer(Operator.COND_GE,
                                           dict(true_func=true_func_vjp_expanded,
                                                false_func=false_func_vjp_expanded),
                                           args_vjp_v)
      assert len(args_adj) == len(e.args) - 1
      add_adjoint(e.args[0], 0.)  # The predicate arguments
      for a, a_adj in zip(e.args[1:], args_adj):
        add_adjoint(a, a_adj)
    else:
      raise NotImplementedError

  @staticmethod
  def _prepare_function_vjp(func: Function) -> Function:
    """Generates the VJP for a Function.

    Args:
      func: the original function
    Returns: a function that takes len(func.invars) + len(func.bodies) arguments,
      for the actual arguments and the output adjoints, and produces
      len(func.invars) outputs, for the input adjoints.
    """
    func_vjp_et = (
        [inv.etype for inv in func.invars] +
        [body.etype for body in func.bodies])
    return func.trace_interpreter(
      Grad.eval_vjp_func_tracer,
      args_t=func_vjp_et)


def grad(func: Callable) -> Callable[..., Function]:
  """Computes the gradient of a function.

  Args:
    func: a function of n-scalar arguments, with a single output.
  Returns: a function that when applied to `n` `func` arguments, returns the `n` partial
    derivatives of the function.
  """

  def wrapped_grad(*args: Sequence[Value]):
    func_f, func_f_env = Function.trace_callable(func,
                                                 map_list(
                                                   Tracer.val_to_type,
                                                   args))
    assert len(
      func_f.bodies) == 1, "grad is only defined for functions that return a single result"
    res_grad = Grad.eval_vjp_func_tracer(
      func_f, *tuple(itertools.chain(args, func_f_env, [1.])))
    # Drop adjoints for the freevars
    # TODO: this is an ugly result of the convention that the result is either tuple or value
    if isinstance(res_grad, tuple):
      if len(args) == 1:
        return res_grad[0]
      else:
        return res_grad[0:len(args)]
    else:
      return res_grad

  return wrapped_grad
