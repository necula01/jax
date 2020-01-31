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
Transformation: Backward differentiation
-----------------------------------------

For backward differentiation we write a transformation that increments
the adjoints (cotangent) for the subexpressions, given the adjoints for each
expression. This effectively computes vector-Jacobian (VJP) product.

To achieve the "backward differentiation" effect, we must process first
the top-level expression before processing the sub-expressions (i.e.,
top-down). Care is taken to ensure that for each unique `Expr` object only a
constant number of other `Expr` are constructed, and an `Expr` object is not
processed more than once.

For a `JIT_CALL` operation, the VJP computation is pushed under the
`jit`. For example, considering a `Function` with two inputs `x1` ans `x2`,
and an invocation `func a1 a2`, where `a1` and `a2` are the actual argument
`Value`s. We first produce another `jit_call` that computes the value and the
VJP of `func`: takes as arguments `a1`, `a2`, and also the adjoint of the outputs,
and returns the adjoints for the inputs: `x1_adj` and `x2_adj`.
, i.e., returns three values when evaluated `a1` and `a2`.

For a `COND_GE` we evaluate the VJP of the branches. Given the a conditional:

```
    if pred:
      out = true_func(arg)
    else:
      out = false_func(arg)
```
where `true_func`, `false_func` are parameters, and `[pred, arg]`
are arguments. We compute the input adjoints (for the 3 arguments in order)
as follows:
```
    if pred:
      input_adj = 0, true_func_vjp(true_arg, out_adj)
    else:
      input_adj = 0, false_func_vjp(false_arg, out_adj)
```

Note that in XLA and real JAX the `cond` primitive has separate sets of
arguments for the `true_func` and `false_func`, e.g.,
`cond_ge(pred, true_func, true_args, false_func, false_args)`. This choice
requires the `true_func_vjp` and `false_func_vjp` to return adjoints
for *both* `true_args` and `false_args` (filled with 0 for the arguments
for the opposite branch). This requires memory usage proportional to the
union of the adjoints. For the common case when we want to pass the same
inputs `x` to both branches, we'll need double the space to return its
adjoint from the two branches.

Concrete examples are in `tests/mini_jax_grad_test.py`.
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import itertools
from typing import cast, Dict, Callable, List, Optional, Sequence, Tuple

from jax.experimental.mini_jax.mini_jax import (
  Expr, ExprType, Operator, Function, Tracer,
  Value, const_like, zero_like, zero_like_shape,
  Shape
)
from jax.experimental.mini_jax.mini_jax_util import map_list


class Grad(object):
  """Methods related to backward differentiation."""

  def eval_function(self, func: Function,
                    *args_out_adj: Value) -> Sequence[Value]:
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
    assert len(args_out_adj) == nr_invars + len(func.results)
    args = args_out_adj[0:nr_invars]
    out_adj = args_out_adj[nr_invars:]

    # First we build a graph with pointers to `Expr` parents
    # For each `Expr` (by `id`), the list of parents.
    parents: Dict[int, List[Expr]] = {}
    var_leaves: List[Expr] = []  # (all the variables)

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

    adjoints: Dict[int, Value] = {}  # the accumulated adjoints for sub-expressions
    def add_adjoint(sube: Expr, adj: Value):
      adjoints[id(sube)] = adjoints.get(id(sube), zero_like(adj)) + adj

    for result, result_out_adj in zip(func.results, out_adj):
      reverse_expr_graph(result, None)
      # The result may occur more than once
      add_adjoint(result, result_out_adj)

    # Build a standard evaluator, used to evaluate lazily the values that are
    # needed from the forward pass.
    eval_std_expr = func.make_evaluator(args,
                                        eval_operator=Expr.eval_std_operator)

    visited = {}  # By sub-expression id

    def visit_expr_backwards(e: Expr):
      """Visit all expressions, to accumulate adjoints for all expressions."""
      id_e = id(e)
      if id_e in visited: return
      visited[id_e] = True
      [visit_expr_backwards(p) for p in parents[id_e]]
      # Now that all parents have been processed, our adjoint is ready
      e_adj = adjoints[id(e)]
      self.add_subexpr_adjoints(e, e_adj, adjoints, eval_std_expr)

    var_adjoints: Dict[Expr, Value] = {}  # The accumulated adjoints for variables
    for lvar in var_leaves:
      visit_expr_backwards(lvar)
      var_adjoints[lvar] = (
          var_adjoints.get(lvar, zero_like(adjoints[id(lvar)])) +
          adjoints[id(lvar)])
    # The unused vars get 0.
    res = [var_adjoints.get(v, zero_like_shape(v.etype.shape)) for v in func.invars]
    if len(func.invars) == 1:
      return res[0]  # type: ignore[return-value]
    else:
      return tuple(res)


  def add_subexpr_adjoints(self, e: Expr,
                           out_adj: Value,
                           adjoints: Dict[int, Value],
                           eval_std_expr: Callable[[Expr], Value]):
    """Increment adjoints for *immediate* subexpressions.

    Args:
      e: the expression whose subexpressions' adjoints to increment.
      out_adj: the output adjoint corresponding to `e`.
      adjoints: the dictionary of adjoints (by id(expr)).
      eval_expr: an expression evaluator.
    """
    def add_adjoint(sube: Expr, adj: Value):
      adjoints[id(sube)] = adjoints.get(id(sube), zero_like(adj)) + adj

    if e.operator == Operator.VAR:
      return
    elif e.operator == Operator.LITERAL:
      return
    elif e.operator == Operator.ADD:
      add_adjoint(e.args[0], out_adj)
      add_adjoint(e.args[1], out_adj)
    elif e.operator == Operator.SUB:
      add_adjoint(e.args[0], out_adj)
      add_adjoint(e.args[1], out_adj * const_like(-1., out_adj))
    elif e.operator == Operator.MUL:
      add_adjoint(e.args[0], out_adj * eval_std_expr(e.args[1]))
      add_adjoint(e.args[1], eval_std_expr(e.args[0]) * out_adj)
    elif e.operator == Operator.POW:
      pow = e.params['pow']
      if pow == 1:
        add_adjoint(e.args[0], out_adj)
      elif pow != 0:
        add_adjoint(e.args[0],
                    out_adj * const_like(float(pow), out_adj) * eval_std_expr(e.args[0]) ** (
                          pow - 1))
    elif e.operator == Operator.PROJECTION:
      # We need to reach into the adjoints and increment only one element
      old_adj = adjoints.get(id(e.args[0]))
      if old_adj is None:
        old_adj = [zero_like_shape(et.shape) for et in e.args[0].etype]  # type: ignore
        adjoints[id(e.args[0])] = old_adj
      old_adj[e.params["idx"]] += out_adj  # type: ignore

    elif e.operator == Operator.JIT_CALL:
      """See comments at top of file."""
      func = e.params['func']
      vjp_func = self.transform_function(func)
      if not isinstance(out_adj, (tuple, list)):
        out_adj = [out_adj]
      args_v = map_list(eval_std_expr, e.args) + out_adj  # type: ignore
      arg_adj = Expr.eval_std_operator(Operator.JIT_CALL,
                                       dict(func=vjp_func),
                                       args_v)
      arg_adj = (arg_adj,) if len(func.invars) == 1 else arg_adj
      for a, a_adj in zip(e.args, arg_adj):  # type: ignore
        add_adjoint(a, a_adj)

    elif e.operator == Operator.COND_GE:
      """See comments at top of file."""
      true_func_f = e.params["true_func"]
      true_func_vjp = self.transform_function(true_func_f)
      false_func_f = e.params["false_func"]
      false_func_vjp = self.transform_function(false_func_f)

      if isinstance(out_adj, (tuple, list)):
        out_adj_list = cast(List[Value], out_adj)
      else:
        out_adj_list = [out_adj]
      assert len(out_adj_list) == len(true_func_f.results) == len(false_func_f.results)
      # We need to evaluate the arguments, to reconstruct the cond
      args_v = map_list(eval_std_expr, e.args)
      args_vjp_v = (args_v + out_adj_list)
      args_adj = Expr.eval_std_operator(Operator.COND_GE,
                                        dict(true_func=true_func_vjp,
                                             false_func=false_func_vjp),
                                        args_vjp_v)
      args_adj_list = cast(List[Value], args_adj)
      assert len(args_adj_list) == len(e.args) - 1
      add_adjoint(e.args[0], 0.)  # The predicate argument is a scalar
      for a, a_adj in zip(e.args[1:], args_adj_list):
        add_adjoint(a, a_adj)
    else:
      raise NotImplementedError

  def transform_function(self, func: Function) -> Function:
    """Generates the VJP for a Function.

    Args:
      func: the original function
    Returns: a function that takes len(func.invars) + len(func.results) arguments,
      for the actual arguments and the output adjoints, and produces
      len(func.invars) outputs, for the input adjoints.
    """
    return func.transform_function(
      "vjp",
      self.eval_function,
      extra_args_typ=[result.etype for result in func.results])


def grad(func: Callable, abstract: bool = True, cache: bool = True) -> Callable[
  ..., Function]:
  """Computes the gradient of a function.

  Args:
    func: a function of n-scalar arguments, with a single output.
    abstract: whether to force tracers to be abstract.
    cache: whether to allow the caching of the result of tracing (only meaningful
      if `abstract`)
  Returns: a function that when applied to `n` `func` arguments, returns the `n` partial
    derivatives of the function.
  """

  def do_grad(*args: Sequence[Value]):
    func_f, func_f_env = Function.trace_user_function(func, args,
                                                      abstract=abstract,
                                                      cache=cache)
    assert len(func_f.results) == 1, (
      "grad is only defined for functions that return a single result")
    res_grad = Grad().eval_function(
      func_f, *args, *func_f_env,
      const_like(1., None, shape=func_f.results[0].etype.shape))

    if not isinstance(res_grad, tuple):
      res_grad = (res_grad,)
    # Drop adjoints for the freevars
    res_grad = res_grad[0:len(args)]
    if len(args) == 1:
      return res_grad[0]
    else:
      return res_grad

  return do_grad
