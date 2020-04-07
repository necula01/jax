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
Transformation: Vectorization
-----------------------------

Vectorization, or `vmap` is a transformation for a function that
takes arguments of rank k to a function that takes arguments or rank `k+1`.
Vectorization is defined technically as follows
(we use upper-case letter for index tuples, `,` as a tuple
and element concatenation, and `a I` for the indexing operation, i.e.,
):

>  If `e` is an expression that depends on free variable `x` of shape `S`,
>  and `b` is a positive integer,
>  then `vmap(e, b)` is an expression that depends on a free variable
>  `xv` of shape `b,S` such that, for any array `a` of shape `S`:
```
      vmap(e, b) xv (i,I) = e(xv i) I  , for i in range(b)
```

(we used `xv i` as the partial application of `xv` to the index `i`, i.e.,
`(xv i) I = xv (i, I)`.

One interesting aspect of implementing `vmap` is that we need to add new
primitive operators: `bcastdim`, `sumdim`, `where_ge`. See the discussion
in `README.md`.

One complication is that if a function has multiple outputs, and some of
the outputs depend only on unvectorized inputs, then we want to leave the
outputs unvectorized.

Concrete examples are in `tests/mini_jax_vmap_test.py`.
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from functools import partial
import itertools
import numpy as np

from typing import Any, Dict, Callable, Optional, Tuple, Sequence, List

from jax.experimental.mini_jax.mini_jax import (
  Expr, ExprType, Operator, Function, Tracer,
  Value, zero_like, CustomOperator
)
from jax.experimental.mini_jax.mini_jax_operators import (
  broadcastInDimOp, whereGeOp,
  broadcast_value, broadcast_values
)

from jax.experimental.mini_jax.mini_jax_util import map_list, unzip


class Vmap(object):
  """Methods related to vectorization."""
  def __init__(self, batch_size: int):
    self.batch_size = batch_size

  def eval_function(self,
                    func: Function,
                    *args_b: Value,
                    aux: Tuple[Sequence[bool], Optional[Sequence[bool]]]
                    ) -> Sequence[Tuple[Value, bool]]:
    """Evaluates the VMAP given maybe-batched Values for invars.

    Args:
      func: the function whose VMAP to evaluate
      args_b: the maybe-batched arguments
      aux: a pair, indicating which args_b are batched and which
        results should be batched. If the second element of the pair
        is non-None, then we force the result to be batched as
        required. If None, we just return the batched status.
    Returns:
       A tuple (one value for each result) of pairs, with the result(s) and
       the indication where the result(s) are batched.
    """
    nr_invars = len(func.invars)
    assert len(args_b) == nr_invars
    args_hasb, res_must_haveb = aux
    assert isinstance(args_hasb, Sequence) and len(args_hasb) == nr_invars
    if res_must_haveb is not None:
      assert isinstance(res_must_haveb, Sequence) and len(res_must_haveb) == len(func.results)

    # Map variables to primal and has_batch values
    eval_vmap = func.make_evaluator(list(zip(args_b, args_hasb)),
                                    eval_operator=self.eval_operator)
    res_withb = [eval_vmap(res) for res in func.results]
    # Maybe we need to batch some results
    if res_must_haveb is not None:
      batched_results = []
      for (r, r_hasb), r_must_haveb in zip(res_withb, res_must_haveb):
        if r_hasb == r_must_haveb:
          batched_results.append((r, r_must_haveb))
        elif not r_hasb and r_must_haveb:
          batched_results.append((broadcast_value(self.batch_size, r), r_must_haveb))
        else:
          assert False, "Function returns batched result but non-batched requested"
    else:
      batched_results = res_withb

    return tuple(batched_results)

  def eval_operator(self,
                    op: Operator,
                    params: Dict,
                    args_withb: Sequence[Tuple[Value, bool]]
                    ) -> Tuple[Value, bool]:
    """Evaluates the VMAP of an operator application.
    Args:
      op: the operator to evaluate
      params: the operator parameters
      args_withb: for each operator argument, a pair of maybe-batched `Value`
        and a boolean that says if batched.
    Returns:
      a pair of result `Value` and whether is batched. If the operator is a tuple_result
      operator, must return a pair of tuples: all results, and all booleans.
    """
    args_b, args_hasb = unzip(args_withb)
    if all(not a_hasb for a_hasb in args_hasb):
      # If all arguments are unbatched, the result is unbatched
      return Expr.eval_std_operator(op, params, args_b), False

    if op == Operator.VAR:
      assert False
    if op == Operator.LITERAL:
      return (params["val"], False)
    if op == Operator.ADD:
      args_b1, res_hasb = broadcast_values(self.batch_size, args_b, args_hasb)
      return (args_b1[0] + args_b1[1], res_hasb)
    if op == Operator.SUB:
      args_b1, res_hasb = broadcast_values(self.batch_size, args_b, args_hasb)
      return (args_b1[0] - args_b1[1], res_hasb)
    if op == Operator.MUL:
      args_b1, res_hasb = broadcast_values(self.batch_size, args_b, args_hasb)
      return (args_b1[0] * args_b1[1], res_hasb)
    if op == Operator.POW:
      pow = params["pow"]
      r = args_b[0] ** pow
      return (r, args_hasb[0])
    if op == Operator.PROJECTION:
      # We use PROJECTION when we have multiple-outputs. But in that
      # case the arguments are are in element 0: ((out1, out2, ...), (out1_hasb, out2_hasb, ...))
      idx = params["idx"]
      assert (len(args_withb) == 1 and len(args_withb[0]) == 2 and
              len(args_withb[0][0]) == len(args_withb[0][1]))  # type: ignore[arg-type]
      return (args_withb[0][0][idx], args_withb[0][1][idx])  # type: ignore[index]

    if op == Operator.JIT_CALL:
      # The JVP calculation has to also be under JIT
      func = params["func"]
      # We do not force the results to be batched
      vmap_func, res_hasb = func.transform_function(
        "vjp", self.eval_function,
        transformed_args_types=partial(self._transformed_args_types, args_hasb),
        aux=(args_hasb, None))
      if len(vmap_func.results) == 1:
        res_hasb = res_hasb[0]  # type: ignore[index]
      # Either build an Expr (if the args are tracers), or execute the JIT
      return Expr.eval_std_operator(Operator.JIT_CALL,  # type: ignore[return-value]
                                    dict(func=vmap_func),
                                    args_b), res_hasb

    if op == Operator.COND_GE:
      true_func_f = params["true_func"]
      args_branch_hasb = args_hasb[1:]
      false_func_f = params["false_func"]
      arg_pred_hasb = args_hasb[0]  # Whether the predicate is batched

      _safety_count = 0
      if arg_pred_hasb:
        # If the predicate is batched, we must batch all results
        true_res_must_haveb: Optional[Sequence[bool]] = (True,) * len(true_func_f.results)
      else:
        # Otherwise, we see first how they come up
        true_res_must_haveb = None
      false_res_must_haveb = true_res_must_haveb
      while True:
        true_func_vmap, true_res_hasb = true_func_f.transform_function(
          "vmap", self.eval_function,
          aux=(args_branch_hasb, true_res_must_haveb),
          transformed_args_types=partial(self._transformed_args_types, args_branch_hasb))
        false_func_vmap, false_res_hasb = false_func_f.transform_function(
          "vmap", self.eval_function,
          aux=(args_branch_hasb, false_res_must_haveb),
          transformed_args_types=partial(self._transformed_args_types, args_branch_hasb))
        if true_res_hasb == false_res_hasb:
          break
        # We must redo the branches to force them both to the same configuration of
        # batched results.
        assert _safety_count == 0
        _safety_count = 1
        true_res_must_haveb = tuple([t_hasb or f_hasb for t_hasb, f_hasb in
                                     zip(true_res_hasb, false_res_hasb)])
        false_res_must_haveb = true_res_must_haveb

      if len(true_res_hasb) == 1:
        true_res_hasb = true_res_hasb[0]

      if arg_pred_hasb:
        # Evaluate the branches
        true_res = Expr.eval_std_operator(Operator.JIT_CALL,
                                          dict(func=true_func_vmap),
                                          args_b[1:])
        false_res = Expr.eval_std_operator(Operator.JIT_CALL,
                                          dict(func=false_func_vmap),
                                          args_b[1:])
        def build_where(xt: Value, xf: Value):
          return whereGeOp.invoke_single(args_b[0], xt, xf)

        if isinstance(true_res_hasb, (tuple, list)):
          assert all(true_res_hasb)
          res = tuple(map(build_where, true_res, false_res))  # type: ignore[arg-type]
        else:
          res = build_where(true_res, false_res)

        return res, true_res_hasb
      else:
        return Expr.eval_std_operator(  # type: ignore[return-value]
          Operator.COND_GE,
          dict(true_func=true_func_vmap,
               false_func=false_func_vmap), args_b), true_res_hasb

    if isinstance(op, CustomOperator):
      res = op.eval_vmap(params, args_withb, batch_size=self.batch_size)
      assert (isinstance(res, tuple) and len(res) == 2 and isinstance(res[0], tuple)
              and isinstance(res[1], tuple) and len(res[0]) == len(res[1])), f"{op}: vmap returned {res}"
      return res  # type: ignore[return-value]

    raise NotImplementedError(f"op is {op}")

  def _vmapped_type(self, orig_typ: ExprType, hasb: bool) -> ExprType:
    return (ExprType(shape=(self.batch_size,) + orig_typ.shape, dtype=orig_typ.dtype)
            if hasb else orig_typ)

  def _transformed_args_types(self, args_hasb: Sequence[bool], args_typ: Sequence[ExprType]) -> Sequence[ExprType]:
    return tuple(map(self._vmapped_type, args_typ, args_hasb))


def vmap(func: Callable, abstract: bool = True, cache: bool = True,
         args_has_batch: Sequence[bool]=(),
        ) -> Callable[..., Function]:
  """Apply the vectorization transformation to a user-function.

  Args:
    func: a function of n arguments.
    abstract: whether to force tracer arguments to be abstract.
    cache: whether to allow the caching of the result of tracing (only meaningful
      if `abstract`)
    args_has_batch: for each argument of `func` whether the actual argument of the
      vectorized invocation is batched on the leading axis. All batch sizes must
      match. At least one argument must be batched.
  Returns: a function that when applied to `func` batched arguments,
     returns the batched results.
  """

  def do_vmap(*args_b: Value):
    args_hasb = tuple(args_has_batch)
    assert len(args_b) == len(args_hasb), f"Function with {len(args_b)} args and args_has_batch {args_has_batch}"
    batch_sizes = set([np.shape(a)[0]
                       for a, a_hasb in zip(args_b, args_hasb)
                       if a_hasb])
    assert len(batch_sizes) == 1, "All batched args have the same batch size"
    batch_size, = batch_sizes

    # Get actual arguments without batches. In order to avoid having to
    # add an indexing operator, we just make up a tensor of 0's for the
    # arguments that we unbatch.
    args_nob = tuple([a if not a_isb else zero_like(0., shape=np.shape(a)[1:])
                     for a, a_isb in zip(args_b, args_hasb)])
    # Trace the function on the unbatched arguments
    func_f, func_f_env = Function.trace_user_function(func, args_nob,
                                                      abstract=abstract,
                                                      cache=cache)
    if func_f_env:
      # The environment arguments are unbatched
      all_args_b = tuple(itertools.chain(args_b, func_f_env))
      all_args_hasb = tuple(itertools.chain(args_hasb, [False for c in func_f_env]))
    else:
      all_args_b = args_b
      all_args_hasb = args_hasb

    # Force all outputs to be batched
    res = Vmap(batch_size).eval_function(func_f, *all_args_b,
                                         aux=(all_args_hasb, (True,) * len(func_f.results)))
    res_b = tuple(r for r, _ in res)
    if len(func_f.results) == 1:
      return res_b[0]
    else:
      return res_b

  return do_vmap
