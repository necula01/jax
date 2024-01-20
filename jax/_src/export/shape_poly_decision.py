# Copyright 2022 The JAX Authors.
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
"""Shape polymorphism support for deciding inequalities of symbolic dimensions.

"""

from __future__ import annotations

from collections.abc import Sequence
import dataclasses
from enum import Enum
import itertools
import math

import numpy as np

from jax._src import core
from jax._src import config
from jax._src.export import shape_poly
from jax._src.export.shape_poly import (
  _DimExpr, _DimTerm, _DimFactor,
  _DimTerm_one,
  SymbolicScope,
  DimSize,
  InconclusiveDimensionOperation,
  Comparator,
  BoundsPrecision,
)


def sgn(x): return 1 if x >= 0 else -1

_USE_SIMPLEX = config.bool_flag(
    'jax_shape_polymorphism_use_simplex',
    config.bool_env('JAX_SHAPE_POLYMORPHISM_USE_SIMPLEX', True),
    help=(
        'Use the more powerful Simplex algorithm for deciding inequalities '
        'of symbolic dimension expressions.'
    )
)


def bounds_decision(e: DimSize,
                    prec: BoundsPrecision) -> tuple[float, float]:
  if not isinstance(e, _DimExpr):
    return (int(e), int(e))
  decision = _DecisionByElimination.build(e.scope)
  return decision.bounds(e, prec, add_implicit_constraints=True)

shape_poly._bounds_decision = bounds_decision


class _DecisionByElimination:
  """A decision procedure based on elimination of terms.

  Given an expression `e = t*t_k + rest_e` for which we want to compute bounds,
  and a constraint `c = t*t_c_k + rest_c >= 0`,

  Let `e0 = e*abs(t_c_k) - c*sgn(t_c_k)*t_k`. (Note that we eliminated `t` from
  `e0`, since `abs(t_c_k)*t_k = sgn(t_c_k)*t_k*t_c_k`.)

  Since `c >= 0`,
  if `sgn(t_c_k)*t_k > 0`:
     then `abs(t_c_k)*e >= e0`, hence, `LB(e) >= ceil(LB(e0) / abs(t_c_k))`,

  if `sgn(t_c_k)*t_k < 0`
    then `abs(t_c_k)*e <= e0`, hence, `UB(e) <= floor(UB(e0) / abs(t_c_k))`,

  See the implementation in self.combine_term_with_existing.
  Do not use the constructor directly, use the `build` static method.
  """
  def __init__(self, scope: SymbolicScope):
    self.scope = scope
    self._processed_for_internal_constraints: set[_DimTerm] = set()
    # The other fields are for keeping an efficient representation of
    # the explicit constraints.
    self._term_bounds: dict[_DimTerm, tuple[float, float]] = {}
    # The _expr_constraints represents a set of constraints that are not
    # just simple terms. The set is represented as a mapping from a
    # term "t" to tuples (cmp, k, c) where "c >= 0" (if cmp is GEQ else "c == 0")
    # represents a constraint that has "t" as the leading term with coefficient "k".
    self._expr_constraints: dict[_DimTerm, set[tuple[Comparator, int, _DimExpr]]] = {}

  def initialize(self) -> _DecisionByElimination:
    # TODO: separate Simplex out, if we decide we keep it
    self.simplex = _Simplex(self.scope) if _USE_SIMPLEX.value else None

    # Process the explicit constraints in the order in which the user specifies
    # them. This is because the heuristics depend on the order in which the
    # constraints are processed, and this way we give the user a way to control
    # the result (albeit, for now, without a good feedback loop to understand
    # how the order matters for inequalities).
    for constr in self.scope._explicit_constraints:
      if not core.is_constant_dim(constr.diff):
        self.add_implicit_constraints_expr(constr.diff)  # type: ignore

      self.combine_and_add_constraint(constr.cmp, constr.diff, 0,
                                      constr.debug_str)

      # Clear the cache, since we have added constraints.
      self.scope._bounds_cache.clear()

    return self

  @staticmethod
  def build(scope: SymbolicScope) -> _DecisionByElimination:
    """Builds an initialized DecisionByElimination for a scope.

    Caches the initial state of the decision procedure in the scope.
    """
    if not scope._initialized or not scope._explicit_constraints:
      # We do not cache until the scope is fully initialized.
      return _DecisionByElimination(scope).initialize()

    if not scope._decision_initial_state:
      scope._decision_initial_state = _DecisionByElimination(scope).initialize()
    d = scope._decision_initial_state
    # Return a copy, because the decision procedure state is mutable
    c = _DecisionByElimination(scope)
    c._processed_for_internal_constraints = d._processed_for_internal_constraints.copy()
    if _USE_SIMPLEX.value:
      c.simplex = d.simplex.copy()
    else:
      c._term_bounds = d._term_bounds.copy()
      c._expr_constraints = {
          lead_t: lead_t_constraints.copy()
          for lead_t, lead_t_constraints in d._expr_constraints.items()}
    return c

  def combine_and_add_constraint(self,
                                 cmp: Comparator,
                                 e1: _DimExpr | int | float,
                                 e2: _DimExpr | int | float,
                                 debug_str: str | None = None):
    """Adds a constraint "e1 >= e2" to the internal state."""
    if isinstance(e1, float):
      if np.isinf(e1) and e1 >= 0 and cmp == Comparator.GEQ: return
      assert e1 == np.floor(e1)
      e1 = int(e1)
    if isinstance(e2, float):
      if np.isinf(e2) and e2 <= 0 and cmp == Comparator.GEQ: return
      assert e2 == np.floor(e2)
      e2 = int(e2)
    e = e1 - e2
    if (const := _DimExpr._to_constant(e)) is not None:
      if const < 0:
        raise ValueError(f"Unsatisfiable constraint: {debug_str or str(e1) + ' >= ' + str(e2)}")
      return
    assert isinstance(e, _DimExpr)
    self.add_to_state(cmp, e, debug_str)
    if not _USE_SIMPLEX.value:
      geq_combinations = self.combine_constraint_with_existing(cmp, e, debug_str)
      for cmp, a in geq_combinations:
        self.add_to_state(cmp, a, None)

  def add_to_state(self,
                   cmp: Comparator,
                   e: _DimExpr,
                   debug_str: str | None):
    """Updates the internal state to reflect "e >= 0". """
    assert _DimExpr._to_constant(e) is None

    if _USE_SIMPLEX.value:
      self.simplex.assume_geq0(e,
                               info=_Simplex.AssumeInfo(
                                   type=_Simplex.AssumeType.IMPLICIT,
                                   msg=debug_str))
      if cmp == Comparator.EQ:
        # TODO: support EQ directly in Simplex
        self.simplex.assume_geq0(- e,
                                 info=_Simplex.AssumeInfo(
                                     type=_Simplex.AssumeType.IMPLICIT,
                                     msg=debug_str))
      return

    if (term_factors := e._to_single_term()) is not None:
      n, t_k, t = term_factors  # n + t * t_k [== | >=] 0
      lb, ub = self._term_bounds.get(t, (- np.inf, np.inf))
      if cmp == Comparator.EQ:
        # n + t_k * t == 0  ->  t == - n // t_k
        if n % t_k:
          raise ValueError(f"Unsatisfiable constraint: {debug_str}")
        t_val = - (n // t_k)
        lb = max(lb, t_val)
        ub = min(ub, t_val)
      else:  # GEQ
        if t_k > 0:
          lb = max(lb, int(np.ceil(- n / t_k)))
        else:
          ub = min(ub, int(np.floor(- n / t_k)))
      if lb > ub:
        raise ValueError(f"Unsatisfiable constraint: {debug_str}")

      self._term_bounds[t] = (lb, ub)
      return

    lead_t, lead_t_k = e._leading_term
    lead_t_constraints = self._expr_constraints.get(lead_t)
    if lead_t_constraints is None:
      lead_t_constraints = set()
      self._expr_constraints[lead_t] = lead_t_constraints
    lead_t_constraints.add((cmp, lead_t_k, e))

  def combine_term_with_existing(self, t: _DimTerm, t_k: int, *,
                                 scope: shape_poly.SymbolicScope,
                                 only_smaller_than_t=True,
                                 ) -> Sequence[tuple[Comparator,
                                                     _DimExpr,
                                                     int,
                                                     int]]:
    """
    Combine a term with existing constraints.
    For input (t, t_k) the tuple (c_eq, c, c_s, t_s) is among the returned
    tuples if there exists a constraint `c =[c_eq] 0` that can be combined
    with `t*t_k` to eliminate `t`, and:

      * `c =[c_eq] 0`
      * The term `comb = t*t_k*t_s + c*c_s` does not contain `t`, and if
        `only_smaller_than_t` then `comb` contains only terms structurally
         smaller than `t`.
      * `c_s > 0`
    """
    # TODO: maybe a generator is useful here instead of materializing the list
    acc: list[tuple[Comparator, _DimExpr, int, int]] = []
    # First combine with the existing term bounds
    t_lb, t_ub = self._term_bounds.get(t, (-np.inf, np.inf))
    if t_lb == t_ub:
      acc.append((Comparator.EQ, _DimExpr(((t, 1),), scope) - int(t_lb),
                  abs(t_k), - sgn(t_k)))
    else:
      if t_lb > -np.inf:
        acc.append((Comparator.GEQ, _DimExpr(((t, 1),), scope) - int(t_lb),
                    abs(t_k), - sgn(t_k)))
      if t_ub < np.inf:
        acc.append((Comparator.GEQ, _DimExpr(((t, -1),), scope) + int(t_ub),
                    abs(t_k), sgn(t_k)))

    prev_constraint: set[tuple[Comparator, int, _DimExpr]]
    for prev_constraint in ([self._expr_constraints.get(t, set())] if only_smaller_than_t
                            else self._expr_constraints.values()):
      for c_eq, _, c in prev_constraint:
        # TODO: optimize this dict()
        tc_k = dict(c._sorted_terms).get(t)
        if tc_k is not None:
          # c =[c_eq] 0 AND t*tc_k appears in c.
          c_s = abs(t_k)
          c_t = - tc_k * sgn(t_k)
          acc.append((c_eq, c, c_s, c_t))
    return acc

  def combine_constraint_with_existing(self,
                                       eq: Comparator,
                                       e: _DimExpr,
                                       debug_str: str | None) -> set[tuple[Comparator, _DimExpr]]:
    combinations: set[tuple[Comparator, _DimExpr]] = set()
    for t, t_k in e._sorted_terms:
      if t.is_constant: continue
      for (c_eq, c, c_s, t_s) in self.combine_term_with_existing(t, t_k,
                                                                 only_smaller_than_t=False,
                                                                 scope=e.scope):
        # c =[c_eq] 0 AND c_s > 0 AND t*t_k*t_s + c*c_s does not contain t
        if t_s > 0 or eq == Comparator.EQ:
          new_eq = Comparator.EQ if (eq == c_eq == Comparator.EQ) else Comparator.GEQ
          new_e = _DimExpr._linear_combination(e, t_s, c, c_s, e.scope)
          if (const := _DimExpr._to_constant(new_e)) is not None:
            if ((new_eq == Comparator.GEQ and const < 0) or
                (new_eq == Comparator.EQ and const != 0)):
              raise ValueError(f"Unsatisfiable constraints: {debug_str or str(e) + ' >= 0'}")
          else:
            combinations.add((new_eq, new_e))  # type: ignore
    return combinations

  def bounds(self, e: DimSize,
             prec: BoundsPrecision,
             add_implicit_constraints: bool = False
             ) -> tuple[float, float]:
    """Returns the lower and upper bounds, or -+inf.

    Args:
      e: the expression for which to compute the bounds.
      prec: the desired precision. See comments in `BoundsPrecision`.
      add_implicit_constraints: if True, then before computing the bounds
        add the implicit constraints for the terms inside `e`.
    """
    if (const := _DimExpr._to_constant(e)) is not None:
      return (const, const)
    assert isinstance(e, _DimExpr)
    # Caching bounds is tricky. Since the underlying _bounds_for_sorted_terms
    # is incomplete, and it may produce better results in the context of
    # specific queries (due to the implicit constraints), if we cache the
    # bounds computation we may stick to sub-optimal results. Also, we should
    # not use the precision as part of the cache key, because a certain result
    # may work for multiple precisions.
    if (res := self.scope._bounds_cache.get(e)) is not None:
      lb, ub, prev_prec = res
      if prec._bounds_are_sufficient(lb, ub): return (lb, ub)
      if prev_prec.value >= prec.value: return (lb, ub)

    if add_implicit_constraints:
      self.add_implicit_constraints_expr(e)

    if _USE_SIMPLEX.value:
      lb, ub = self.simplex.bounds(e, prec)
    else:
      lb, ub = self._bounds_for_sorted_terms(e.scope, e._sorted_terms, 0, prec)
    lb, ub = (int(lb) if lb > -np.inf else lb,
              int(ub) if ub < np.inf else ub)
    self.scope._bounds_cache[e] = (lb, ub, prec)
    return (lb, ub)

  def _bounds_for_sorted_terms(self,
                               scope: SymbolicScope,
                               e: Sequence[tuple[_DimTerm, int]],
                               i: int,
                               prec: BoundsPrecision) -> tuple[float, float]:
    """The lower and upper bounds of e[i:].

    See comments about soundness and `cmp_with` in the `shape_poly.bounds_decision`` method.
    Returns (lower-bound, upper-bound)
    """
    if i >= len(e): return (0, 0)

    t, t_k = e[i]
    if t.is_constant:
      assert i == len(e) - 1  # Must be last
      return (t_k, t_k)

    lb = -np.inf
    ub = np.inf

    for (c_eq, c, c_s, t_s) in self.combine_term_with_existing(t, t_k,
                                                               only_smaller_than_t=True,
                                                               scope=scope):
      # `c =[eq] 0` AND `t*t_k*t_s + c*c_s` contains only terms smaller than t
      # AND c_s > 0.
      # `rest = e[i:]*t_s + c*c_s` AND `rest_ub >= rest >= rest_lb`
      # `rest` contains only terms smaller than `t`.
      rest = _DimExpr._linear_combination_sorted_pairs(e, i, t_s,
                                                       c._sorted_terms, 0, c_s)
      rest_lb, rest_ub = self._bounds_for_sorted_terms(scope, rest, 0,
                                                       BoundsPrecision.BEST)
      if rest_ub < np.inf:
        # We have: e[i:]*t_s = rest - c*c_s <= rest_ub
        if t_s > 0:
          ub = min(ub, int(np.floor(rest_ub / t_s)))
        else:
          lb = max(lb, int(np.ceil(rest_ub / t_s)))

      if rest_lb > - np.inf and c_eq == Comparator.EQ:
        # We have: e[i:]*t_s = rest - c*c_s = rest >= rest_lb
        if t_s > 0:
          lb = max(lb, int(np.ceil(rest_lb / t_s)))
        else:
          ub = min(ub, int(np.floor(rest_lb / t_s)))

      if prec._bounds_are_sufficient(lb, ub): return (lb, ub)

    # Now look for special rules for factors
    if (t_f := t.to_factor()) is not None:
      if t_f.operation in [_DimFactor.MAX, _DimFactor.MIN]:
        # m_c*MAX(op1, op2) + rest_e >= max(m_c * op1 + rest_e, m_c * op2 + rest_e)
        #   if m_c > 0. Similar rules for when m_c < 0 and for MIN.
        op1, op2 = t_f.operands
        rest1 = _DimExpr._linear_combination_sorted_pairs(e, i + 1, 1,
                                                          op1._sorted_terms, 0, t_k)
        rest2 = _DimExpr._linear_combination_sorted_pairs(e, i + 1, 1,
                                                          op2._sorted_terms, 0, t_k)
        rest1_lb, rest1_ub = self._bounds_for_sorted_terms(scope, rest1, 0,
                                                           BoundsPrecision.BEST)
        rest2_lb, rest2_ub = self._bounds_for_sorted_terms(scope, rest2, 0,
                                                           BoundsPrecision.BEST)
        like_max = (t_k > 0 if t_f.operation == _DimFactor.MAX else t_k < 0)
        if like_max:
          lb = max(lb, max(rest1_lb, rest2_lb))
          ub = min(ub, max(rest1_ub, rest2_ub))
        else:
          lb = max(lb, min(rest1_lb, rest2_lb))
          ub = min(ub, min(rest1_ub, rest2_ub))
        if prec._bounds_are_sufficient(lb, ub, ): return (lb, ub)

    return lb, ub

  def add_implicit_constraints_expr(self, e: _DimExpr):
    """Adds the implicit constraints for the expression `e`"""
    for t, _ in e._sorted_terms:
      if t.is_constant: continue
      self.add_implicit_constraints_term(t)

  def add_implicit_constraints_term(self, t: _DimTerm):
    if t in self._processed_for_internal_constraints: return
    self._processed_for_internal_constraints.add(t)
    t_e = _DimExpr._from_term(t, 1, self.scope)  # m as a _DimExpr
    f = t.to_factor()
    if f is None:
      # This is a multiplication of factors. Try to compute bounds based on
      # the bounds of the factors.
      bounds = []
      for f1, f1_exp in t._factors:
        f1_t = _DimTerm.from_factor(f1, 1)
        f1_e = _DimExpr._from_term(f1_t, 1, self.scope)
        self.add_implicit_constraints_term(f1_t)
        a1_l, a1_u = self.bounds(f1_e, BoundsPrecision.BEST)
        assert a1_l <= a1_u
        bounds.append((a1_l ** f1_exp, a1_u ** f1_exp))

      candidate_bounds = [math.prod(factor_bounds)
                          for factor_bounds in itertools.product(*bounds)]
      m_l = min(*candidate_bounds)
      m_u = max(*candidate_bounds)
      self.combine_and_add_constraint(Comparator.GEQ, t_e, m_l)
      self.combine_and_add_constraint(Comparator.GEQ, m_u, t_e)
      return

    # It is a factor, is it a variable?
    if (v := f.to_var()) is not None:
      self.combine_and_add_constraint(Comparator.GEQ, t_e, 1)  # v >= 1
      return

    for oper in f.operands:
      self.add_implicit_constraints_expr(oper)

    if f.operation == _DimFactor.MOD:
      op1, op2 = f.operands
      op2_b_l, op2_b_u = self.bounds(op2, BoundsPrecision.FOR_GEQ0_OR_LT0)
      if op2_b_l > 0:  # positive divisor
        self.combine_and_add_constraint(Comparator.GEQ, t_e, 0)  # m >= 0
        self.combine_and_add_constraint(Comparator.GEQ, op2 - 1, t_e)  # m <= op2 - 1
        self.combine_and_add_constraint(Comparator.GEQ, op2_b_u - 1, t_e)
      elif op2_b_u < 0:  # negative divisor
        self.combine_and_add_constraint(Comparator.GEQ, t_e, op2 + 1)  # m >= op2 + 1
        self.combine_and_add_constraint(Comparator.GEQ, t_e, op2_b_l + 1)
        self.combine_and_add_constraint(Comparator.GEQ, 0, t_e)  # m <= 0
      return

    if f.operation == _DimFactor.FLOORDIV:
      op1, op2 = f.operands
      (op1_l, op1_u) = self.bounds(op1, BoundsPrecision.BEST)
      (op2_l, op2_u) = self.bounds(op2, BoundsPrecision.BEST)

      def math_floor_with_inf(a: float, b: float):
        # math.floor(a / b), but aware of inf.
        # When either a or b are infinite, the result represents the limit
        # of "a // b".
        assert b != 0  # we caught division by 0 earlier
        if not np.isinf(b):  # divisor b is finite
          if not np.isinf(a):  # both dividend a and divisor b are finite
            return math.floor(a / b)
          # a is infinite, b is finite
          return -np.inf if (a >= 0) != (b >= 0) else np.inf
        elif not np.isinf(a):  # dividend a is finite and divisor b is infinite
          return -1 if (a >= 0) != (b >= 0) else 0
        else:  # both dividend and divisor are infinite
          return -np.inf if (a >= 0) != (b >= 0) else np.inf

      # Same reasoning as for multiplication: the bounds are among the cross-product
      # of the bounds.
      if op2_l <= 0 <= op2_u:
        raise InconclusiveDimensionOperation(
            f"Possible division by 0 in division by {op2}")
      candidate_bounds = [math_floor_with_inf(op1_l, op2_l),
                          math_floor_with_inf(op1_l, op2_u),
                          math_floor_with_inf(op1_u, op2_l),
                          math_floor_with_inf(op1_u, op2_u)]
      m_l = min(*candidate_bounds)
      m_u = max(*candidate_bounds)
      self.combine_and_add_constraint(Comparator.GEQ, t_e, m_l)
      self.combine_and_add_constraint(Comparator.GEQ, m_u, t_e)
      if op2_l >= 0:
        if op1_l >= 0:
          self.combine_and_add_constraint(Comparator.GEQ, t_e, 0)
        mod_e = _DimExpr._from_operation(_DimFactor.MOD, op1, op2,
                                         scope=self.scope)
        if isinstance(mod_e, _DimExpr):
          self.add_implicit_constraints_expr(mod_e)
        combined = op2 * t_e + mod_e
        self.combine_and_add_constraint(Comparator.EQ, op1, combined)
      return

    if f.operation == _DimFactor.MAX:
      op1, op2 = f.operands
      op1_b_l, op1_b_u = self.bounds(op1, BoundsPrecision.BEST)
      op2_b_l, op2_b_u = self.bounds(op2, BoundsPrecision.BEST)
      self.combine_and_add_constraint(Comparator.GEQ, t_e, max(op1_b_l, op2_b_l))
      self.combine_and_add_constraint(Comparator.GEQ, max(op1_b_u, op2_b_u), t_e)
      self.combine_and_add_constraint(Comparator.GEQ, t_e, op1)
      self.combine_and_add_constraint(Comparator.GEQ, t_e, op2)
      return

    if f.operation == _DimFactor.MIN:
      op1, op2 = f.operands
      op1_b_l, op1_b_u = self.bounds(op1, BoundsPrecision.BEST)
      op2_b_l, op2_b_u = self.bounds(op2, BoundsPrecision.BEST)
      self.combine_and_add_constraint(Comparator.GEQ, t_e, min(op1_b_l, op2_b_l))
      self.combine_and_add_constraint(Comparator.GEQ, min(op1_b_u, op2_b_u), t_e)
      self.combine_and_add_constraint(Comparator.GEQ, op1, t_e)
      self.combine_and_add_constraint(Comparator.GEQ, op2, t_e)
      return


## Simplex decision procedure

# Many things still to do:
#  - better trimming and presentation of the satisfying assignment
#  - better collection of disjunctions, e.g., too few when under floordiv and
#    too many when collecting from unrelevant assumptions
#  - too slow, way too slow, e.g., try the indexing shape_poly_test.
#    Maybe we can do the case analysis after we try
#    to prove with just implicit assumptions?
#  - when we try to add an assumption "a >= 0" and we already have a varible for
#    "a", then we re-use that variable, but we do not give it an AssumptionInfo.
#
class _Simplex:

  TOLERANCE = 1e-2

  class AssumeType(Enum):
    GOAL = 1,  # The expression being optimized
    IMPLICIT = 2,  # e.g., b >= 1, div * divisor + mod = dividend
    EXPLICIT = 3,  # Passed in

  @dataclasses.dataclass
  class AssumeInfo:
    type: _Simplex.AssumeType
    msg: str = ""

  @dataclasses.dataclass
  class Var:
    info: _Simplex.AssumeInfo | None
    expr: _DimExpr
    short_name: str  # Either a dim var, or an auxiliary var
    row: int | None = None  # TODO: separate out, it is mutable
    col: int | None = None  # TODO: separate out, it is mutable
    restricted: bool = False  # TODO: separate out, it is mutable

    def __str__(self):
      v_name = self.short_name
      if self.restricted:
        v_name += "+"
      return v_name

    @property
    def pp_with_loc(self):
      v_name = str(self)
      if self.row is not None:
        v_name += f"(r{self.row})"
      else:
        v_name += f"(c{self.col})"
      return v_name

    def __repr__(self):
      return self.pp_with_loc

    def copy(self) -> _Simplex.Var:
      return _Simplex.Var(info=self.info, expr=self.expr,
                          short_name=self.short_name, row=self.row, col=self.col,
                          restricted=self.restricted)


  class Unsatisfiable(Exception):
    pass

  def __init__(self, scope: SymbolicScope):
    self.scope = scope
    self.var_counter = itertools.count()
    self.nr_rows = self.nr_cols = 0
    self.tableau: np.ndarray = np.zeros((16, 16), dtype=np.float32)  # [nr_rows, nr_cols]
    self.row_owner: list[_Simplex.Var] = []
    self.col_owner: list[_Simplex.Var] = []
    self.expr_to_var: dict[_DimExpr, _Simplex.Var] = {}

    # A number that when multiplied with the tableau elements makes them all
    # be whole numbers. Used for invariant checking, and for error messages.
    self.denominator: int = 1

    # Whether to collect coverage
    self.coverage: set[str] | None = None

    one_v = self.add_col(_DimTerm_one)
    assert one_v.col == 0

  def copy(self):
    c = _Simplex(self.scope)
    c.var_counter = self.var_counter  # TODO: fix
    c.nr_rows = self.nr_rows
    c.nr_cols = self.nr_cols
    c.tableau = np.copy(self.tableau)
    c.denominator = self.denominator
    c.row_owner = [v.copy() for v in self.row_owner]
    c.col_owner = [v.copy() for v in self.col_owner]
    c.expr_to_var = {}
    for v in itertools.chain(c.row_owner, c.col_owner):
      c.expr_to_var[v.expr] = v
    return c

  def check_invariant_if_enabled(self):
    if config.enable_checks.value:
      # This is the default in tests, but we want to turn it off with
      # JAX_ENABLE_CHECKS
      if not config.bool_env("JAX_ENABLE_CHECKS", True): return
    assert self.nr_rows == len(self.row_owner)
    assert self.nr_cols == len(self.col_owner), (self.nr_cols, self.col_owner)
    assert self.col_owner[0].short_name == "1", self.col_owner[0]
    assert not self.col_owner[0].restricted

    restricted_owners: set[_DimExpr] = set()
    for c, co in enumerate(self.col_owner):
      assert co.col == c
      assert co.row is None
      if c == 0:
        assert not co.restricted
        assert co.short_name == "1"
      if co.restricted:
        restricted_owners.add(co.expr)

    scaled_tableau = self.tableau * self.denominator
    if config.enable_checks.value:
      assert self.within_tolerance_of_some_integer(scaled_tableau)
    scaled_tableau = np.round(scaled_tableau)

    for r, ro in enumerate(self.row_owner):
      assert ro.row == r
      assert ro.col is None
      if ro.restricted:
        assert self.tableau[r, 0] >= 0.
        restricted_owners.add(ro.expr)
      ro_expr_calculated = sum([int(scaled_tableau[r, co.col]) * co.expr
                              for co in self.col_owner])
      assert self.denominator * ro.expr == ro_expr_calculated, (
        dict(r=r, scaled_ro_expr=self.denominator * ro.expr,
             ro_expr_calculated=ro_expr_calculated,
             tableau_pp=self.state_pp,
             denominator=self.denominator))

  @property
  def state_pp(self):
    rows: list[str] = []
    header_row = " | ".join([" " * 8] +
                            [f"{c:2d}:{str(co):5s}" for c, co in enumerate(self.col_owner)])
    rows.append(header_row)
    separator_row = "-|-".join(["-" * 8] * (1 + self.nr_cols))
    rows.append(separator_row)
    def pp_entry(v: float) -> str:
      if self.within_tolerance_of_some_integer(v) and self.round_to_tolerance(v) == 0:
        return " " * 8
      else:
        return f"{v:8.3f}"
    for r, ro in enumerate(self.row_owner):
      row = " | ".join([f"{r:2d}:{str(ro):5s}"] +
                       [pp_entry(v) for v in self.tableau[r, :self.nr_cols]])
      rows.append(row)
    rows.append(separator_row)
    rows.append(f"\nDenominator = {self.denominator}")
    rows.append("Legend:")
    # Collect the mapping of short_names to expr
    for owner in self.row_owner + self.col_owner:
      if owner.short_name == str(owner.expr): continue
      msg = ""
      if owner.info is not None:
        if owner.info.type == _Simplex.AssumeType.GOAL:
          msg = "goal"
        elif owner.info.type == _Simplex.AssumeType.IMPLICIT:
          msg = owner.info.msg
      if msg:
        msg = "  # " + msg
      rows.append(f"{str(owner):8s} = {owner.expr}{msg}")

    # Explain the sample solution
    rows.append("\nSample solution:")
    sol = self.sample_solution_assignment()
    for e, v in sorted(sol.items()):
      rows.append(f"  {e} = {v}")
    return "\n".join(rows)

  def __repr__(self):  # To show nicely in the debugger
    return self.state_pp
  __str__ = __repr__

  def collect_coverage(self, what: str):
    # For testing
    if self.coverage is None: return
    self.coverage.add(what)

  def new_var(self, expr: _DimExpr,
              info: _Simplex.AssumeInfo | None,
              row=None, col=None,
              short_name: str | None = None) -> _Simplex.Var:
    if short_name is None:
      if info is not None and info.type == _Simplex.AssumeType.GOAL:
        short_name = f"_g{next(self.var_counter)}"
      elif expr._is_constant:
        short_name = str(expr)
      elif (expr_mon_tuple := expr._to_single_term()) is not None:
        k, c, mon = expr_mon_tuple
        if (v := mon.to_var()) is not None:
          if k == 0 and c == 1:
            short_name = v
          elif k == 1 and c == 1:
            short_name = f"_s{v}"  # A slack for a variable
        if short_name is None:
          short_name = f"_e{next(self.var_counter)}"
      else:
        short_name = f"_e{next(self.var_counter)}"
    return _Simplex.Var(info=info, expr=expr,
                        short_name=short_name,
                        row=row, col=col)

  def assume_geq0(self, e: _DimExpr, *,
                  info: _Simplex.AssumeInfo,
                  short_name: str | None = None):
    if (const := _DimExpr._to_constant(e)) is not None:
      if const < 0:
        # TODO: is this needed?
        raise _Simplex.Unsatisfiable
      return

    ev = self.add_expr(e, info=info, short_name=short_name)
    self.check_invariant_if_enabled()
    # Prepare to restrict ev
    if ev.restricted: return
    if ev.row is not None and self.tableau[ev.row, 0] < 0:
      v_lb = self.improve_var(ev, 1, for_restriction=True)
      if v_lb < 0.:
        raise ValueError(f"Unsatisfiable constraint: {ev.info.msg}")
    ev.restricted = True
    self.check_invariant_if_enabled()

  def bounds(self, e: _DimExpr,
             prec: BoundsPrecision) -> tuple[float, float]:
    if not isinstance(e, _DimExpr) or e._is_constant:
      return (int(e), int(e))
    ev = self.add_expr(e, _Simplex.AssumeInfo(_Simplex.AssumeType.GOAL))
    lb = self.improve_var(ev, -1)
    ub = np.inf
    if prec._bounds_are_sufficient(lb, ub): return (lb, ub)
    ub = self.improve_var(ev, 1)
    if prec._bounds_are_sufficient(lb, ub): return (lb, ub)

    # Special (and expensive) handling of conditionals.
    for m, m_c in e._sorted_terms:
      if m.is_constant: continue
      if (m_a := m.to_factor()) is not None and m_a.operation in [_DimFactor.MAX, _DimFactor.MIN]:
        like_max = (m_c > 0 if m_a.operation == _DimFactor.MAX else m_c < 0)
        # TODO: use linear combination
        e_without_m = e - _DimExpr(((m, m_c),), scope=e.scope)
        op1, op2 = m_a.operands
        rest_1 = e_without_m + m_c * op1
        rest_2 = e_without_m + m_c * op2
        rest1_lb, rest1_ub = self.bounds(rest_1, BoundsPrecision.BEST)
        rest2_lb, rest2_ub = self.bounds(rest_2, BoundsPrecision.BEST)
        if like_max:
          lb = max(lb, max(rest1_lb, rest2_lb))
          ub = min(ub, max(rest1_ub, rest2_ub))
        else:
          lb = max(lb, min(rest1_lb, rest2_lb))
          ub = min(ub, min(rest1_ub, rest2_ub))
        if prec._bounds_are_sufficient(lb, ub): return (lb, ub)

    return (lb, ub)


  def add_monomials(self, e: _DimExpr) -> None:
    # Add all the inner monomials and the constraints associated with them
    for t, t_k in e._sorted_terms:
      if t.is_constant: continue

      # TODO: maybe add a term_to_var cache?
      t_e = _DimExpr(((t, 1),), e.scope)
      if self.expr_to_var.get(t_e) is not None: continue
      # Not in the tableau, add as a column
      mv = self.add_col(t)
      self.expr_to_var[t_e] = mv

    return

  def add_expr(self, e: _DimExpr,
               info: _Simplex.AssumeInfo,
               short_name: str | None = None) -> _Simplex.Var:
    if (added := self.expr_to_var.get(e)) is not None: return added
    self.add_monomials(e)
    if self.nr_rows >= self.tableau.shape[0]:
      self.expand_tableau(rows=self.nr_rows + 16)
    # Temporarily, fill in a new row for the expression
    row = self.nr_rows

    for mon, mon_count in e._sorted_terms:
      if mon.is_constant:
        self.tableau[row, 0] += mon_count
        continue

      mon_e = _DimExpr(((mon, 1),), e.scope)
      mv = self.expr_to_var.get(mon_e)
      assert mv is not None
      if mv.col is not None:
        self.tableau[row, mv.col] += mon_count
      else:
        self.tableau[row] += mon_count * self.tableau[mv.row]

    # Is the row trivially equal to a column?
    other_col = np.argmax(self.tableau[row])
    if self.tableau[row, other_col] == 1.:
      zeros_in_row = (self.tableau[row] == 0.)
      zeros_in_row[other_col] = True
      if zeros_in_row.all():
        self.tableau[row] = 0.
        self.expr_to_var[e] = self.col_owner[other_col]
        return self.col_owner[other_col]

    # If the row identical to another row?
    is_equal_to_this_row = (self.tableau == self.tableau[row, :]).all(axis=1)
    is_equal_to_this_row[row] = False
    other_row = np.argmax(is_equal_to_this_row)
    if is_equal_to_this_row[other_row]:
      self.tableau[row] = 0.
      self.expr_to_var[e] = self.row_owner[other_row]
      return self.row_owner[other_row]

    # We actually add a new row
    v = self.new_var(expr=e, row=row, info=info, short_name=short_name)
    self.row_owner.append(v)
    self.nr_rows += 1
    self.expr_to_var[e] = v
    return v

  def add_col(self, mon: _DimTerm):
    if self.nr_cols >= self.tableau.shape[1]:
      self.expand_tableau(cols=self.tableau.shape[1] + 16)
    col = self.nr_cols
    self.nr_cols += 1
    v = self.new_var(expr=_DimExpr(((mon, 1),), self.scope), col=col, info=None)
    self.col_owner.append(v)
    return v

  def find_best_pivot(
      self, *,
      goal_r: int | None,
      only_in_column: int | None,
      direction: int,
      ) -> tuple[int, int, float]:
    """Finds the best row and column for the next pivot.

    See Nelson, G. https://people.eecs.berkeley.edu/~necula/Papers/nelson-thesis.pdf,
      Algorithm F2-4, page 54.

    We use the notation `t_ij` to denote `tableau[i, j]`.
    If we pivot on (r, c), then
       * the sample value of `c` will be `- t_r0/t_rc`. Hence, if `c` is restricted
         we must have `- t_r0/t_rc >= 0`.
       * the sample value of a row `p !=  r` will be `t_p0 - t_pc * t_r0 / t_rc`.
         Hence, `t_r0 / t_rc <= min(abs(t_p0 / t_pc), for all restricted p)`.

    If we need to increase the sample value of `g * direction` then we need
    only consider the columns with `t_gc != 0`.  Also, if `c` is restricted
    we are only allowed to increase its sample value, hence we should only
    consider it if `t_gc * direction > 0`.

    See Nelson, G. https://people.eecs.berkeley.edu/~necula/Papers/nelson-thesis.pdf,
      Algorithm F1, page 54.

    Args:
      goal_r: (optional) seek to increase the sample value of `g * direction`.
      only_in_column: (optional) seek for a pivot only in this column, to
        increase the sample value of `in_column * direction`. If given
        the `goal_r` is ignored.
      direction: +1 if we want to increase the sample value of `goal_r` and
        -1 if we want to decrease it. Hence, we always want to increas
        `g * direction`.

    Return:
      a tuple of (`r`, `c`, `best_increase`). Among all the pivots that
      preserve the validity of the tableau, (`r`, `c`) provides the largest
      increase for the sample value of `c * direction * t_gc`, and this increase is
      `best_increase` (always >= 0). If `best_increase` is 0, the no valid pivot
      can improve the sample value of `c * direction * t_gc`. In that case `r` and `c`
      are `-1`.
      If `best_increase` is infinity then the column `g` can be changed
      arbitrarily in the direction of `c_factor` without violating the tableau
      constraints. In that case `r` and `c` are some example pivots for
      increasing the sample value of `c * direction * t_gc`. In this case `r` may
      be `None`. TODO
    """
    assert (goal_r is not None) != (only_in_column is not None), (goal_r, only_in_column)
    p_row = p_col = -1  # Store here the best pivot found up to this column.
                        # -1 means that the goal is manifestly optimized.
    p_increase = 0.
    for c, co in enumerate([self.col_owner[only_in_column]] if only_in_column is not None
                           else self.col_owner):
      if only_in_column is not None:
        c = only_in_column
        c_factor = direction
      else:
        if c == 0: continue
        c_factor = direction * self.tableau[goal_r, c]
        if c_factor == 0.: continue
        if co.restricted and c_factor <= 0.: continue

      # We may be constrained by restricted rows in this column
      most_constraining_restricted_row = None  # most constraining row in this column
      # how much we can increase c * c_factor considering the restricted rows
      most_constraining_increase = np.inf
      candidate_row = None
      candidate_row_increase = 0.
      for r, ro in enumerate(self.row_owner):
        t_rc = self.tableau[r, c]
        if t_rc != 0.:
          # The increase in sample value of `c * factor_c` if we pivot at (r, c)
          increase_for_rc = - c_factor * self.tableau[r, 0] / t_rc
          if ro.restricted and c_factor * t_rc < 0 and increase_for_rc < most_constraining_increase:
            most_constraining_restricted_row, most_constraining_increase = r, increase_for_rc
          elif increase_for_rc >= 0. and increase_for_rc > candidate_row_increase:
            candidate_row = r  # We could pivot on this row, if nothing else is more restrictive

      if most_constraining_restricted_row is None:
        # We can increase c indefinitely, but we want to include a row that is pivotable
        return candidate_row, c, np.inf
      elif most_constraining_restricted_row >= p_increase:
          p_row, p_col, p_increase = most_constraining_restricted_row, c, most_constraining_increase

    return p_row, p_col, p_increase

  def improve_var(self, v: _Simplex.Var,
                  direction: int,
                  for_restriction: bool = False) -> float:
    """

    Args:
      v: the var to find the bounds for
      direction: +1 to find the upper bound, -1 to find the lower bound.
      for_restriction: we improve a Var that is not restricted, in a row,
        and with negative sample value in preparation for restriction. We
        must bring its sample value to >= 0. In this case `direction` is 1.
    """
    if v.col is not None:
      assert not for_restriction
      # We should try to pivot it into a row so that we can start improving
      # its sample value
      if v.restricted and direction < 0: return 0.  # Already optimal at 0.
      if self.nr_rows == 0:
        return direction * np.inf
      # In order to find the bounds, we must pivot it into a row
      p_row, p_col, best_improvement = self.find_best_pivot(only_in_column=v.col,
                                                            direction=direction,
                                                            goal_r=None)
      if best_improvement == np.inf:
        return direction * np.inf
      self.pivot(p_row, v.col)

    goal_r = v.row
    max_iteration_bound = 100   # Safety net for termination
    while True:
      max_iteration_bound -= 1
      assert max_iteration_bound >= 0
      t_g0 = self.tableau[goal_r, 0]
      if for_restriction:
        assert direction > 0
        if t_g0 >= 0:
          return int(np.floor(t_g0))

      # Find a pivot column to optimize goal_r
      p_row, p_col, best_improvement = self.find_best_pivot(
          goal_r=goal_r,
          direction=direction,
          only_in_column=None)
      if best_improvement == np.inf:
        if for_restriction:
          # We know that we can optimize goal_r to infinity, but we need to
          # still pivot it before we restrict. p_col is a column that can be
          # changed unboundedly.
          assert direction > 0
          assert t_g0 < 0.  # Otherwise we bail out earlier
          assert p_row is not None  # At least goal_r is pivotable
          self.pivot(p_row, p_col)
        return direction * np.inf
      if best_improvement == 0. and p_row == -1 and p_col == -1:
        if direction < 0:
          return int(np.ceil(self.tableau[goal_r, 0]))
        else:
          return int(np.floor(self.tableau[goal_r, 0]))

      if p_row == goal_r:
        # Avoid pivoting goal_r into a column, because we may soon want to
        # optimize it in another direction
        # TODO: check soundness here
        assert self.row_owner[goal_r].restricted
        assert not for_restriction
        assert direction == -1
        assert self.tableau[goal_r, 0] == best_improvement
        return 0
      assert self.row_owner[p_row].restricted
      self.pivot(p_row, p_col)

  def pivot(self, r: int, c: int):
    pivot_elem = self.tableau[r, c]
    # TODO: do we need the copy?
    prev_pivot_col = np.copy(self.tableau[:, c])
    self.tableau[r] = self.tableau[r] / (-1. * pivot_elem)
    for i in range(self.nr_rows):
      if i != r:
        self.tableau[i] += self.tableau[r] * self.tableau[i, c]

    self.tableau[:, c] = prev_pivot_col / pivot_elem
    self.tableau[r, c] = 1. / pivot_elem

    if not self.within_tolerance_of_some_integer(self.tableau[r, c] * self.denominator):
      self.denominator = int(np.abs(self.round_to_tolerance(self.denominator * pivot_elem)))
      if config.enable_checks.value:
        scaled_pivot_cell = self.tableau[r, c] * self.denominator
        assert self.within_tolerance_of_some_integer(scaled_pivot_cell)
    prev_ro = self.row_owner[r]
    prev_ro.row = None
    prev_ro.col = c
    prev_co = self.col_owner[c]
    prev_co.row = r
    prev_co.col = None
    self.row_owner[r] = prev_co
    self.col_owner[c] = prev_ro
    self.check_invariant_if_enabled()

  def expand_tableau(self, rows=None, cols=None):
    new_rows = max(self.tableau.shape[0], rows or self.tableau.shape[0])
    new_cols = max(self.tableau.shape[1], cols or self.tableau.shape[1])
    new_tableau = np.zeros((new_rows, new_cols), dtype=self.tableau.dtype)
    new_tableau[:self.nr_rows, :self.nr_cols] = self.tableau[:self.nr_rows, :self.nr_cols]
    self.tableau = new_tableau

  def sample_solution_assignment(self) -> dict[str, int]:
    """For error messages"""
    assignment: dict[str, int] = {}
    scaled_tableau = (self.tableau * self.denominator).astype(int)
    for co in self.col_owner:
      if co.short_name != "1" and (co.info is None or co.info.type != _Simplex.AssumeType.GOAL):
        assignment[str(co.expr)] = 0
    for ro in self.row_owner:
      if ro.info is None or not ro.info.type != _Simplex.AssumeType.GOAL:
        if np.floor(self.tableau[ro.row, 0]) == self.tableau[ro.row, 0]:
          # No scaling needed
          assignment[str(ro.expr)] = int(self.tableau[ro.row, 0])
        else:
          # TODO: maybe refine the solution to integers
          assignment[str(self.denominator * ro.expr)] = scaled_tableau[ro.row, 0]
    return assignment

  def within_tolerance_of_some_integer(self, e: np.ndarray) -> np.ndarray:
    return self.within_tolerance_of_integer(e, np.round(e))

  def within_tolerance_of_integer(self, e: np.ndarray, i: int) -> np.ndarray:
    return (np.abs(e - i) < _Simplex.TOLERANCE).all()

  def round_to_tolerance(self, e: np.ndarray) -> np.ndarray:
    if config.enable_checks.value:
      assert self.within_tolerance_of_some_integer(e)
    return np.round(e)
