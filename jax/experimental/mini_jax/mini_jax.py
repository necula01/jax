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
"""Mini-JAX: a pedagogical model of JAX tracing and transformations.

See README.md
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import enum
import functools
import itertools
import typing
from typing import Any, Dict, Callable, Tuple, Sequence, List, Optional, Union

from jax import pprint_util as ppu
from jax.pprint_util import PrettyPrint, pp_kv_pairs
from jax.experimental.mini_jax.mini_jax_util import (
  map_tuple, map_list, unzip,
  pp_str, pp_list
)

TA = typing.TypeVar('TA')
TB = typing.TypeVar('TB')
Value = Union[Any, 'Tracer']  # Either a Python value, or a Tracer


class Globals(object):
  variable_id = itertools.count()  # Unique ids for variables
  function_id = itertools.count()  # Unique ids for functions (for JIT)
  scope_nesting_depth = 0

  @staticmethod
  def reset():
    """Reset counters, for deterministic testing."""
    Globals.variable_id = itertools.count()
    Globals.function_id = itertools.count()
    Globals.scope_nesting_depth = 0


class ExprType(object):
  """Types for expressions."""

  def __init__(self, dtype: type):
    self.dtype = dtype

  def __repr__(self):
    return self.dtype.__name__

  __str__ = __repr__


class Operator(enum.Enum):
  """Operators are applied to `Expr` arguments and may have parameters (a Dict)."""
  LITERAL = "literal"
  """Represents a literal, no arguments. params = {var : constant}"""

  VAR = "var"  # 0-ary, params = {'id' : int}
  """Represents a variable, no arguments, params = {id: int}. Variables get
  assigned unique integers.
  """

  ADD = "add"  # binary, no params
  SUB = "sub"  # binary, no params
  MUL = "mul"  # binary, no params
  POW = "pow"
  """Exponentiation, unary, has one param {pow: int}."""

  JIT_CALL = "jit_call"
  """A jit-wrapped Function. One params {func: Function}. Has as many arguments
  as there are func.invars. If the result is a tuple, it must be a tuple with 
  at least 2 elements. In that case, `e.etype` is a tuple, and 
  `len(e.etype) = len(func.bodies)`.
  """

  COND_GE = "cond_ge"
  """Conditional for greater-equal to 0. Has the following parameters
    func_true: a Function, for the true  branch
    func_false: a Function, for the false branch
    
  The arguments are in order:
    1 argument representing the value to be compared
    `len(func_true.invars)` arguments to be passed to the `true_func`
    `len(fund_false.invars)` arguments to be passed to the `false_func`.
    
  When pretty-printing, the arguments are actually shown as part of 
  parameters:
    pred_arg:
    true_args: arguments to be passed to the `true_func`
    false_args: arguments to be passed to the `false_func`
  """

  PROJECTION = "proj"
  """Used to extract a single value from an expression that denotes a tuple. 
  There is one argument, whose `e.args[0].etype` is a sequence (can be a
  JIT_CALL or COND_GE). The parameters are {idx: int}.
  """

  def __repr__(self):
    return self.value

  __str__ = __repr__


class Expr(object):
  """Symbolic typed expressions."""

  def __init__(self,
               operator: Operator,
               args: Sequence['Expr'],
               etype: Optional[ExprType] = None,
               **params):
    self.operator = operator
    self.params = params
    self.args = args
    if etype is not None:
      self.etype = etype
    else:
      self.etype = Expr.type_check(self.operator, params,
                                   map_tuple(lambda e: e.etype, args))

  def __repr__(self):
    if self.operator == Operator.VAR:
      return "v{}".format(self.params['id'])
    elif self.operator == Operator.LITERAL:
      return str(self.params['val'])
    else:
      # This is for debugging only, we normally pretty print entire functions
      bindings, names = Expr.three_address_code([self])
      return str(Expr.pp_three_address_code(bindings, names))

  __str__ = __repr__

  @staticmethod
  def type_check(op: str, params: Dict,
                 args_et: Sequence[ExprType]) -> ExprType:
    """Computes the type for an operator application given types for arguments."""
    if op in (Operator.ADD, Operator.MUL, Operator.SUB):
      arg1_et, arg2_et = args_et
      if arg1_et.dtype is float and arg2_et.dtype is float:
        return arg1_et
      assert False, "{} {} {}".format(op, arg1_et, arg2_et)
    if op == Operator.POW:
      if args_et[0].dtype is float:
        return args_et[0]
      assert False
    raise NotImplementedError

  @staticmethod
  def eval_operator_tracer(op: Operator, params, args_v: Sequence[Value],
                           env: Dict[int, Value] = None) -> Value:
    """Evaluates an expression, given values or tracing values for arguments.

    If any of the arguments is a tracer, the result is a tracer, with the
    corresponding symbolic expression. If all arguments are Python values,
    then the expression is evaluated and the result is a Python value.

    Args:
      args_v: the values for the `self.args`
      env: values for the variables, indexed by variable id;
        needed only if self.operator == VAR.
    """
    if op == Operator.VAR:
      return env[params["id"]]
    if op == Operator.LITERAL:
      return params["val"]
    if op == Operator.ADD:
      return args_v[0] + args_v[1]
    if op == Operator.SUB:
      return args_v[0] - args_v[1]
    if op == Operator.MUL:
      return args_v[0] * args_v[1]
    if op == Operator.POW:
      return args_v[0] ** params["pow"]
    if op == Operator.PROJECTION:
      return args_v[0][params["idx"]]
    if op == Operator.JIT_CALL:
      func = params["func"]
      assert len(args_v) == len(func.invars)
      all_constants = all([not isinstance(a, Tracer) for a in args_v])
      if all_constants:  # Arguments are all Python values: JIT and execute
        return Jit.compile_and_execute(func, args_v)

      args_t = map_list(Tracer.val_to_tracer, args_v)
      call_res = Tracer.build(Operator.JIT_CALL, dict(func=func),
                              args_t, etype=func.result_type())
      return Tracer.handle_tuple_result(call_res)

    if op == Operator.COND_GE:
      true_func_f = params["true_func"]
      false_func_f = params["false_func"]
      assert len(true_func_f.bodies) == len(false_func_f.bodies)
      all_constants = all([not isinstance(a, Tracer) for a in args_v])
      if all_constants:  # Arguments are all Python values: evaluate now
        if args_v[0] >= 0.:
          # Cheat and use the jitter to evaluate
          return Jit.compile_and_execute(
            true_func_f, args_v[1:1 + len(true_func_f.invars)])
        else:
          return Jit.compile_and_execute(
            false_func_f, args_v[1 + len(true_func_f.invars):])
      cond_t = Tracer.build(Operator.COND_GE, params,
                            map_tuple(Tracer.val_to_tracer, args_v),
                            etype=true_func_f.result_type())
      return Tracer.handle_tuple_result(cond_t)

    raise NotImplementedError

  def visit_expr(self,
                 visitor: Callable[['Expr', Sequence[TA]], TA],
                 memo: Dict[int, TA] = None,
                 **visitor_params) -> TA:
    """Visit an expression, applying a visitor to all sub-expressions, with memoization.
    Args:
      visitor: a function called with an expression being visited, and
        the result of visiting its `args`. The results of visiting expressions
        should not be mutated; they are stored in a memo table.
      memo: a memo table to use, useful when we want to share memo table across
        multiple expression visits. Default is to use an internal memo table,
        whose scope is this call to `visit_expr`.
      visitor_params: parameters to pass to the visitors.
    Returns: the result of visiting the expression
    """
    memo = {} if memo is None else memo  # Map id(e) to the result of visiting e
    sentinel = object()

    def do_visit(e: 'Expr') -> TA:
      res = memo.get(id(e), sentinel)
      if res is not sentinel:
        return res
      args_v = map_tuple(do_visit, e.args)
      memo[id(e)] = res = visitor(e, args_v, **visitor_params)
      return res

    return do_visit(self)

  @staticmethod
  def make_memoized_expr_evaluator(env: Dict[int, Value]) -> Callable[['Expr'], Value]:
    expr_values = {}  # Dict[int, Value] - memoize the value of sub-expressions

    def eval_expr(e: 'Expr') -> Value:
      return e.visit_expr(
        lambda e, args_v: Expr.eval_operator_tracer(e.operator, e.params, args_v,
                                                    env),
        memo=expr_values)
    return eval_expr

  @staticmethod
  def three_address_code(elst: List['Expr']) -> Tuple[List[Tuple[str, 'Expr']],
                                                      List['Expr']]:
    """Converts a list of `Expr` to 3-address-codes.

    All sub-expressions that are not literals or `VAR` are turned into::

      n = op arg1 ...

    where `n` are new names, `arg1` are all *simple* expressions (literal, VAR,
    or one of the previously defined names. Shared sub-expressions will reuse
    the same name.)

    Returns:
      a pair with: a list of bound sub-expressions (each with a name), and
      a list of simple expressions corresponding to the input `elst`.
    """
    bindings = []  # List[Tuple(str, Expr)] list of bound non-simple Expr.
    memo = dict()  # Share the memo across all expressions in the list
    name_id = itertools.count()  # New names local to this expression list

    def visitor_simplify(e: Expr, args_v: Sequence[Expr]) -> Expr:
      """Simplification visitor.
      Returns:
        a simple expression.
      """
      if e.operator in [Operator.VAR, Operator.LITERAL]:
        return e
      else:
        name = "n{}".format(next(name_id))
        binding = (name, Expr(e.operator, args_v, etype=e.etype, **e.params))
        bindings.append(binding)
        return name

    names = map_tuple(lambda e: e.visit_expr(visitor_simplify, memo=memo), elst)
    return (bindings, names)

  @staticmethod
  def pp_three_address_code(bindings: Sequence[Tuple[str, 'Expr']],
                            names: Sequence[str]):
    """Pretty-prints 3-address-code."""

    def pp_binding(binding: Tuple[str, Expr]):
      name, e = binding
      # Special case the printing of some operators
      args, params = e.args, e.params
      if e.operator == Operator.COND_GE:
        # Put the true_args and false_args among parameters, make it easier to read
        params = dict(**params)
        params["pred_arg"] = args[0]
        params["true_args"] = args[1:1+len(params["true_func"].invars)]
        params["false_args"] = args[1 + len(params["true_func"].invars):]
        args = []
      pp_args = pp_str(" ".join(map_list(str, args)))
      return (pp_str("{} = {}".format(name, e.operator)) >>
              pp_kv_pairs(sorted(params.items())) >>
              pp_str(" ") >> pp_args)

    if len(names) > 1:
      result = pp_str("in (") >> pp_list(names) >> pp_str(",)")
    else:
      result = pp_str("in {}".format(names[0]))
    return (pp_list(map_list(pp_binding, bindings), vertical=True) +
            result)



class Function(object):
  """Denotes a closed function."""

  def __init__(self,
               invars: Sequence[Expr],
               bodies: Sequence[Expr]):
    """
    If there are more than 1 bodies, then and only then the function returns
    a tuple.

    Args:
      invars: the variables that occur in the body, including free variables.
      bodies: a list of Expr that contain only the Vars in `invars`.
    """
    self.bodies = bodies
    self.invars = invars

  def __repr__(self):
    return str(self.pp())

  __str__ = __repr__

  def pp(self):
    """Pretty prints a function definition, using 3-address-codes."""
    bindings, names = Expr.three_address_code(self.bodies)
    return ((pp_str("{lambda ") >> pp_list(self.invars) >> pp_str(".")) +
            (pp_str("  # ") >> pp_list(["{}: {}".format(v, v.etype)
                                        for v in self.invars], hsep=", ")) +
            Expr.pp_three_address_code(bindings, names).indent(2)
            >> pp_str("}"))

  @staticmethod
  def trace_callable(func: Callable, args_et: List[ExprType]
                     ) -> Tuple['Function', Sequence['Tracer']]:
    """Trace a callable given some types for the arguments.

    Args:
      args_et: types corresponding to the `func` arguments

    Returns: a pair of a closed `Function` along with a list of `Tracer`s
      corresponding to computations from shallower scopes used in the function
      body (the environment). The tail of the `invars` of the function correspond
      to the environment values.
    """
    Globals.scope_nesting_depth += 1
    try:
      scope_nesting_depth = Globals.scope_nesting_depth

      def make_tracing_variable(etype):
        return Tracer.build(Operator.VAR,
                            dict(id=next(Globals.variable_id)),
                            (), etype=etype)

      args_t = map_tuple(make_tracing_variable, args_et)
      res = func(*args_t)
      if not isinstance(res, tuple):
        res = (res,)

      # res may contain literals, turn them into Tracer
      res_t = tuple(Tracer.val_to_tracer(r) for r in res)
      res_e, res_env = Tracer._force_scope_depth(res_t, scope_nesting_depth)
      freevars = []
      freevars_env = []
      for v, env_t in res_env:
        if v not in freevars:  # We may have a variable twice in an environement
          freevars.append(v)
          freevars_env.append(env_t)

      return (
        Function([v_t.expr for v_t in args_t] + freevars, res_e),
        freevars_env)
    finally:
      Globals.scope_nesting_depth -= 1

  def trace_interpreter(
      self,
      interpreter: Callable[['Function', Sequence['Tracer']], Any],
      args_t: Sequence[ExprType] = None
  ) -> 'Function':
    """Runs a traceable interpreter over a Function to produce transformed Function.

    This is the workhorse of composable transformations. Given an interpreter
    that evaluates a `Function` with different semantics for operators and uses
    only overloaded operations on the arguments, run `trace_interpreter` to
    interpret the function and get the transformed `Function`.

    Args:
      interpreter: a Python traceable function, that given a `Function` and
        a set of Tracer evaluates the Function according to the transformed
        semantics.
      args_t: optional, the list of types for the resulting function. If not
        given, then use the types of the original function.

    Returns:
      a transformed Function.
    """
    args_t = [inv.etype for inv in self.invars] if args_t is None else args_t
    res_func_f, res_func_env = Function.trace_callable(
      lambda *x: interpreter(self, *x),
      args_t)
    assert not res_func_env, "The function was already closed"
    return res_func_f

  def result_type(self):
    """The result type of the function.
    The result type is a tuple iff there are more than 1 bodies.
    """
    res = [body.etype for body in self.bodies]
    return res[0] if len(res) == 1 else res

  def visit_bodies_memoized(self,
                            visitor: Callable[[Expr, Sequence[TA]], TA],
                            **visitor_params
                            ) -> Sequence[TA]:
    """Visit all bodies in order, memoized with a shared memo table.
    Returns: the result of visiting the bodies
    """
    memo = {}
    return [body.visit_expr(visitor, memo=memo, **visitor_params)
            for body in self.bodies]

# An environment is a sequence of pairs of variables and the tracing values
# they stand for from shallower scope depths.
Environment = Sequence[Tuple[Expr, 'Tracer']]


class Tracer(object):
  """A value to be used in lieu of actual Python values for tracing."""

  def __init__(self, expr: Expr,
               scope_nesting_depth: int,
               env: Environment):
    """
    Args:
      expr: the expression representing the traced value.
      scope_nesting_depth: the scope depth at which was built. May be `None`
        for literals.
      env: the environment for the expression.
    """
    self.expr = expr
    self.scope_nesting_depth = scope_nesting_depth
    self.env = env

  @staticmethod
  def val_to_tracer(v: Value) -> 'Tracer':
    """Make a Tracer from a Value."""
    if isinstance(v, Tracer):
      return v
    # Must be a constant
    assert isinstance(v, (int, float)), "{}".format(v)
    if isinstance(v, int):
      raise NotImplementedError
    elif isinstance(v, float):
      v_et = ExprType(float)
    else:
      assert False, "{}".format(v)
    return Tracer.build(Operator.LITERAL, dict(val=v), (), etype=v_et)

  def __repr__(self):
    op = self.expr.operator  # Special-case some operators, for debugging
    if op in (Operator.VAR, Operator.LITERAL):
      fmt = str(self.expr)
    else:
      fmt = op
    return str("Tr[{}/{}]".format(fmt, self.scope_nesting_depth))

  __str__ = __repr__

  @staticmethod
  def val_to_type(arg: Value) -> ExprType:
    """Given a Value, get its type."""
    return Tracer.val_to_tracer(arg).expr.etype

  @staticmethod
  def _force_scope_depth(args_t: Sequence['Tracer'],
                         scope_nesting_depth: int
                         ) -> Tuple[Sequence[Expr], Environment]:
    """Prepares a list of tracing values for use at a given scope depth.

    If there are tracers from shallower scope depths, they are replaced
    with new variables, and an environment is constructed for these variables
    along with the shallow tracers they represent.
    """
    env = []  # Sequence[Environment]
    args_e = []  # Sequence[Expr]
    for a_t in args_t:
      if a_t.scope_nesting_depth is None or a_t.scope_nesting_depth == scope_nesting_depth:
        env.extend(a_t.env)
        args_e.append(a_t.expr)
      else:
        assert a_t.scope_nesting_depth < scope_nesting_depth
        if a_t.expr.operator == Operator.VAR:  # Reuse variables, for convenience
          new_v = a_t.expr
        else:
          new_v = Expr(Operator.VAR, (), etype=a_t.expr.etype,
                       id=next(Globals.variable_id))
        env.append((new_v, a_t))
        args_e.append(new_v)

    return (args_e, env)

  @staticmethod
  def build(op: str, params: Dict,
            args_t: Sequence['Tracer'],
            etype: Optional[ExprType] = None,
            ) -> 'Tracer':
    """Builds a Tracer for an operator applied to arguments.

    Args:
      op: the operator.
      params: the operator parameters.
      args_t: the `Tracer`s for the arguments.
      etype: the optional `ExprType` for the `Expr` being built. If not given,
        then the typeis calculated through type checking.
    Returns:
      a `Tracer` at the current scope nesting depth.
    """
    args_e, args_env = Tracer._force_scope_depth(args_t,
                                                 Globals.scope_nesting_depth)
    expr = Expr(op, tuple(args_e), etype=etype, **params)
    return Tracer(expr, Globals.scope_nesting_depth, args_env)

  @staticmethod
  def handle_tuple_result(res: 'Tracer'
                          ) -> Union['Tracer', Sequence['Tracer']]:
    """Wrap tracing val in projections, if we need to return a tuple result.
    Args:
      res: a tracing value, that has a tuple `res.expr.etype`
    """
    if isinstance(res.expr.etype, (tuple, list)):
      assert len(res.expr.etype) > 1
      res_tuple = map_tuple(
        lambda idx: Tracer.build(Operator.PROJECTION,
                                 dict(idx=idx), (res,),
                                 etype=res.expr.etype[idx]),
        range(len(res.expr.etype)))
      return res_tuple
    else:
      return res

  ############## Overload Python operators
  def __add__(self, other: Value) -> 'Tracer':
    if other == 0.: return self
    return Tracer.build(Operator.ADD, {}, (self,
                                           self.val_to_tracer(other)))

  def __radd__(self, other: Value) -> 'Tracer':
    if other == 0.: return self
    return Tracer.build(Operator.ADD, {}, (self.val_to_tracer(other),
                                           self))

  def __mul__(self, other: Value) -> 'Tracer':
    if other == 0.: return 0.
    if other == 1.: return self
    return Tracer.build(Operator.MUL, {}, (self,
                                           self.val_to_tracer(other)))

  def __rmul__(self, other: Value) -> 'Tracer':
    if other == 0.: return 0.
    if other == 1.: return self
    return Tracer.build(Operator.MUL, {}, (self.val_to_tracer(other),
                                           self))

  def __sub__(self, other: Value) -> 'Tracer':
    if other == 0.: return self
    return Tracer.build(Operator.SUB, {}, (self,
                                           self.val_to_tracer(other)))

  def __rsub__(self, other: Value) -> 'Tracer':
    return Tracer.build(Operator.SUB, {}, (self.val_to_tracer(other),
                                           self))

  def __pow__(self, power: int) -> 'Tracer':
    assert isinstance(power, int)
    if power == 0: return 1.
    if power == 1: return self
    return Tracer.build(Operator.POW, dict(pow=power), (self,))


class Ops(object):
  """Special primitive operations that operate either on values of tracers"""

  @staticmethod
  def cond_ge(to_test: Value, true_func: Callable, true_args: Sequence[Value],
              false_func: Callable, false_args: Sequence[Value]
              ) -> Sequence[Value]:
    # Collect the branch functions
    true_func_f, true_func_env = Function.trace_callable(
      true_func, map_list(Tracer.val_to_type, true_args))
    false_func_f, false_func_env = Function.trace_callable(
      false_func, map_list(Tracer.val_to_type, false_args))
    assert str(true_func_f.result_type()) == str(false_func_f.result_type()), (
      "{} != {}".format(true_func_f.result_type(), false_func_f.result_type()))
    return Expr.eval_operator_tracer(
      Operator.COND_GE,
      dict(true_func=true_func_f,
           false_func=false_func_f),
      tuple(itertools.chain([to_test],
                            true_args, true_func_env,
                            false_args, false_func_env)))


############ TRACE ##############
def trace(func: Callable) -> Callable[..., Function]:
  """
  Returns: a function that when applied to `func` arguments traces the function
    and returns the `Function`.
  """

  def wrapped_trace(*args: Sequence[Value]):
    assert Globals.scope_nesting_depth == 0, "Outside any other tracers"
    Globals.reset()  # Produces more deterministic results

    func_f, func_f_env = Function.trace_callable(func,
                                                 map_list(
                                                   Tracer.val_to_type,
                                                   args))
    assert not func_f_env  # We usually trace only outside other transformations
    return func_f

  return wrapped_trace


# Circular dependency Jit<->Mini-jax core
from jax.experimental.mini_jax.mini_jax_jit import Jit
