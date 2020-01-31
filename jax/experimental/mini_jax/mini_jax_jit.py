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
Transformation: JIT
--------------------

The JIT transformation in mini-JAX consists of pretty-printing the symbolic
expressions to Python source code, and `exec`-ing the code. This is very simple
but does expose the interesting aspects of supporting JIT. Most of these
aspects permeate the core mini-JAX (through the presence of higher-order
`JIT_CALL` operator), so the code in this file is actually not very interesting.

You can set the environment variable `MINI_JAX_LOG_COMPILES=1` to see
the code being compiled.

Concrete examples are in `tests/mini_jax_test.py`.
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import functools
import numpy as np
import os
from typing import Dict, Callable, Tuple, Sequence, List

from jax.experimental.mini_jax.mini_jax import (
  Expr, ExprType, Operator, Function, Tracer,
  Value, Globals, Cache
)
from jax.experimental.mini_jax.mini_jax_util import map_list, map_tuple, \
  pp_seq, pp_str
from jax.pprint_util import PrettyPrint


class Jit(object):
  """Methods related to compilation."""

  @staticmethod
  def compile_expr_assigned(e: Expr, args_s: Sequence[str], name: str) -> PrettyPrint:
    """Compiles the application of an operator.

    Produces a PrettyPrint for assigning the result of the operator::

       name = args_s[0] e.operator args_s[1]

    Args:
      e: the expression to compile, a *simple expression*.
      args_s: the string forms of the *simple* arguments
      name: the name for assigning the result of computing the expression
    """
    if e.operator in [Operator.LITERAL, Operator.VAR]:
      return pp_str("{} = {}".format(name, Jit.compile_simple_expr(e)))
    if e.operator == Operator.ADD:
      return pp_str("{} = {} + {}".format(name, *args_s))
    if e.operator == Operator.SUB:
      return pp_str("{} = {} - {}".format(name, *args_s))
    if e.operator == Operator.MUL:
      return pp_str("{} = {} * {}".format(name, *args_s))
    if e.operator == Operator.POW:
      return pp_str("{} = {} ** {}".format(name, args_s[0], e.params["pow"]))
    if e.operator == Operator.PROJECTION:
      return pp_str(
        "{} = {}[{}]".format(name, args_s[0], e.params["idx"]))

    if e.operator == Operator.JIT_CALL:
      return Jit.compile_function_call(e.params["func"], args_s, name)

    if e.operator == Operator.COND_GE:
      true_func_f = e.params["true_func"]
      true_func_compiled = Jit.compile_function_call(
        true_func_f, args_s[1:], name)
      false_func_f = e.params["false_func"]
      false_func_compiled = Jit.compile_function_call(
        false_func_f, args_s[1:], name)
      return (pp_str("if {} >= 0.:".format(args_s[0])) +
              true_func_compiled.indent(2) +
              pp_str("else:") +
              false_func_compiled.indent(2))

    raise NotImplementedError

  @staticmethod
  def compile_simple_expr(e: Expr) -> str:
    """Compiles a 'simple' expression, as from three_address_code."""
    if isinstance(e, str):  # a name
      return e
    elif e.operator == Operator.LITERAL:
      return repr(e.params['val'])
    elif e.operator == Operator.VAR:
      return str(e)
    else:
      assert False

  @staticmethod
  def compile_function(func: Function) -> Tuple[str, PrettyPrint]:
    """Compile the function into a PrettyPrinter representing
    an executable Python string.

    Generate a string of the form::

      def fxxx(v0, v1, ..., vn):  # The invars
        n0 = v0 op v1             # The body in 3-address-form
        ...
        return [n3, n4]           # The names in 3-address-form

    Args:
      func: the function to compile

    Returns: a pair, with the function name, and the PrettyPrinter for
      printing the function definition.
    """
    cache_key = "jit_compile"
    result = Cache.get(func, cache_key)
    if result is not None:
      return result
    bindings, names = Expr.three_address_code(func.results)
    func_name = "f{}".format(next(Globals.function_id))
    header = (pp_str("def {}(".format(func_name)) >>
              pp_seq(func.invars, hsep=", ") >> pp_str("):"))
    header_types = (pp_str("  # ") >>
                    pp_seq(["{}: {}".format(v, v.etype)
                            for v in func.invars], hsep=", "))

    def compile_binding(bind):
      name, expr = bind
      return Jit.compile_expr_assigned(expr,
                                       map_tuple(Jit.compile_simple_expr, expr.args),
                                       name)

    body = pp_seq(map_list(compile_binding, bindings), vertical=True)
    if len(names) > 1:
      result = pp_str("return (") >> pp_seq(names, hsep=", ") >> pp_str(", )")
    else:
      result = pp_str("return {}".format(names[0]))

    result = (func_name, header + header_types + (body + result).indent(2))
    Cache.set(func, cache_key, result)
    return result

  @staticmethod
  def compile_function_call(func: Function,
                            args_s: Sequence[str], res_name: str
                            ) -> PrettyPrint:
    """Compile the function and a call to it.
    Args:
      func: the function to compile
      args_s: strings representing the arguments
      res_s: string for where to put the result

    Returns: the PrettyPrinter for
      printing the function definition and the call.
    """
    func_name, func_body = Jit.compile_function(func)
    func_call = (pp_str("{} = {}(".format(res_name, func_name)) >>
                 pp_seq(args_s, hsep=", ") >> pp_str(")"))
    return func_body + func_call

  @staticmethod
  def compile_and_execute(func: Function, args: Sequence[Value]) -> Value:
    """Compile the function into a Python string that can be exec.

    Args:
      args: actual Python values (no TracingVals)
    """
    assert len(args) == len(func.invars)
    compiled = Jit.compile_function_call(func, map_tuple(str, func.invars),
                                         "_result")
    fun_locals = {str(iv): arg for iv, arg in zip(func.invars, args)}
    compiled_str = str(compiled)
    if os.getenv("MINI_JAX_LOG_COMPILES", 0):
      print("Running compiled function:\n" + compiled_str)
    exec(compiled_str, {"array": np.array, "float32": np.float32}, fun_locals)
    return fun_locals['_result']


def jit(func: Callable, cache: bool = True):
  """
  Args:
    func: the user function to be traced and jitted
    cache: whether to allow the caching of the result of tracing

  Returns: a function that when applied to `func` arguments traces the function
    and builds an `Expr` that when executed, will JIT-compile the function
    and then execute it.
  """

  def do_jit(*args: Value) -> Value:
    func_f, func_f_env = Function.trace_user_function(func, args,
                                                      abstract=True,
                                                      cache=cache)
    # Turn it into an Expr, or evaluate if none of the arguments are Tracer
    return Expr.eval_std_operator(Operator.JIT_CALL,
                                  dict(func=func_f),
                                  list(args) + list(func_f_env))

  return do_jit
