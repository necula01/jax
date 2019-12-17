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

"""An attempt to use Hypothesis to generate interested inputs for JAX.

Use as follows:
  JAX_NUM_GENERATED_CASES=1 JAX_HYPOTHESIS_EXAMPLES=2 pytest -n auto --durations=20 jax/experimental/mini_jax/tests/mini_jax_hypothesis_test.py --hypothesis-show-statistics --hypothesis-verbosity=verbose

or to see the output (don't know how to make pytest not capture the output):
  JAX_HYPOTHESIS_VERBOSITY=1 JAX_HYPOTHESIS_EXAMPLES=1000 python jax/experimental/mini_jax/tests/mini_jax_hypothesis_test.py
  # verbosity = 1 (normal), 2 (verbose), 3 (debug)
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from absl.testing import absltest
from collections import defaultdict
import functools
import itertools
from jax import test_util as jtu
from jax.pprint_util import pp

import os
from typing import Callable, List, Dict, NamedTuple, Tuple, Union

from jax.experimental import mini_jax as mj
from jax.experimental.mini_jax.mini_jax import Operator, Globals
from jax.experimental.mini_jax.mini_jax_util import pp_list, pp_str, \
  PrettyPrint, map_list

import numpy as np
try:
  from hypothesis.strategies._internal.core import cacheable, defines_strategy_with_reusable_values
except ImportError:
  def cacheable(f):
    return f
  def defines_strategy_with_reusable_values(f):
    return f

from hypothesis import given, settings, assume, seed
from hypothesis import strategies as st
from hypothesis import stateful
import traceback

HYPOTHESIS_EXAMPLES = int(os.environ.get("JAX_HYPOTHESIS_EXAMPLES", 0))
HYPOTHESIS_VERBOSITY = int(os.environ.get("JAX_HYPOTHESIS_VERBOSITY", 1))
# quiet = 0
# normal = 1
# verbose = 2
# debug = 3

if HYPOTHESIS_EXAMPLES == 0:
  USE_HYPOTHESIS = False
  HYPOTHESIS_EXAMPLES = 1  # Hypothesis does not like 0 examples
else:
  USE_HYPOTHESIS = True

class NumericalDiffDiscontinuity(Exception):
  pass

class FakeMiniJax(object):
  """A fake implementation of the mini-JAX transformations."""

  @staticmethod
  def cond_ge(pred, true_func, true_ops, false_func, false_ops):
    if pred >= 0.:
      return true_func(*true_ops)
    else:
      return false_func(*false_ops)

  @staticmethod
  def sum(*args):
    return functools.reduce(lambda acc, a: acc + a, args, 0.)

  @staticmethod
  def jit(func):
    return func

  @staticmethod
  def jvp(func):
    """Estimate tangent numerically."""
    def wrapped(*args_with_tangents):
      assert len(args_with_tangents) % 2 == 0
      nr_orig_vars = len(args_with_tangents) // 2
      args = args_with_tangents[0:nr_orig_vars]
      args_tan = args_with_tangents[nr_orig_vars:]

      EPS = 1e-4
      def eval_eps(eps):
        return func(*[x + eps * t
                      for x, t in zip(args, args_tan)])

      def numeric_tan(first, second):
        # Return the tangent from first->second (for arguments at distance EPS)
        if isinstance(first, tuple):
          return [(s - f) / EPS for f, s in zip(first, second)]
        else:
          return (second - first) / EPS

      # Check that tangent is the same x - EPS, x + EPS, x + 2*EPS
      # If not, try also x + 2 * ESP
      res_m1 = eval_eps(- EPS)
      res_0 = eval_eps(0.)
      res_1 = eval_eps(EPS)
      tan_0 = numeric_tan(res_m1, res_0)
      tan_1 = numeric_tan(res_0, res_1)
      if not np.allclose(tan_0, tan_1):
        raise NumericalDiffDiscontinuity()

      if isinstance(res_0, tuple):
        return tuple(itertools.chain(res_0, tan_0))
      else:
        return (res_0, tan_0)

    return wrapped

  @staticmethod
  def grad(func):
    jvp_func = FakeMiniJax.jvp(func)

    def wrapped(*args):
      partial_derivatives = []
      for i in range(len(args)):
        args_with_tan = tuple(
          itertools.chain(args,
                          ([0.] * i + [1.] + [0.] * (len(args) - i - 1))))
        _, res_tan = jvp_func(*args_with_tan)
        partial_derivatives.append(res_tan)
      if len(partial_derivatives) == 1:
        return partial_derivatives[0]
      else:
        return tuple(partial_derivatives)

    return wrapped


SUT_MJ = dict(jit=mj.jit, grad=mj.grad, jvp=mj.jvp, cond_ge=mj.Ops.cond_ge,
              sum=FakeMiniJax.sum)
SUT_FAKE = dict(jit=FakeMiniJax.jit, grad=FakeMiniJax.grad,
                jvp=FakeMiniJax.jvp, cond_ge=FakeMiniJax.cond_ge,
                sum=FakeMiniJax.sum)


def check_code_example(code: PrettyPrint):
  """Run a code example, with MJ and with a fake implementation."""
  code_str = str(code)
  globals_dict = dict(**SUT_MJ)
  Globals.reset()
  compare_results = True
  try:
    exec(code_str, globals_dict)
    _result_mj = globals_dict["_result"]
  except OverflowError:
    _result_mj = "overflow"
    print("WARNING: overflow in MJ computation")
  except Exception as e:
    print("Mini-JAX execution failed on\n{}\nwith traceback\n{}".format(
      code,
      traceback.format_exc()
    ))
    raise

  globals_dict = dict(**SUT_FAKE)
  try:
    exec(code_str, globals_dict)
    _result_fake = globals_dict["_result"]
  except NumericalDiffDiscontinuity:
    print("WARNING: numerical diff discontinuity in Fake computation")
    compare_results = False
  except OverflowError:
    print("WARNING: overflow in Fake computation")
    _result_fake = "overflow"
  except Exception as e:
    print("Fake JAX execution failed on\n{}\nwith traceback\n{}".format(
      code,
      traceback.format_exc()
    ))
    raise

  if compare_results:
    if _result_fake == "overflow" or _result_mj == "overflow":
      assert _result_mj == _result_fake
    else:
      assert np.allclose(_result_mj, _result_fake, rtol=0.1), \
        "Value mismatch (MJ vs Fake):\n{}\n  vs\n{}\non\n{}".format(
          _result_mj, _result_fake, code)


###
class Environment(object):
  """What vars and funcs can be used in each hole

  We keep an immutable representation as tuples (cons-cells).
  """

  def __init__(self, vars, funcs, depth: int):
    self._vars = vars  # First is the last declared
    self._funcs = funcs
    self.depth = depth

  def add_var(self, vname: str) -> 'Environment':
    return Environment((vname, self._vars), self._funcs, self.depth)

  def add_vars(self, vnames: List[str], depth_incr: int = 0) -> 'Environment':
    if depth_incr:
      env = Environment(self._vars, self._funcs, self.depth + depth_incr)
    else:
      env = self
    return functools.reduce(lambda acc, a: acc.add_var(a), vnames, env)

  def add_func(self, fname: str, nr_args: int, nr_res: int) -> 'Environment':
    return Environment(self._vars, ((fname, nr_args, nr_res), self._funcs),
                       self.depth)

  @staticmethod
  def empty():
    return Environment([], [], 0)

  def _get_elements(self, where):
    res = []
    while where:
      res.append(where[0])
      where = where[1]
    return res

  def get_vars(self):
    return self._get_elements(self._vars)

  def get_funcs(self):
    return self._get_elements(self._funcs)

  def difference(self, other: 'Environment'):
    """Get an environment with only stuff new since 'other'"""
    def diff_rec(where, sentinel):
      if where is sentinel:
        return ()
      else:
        assert where, "Sentinel not found"
        return (where[0], diff_rec(where[1], sentinel))
    return Environment(diff_rec(self._vars, other._vars),
                       diff_rec(self._funcs, other._funcs),
                       self.depth)


class GlobalCounters(object):
  coverage_counters = defaultdict(int)

  @staticmethod
  def accum_counters(counters):
    GlobalCounters.coverage_counters["examples"] += 1
    for k, v in counters.items():
      GlobalCounters.coverage_counters[k] += v


class JaxExampleStrategy(st.SearchStrategy):
  def __init__(self, max_body_length=3, max_body_depth=3):
    super(JaxExampleStrategy, self).__init__()
    self.counter = itertools.count()
    self.max_body_length = max_body_length
    self.max_body_depth = max_body_depth

  def reset(self):
    self.counter = itertools.count()

  def new_jax_name(self, prefix: str) -> str:
    return "{}{}".format(prefix, next(self.counter))

  def do_draw(self, data):
    """The entry point, draws a code example."""
    self.reset()
    return self.draw_body(data, 1, Environment.empty())

  def draw_body(self, data, body_nr_res: int, env: Environment) -> PrettyPrint:
    # Get the structure of the body first
    # Kinds of body statements; first should be simplest, to help shrinking
    decl_strategies = [
      # A var initialized from an atomic expression
      st.fixed_dictionaries(dict(kind=st.just("var_decl_atom"))),
      # A var initialized from a binary expression
      st.fixed_dictionaries(dict(kind=st.just("var_decl_op"))),
      # One or more vars initialized from function call
      st.fixed_dictionaries(dict(kind=st.just("var_decl_func_call"),
                                 transform=st.sampled_from(
                                   ["direct", "jit", "jvp", "grad"]))),
    ]
    if env.depth <= self.max_body_depth:
      # These are the declarations with nested bodies
      decl_strategies += [
        # Function declaration
        st.fixed_dictionaries(
          dict(kind=st.just("func_decl"),
               nr_args=st.integers(min_value=1, max_value=2),
               nr_res=st.integers(min_value=1, max_value=2))),
        # One or more vars initialized from cond_ge
        st.fixed_dictionaries(dict(kind=st.just("var_decl_cond"),
                                   nr_res=st.integers(min_value=1, max_value=2),
                                   nr_true_ops=st.integers(min_value=1,
                                                           max_value=2),
                                   nr_false_ops=st.integers(min_value=1,
                                                            max_value=2)))
      ]
    # Pick the body structure
    body_structure = st.lists(
      st.one_of(decl_strategies),
      min_size=2,
      max_size=self.max_body_length).do_draw(data)

    # Now pick the expressions, based on a running Environment
    start_env = env
    body = []
    for elem in body_structure:
      if elem["kind"] == "var_decl_atom":
        var_names = [self.new_jax_name("v")]
        body.append(self.pp_vars_decl(var_names,
                                      self.draw_expr_atom(data, env)))
        env = env.add_vars(var_names)
      elif elem["kind"] == "var_decl_op":
        var_names = [self.new_jax_name("v")]
        body.append(self.pp_vars_decl(var_names, self.draw_expr_op(data, env)))
        env = env.add_vars(var_names)
      elif elem["kind"] == "var_decl_func_call":
        # Maybe we don't yet have functions
        funcs = env.get_funcs()
        if funcs:
          func_name, nr_args, nr_res = st.sampled_from(funcs).do_draw(data)
          transform = elem["transform"]
          if transform == "grad" and nr_res > 1:
            transform = "direct"
          if transform == "direct":
            pass
          elif transform == "jit":
            func_name = "jit({})".format(func_name)
          elif transform == "grad":
            func_name = "grad({})".format(func_name)
            nr_res = nr_args
          elif transform == "jvp":
            func_name = "jvp({})".format(func_name)
            nr_args = 2 * nr_args
            nr_res = 2 * nr_res
          else:
            assert False

          var_names = [self.new_jax_name("v") for _ in range(nr_res)]
          args = [self.draw_expr_atom(data, env) for _ in range(nr_args)]
          body.append(
            self.pp_vars_decl(var_names, self.pp_func_call(func_name, args)))
          env = env.add_vars(var_names)

      elif elem["kind"] == "func_decl":
        nr_args, nr_res = elem["nr_args"], elem["nr_res"]
        func_name, arg_names, func_body = self.draw_func_decl(data, nr_args,
                                                              nr_res, env)
        body.append(self.pp_func_decl(func_name, arg_names, func_body))
        env = env.add_func(func_name, nr_args, nr_res)

      elif elem["kind"] == "var_decl_cond":
        nr_res, nr_true_ops, nr_false_ops = [
          elem[f] for f in ["nr_res", "nr_true_ops", "nr_false_ops"]]

        var_names = [self.new_jax_name("v") for _ in range(elem["nr_res"])]
        true_func_name, true_arg_names, true_func_body = (
          self.draw_func_decl(data, nr_true_ops, nr_res, env))
        false_func_name, false_arg_names, false_func_body = (
          self.draw_func_decl(data, nr_false_ops, nr_res, env))
        true_func_pp = self.pp_func_decl(true_func_name, true_arg_names,
                                         true_func_body)
        false_func_pp = self.pp_func_decl(false_func_name, false_arg_names,
                                          false_func_body)

        pred = self.draw_expr_atom(data, env)
        true_args = [self.draw_expr_atom(data, env) for _ in range(nr_true_ops)]
        false_args = [self.draw_expr_atom(data, env) for _ in
                      range(nr_false_ops)]

        true_ops_pp = (
              pp_str("(") >> pp_list(true_args, hsep=", ") >> pp_str(",)"))
        false_ops_pp = (
            pp_str("(") >> pp_list(false_args, hsep=", ") >> pp_str(",)"))
        cond_args = pp_list([pred, true_func_name, true_ops_pp,
                             false_func_name, false_ops_pp],
                            hsep=", ")
        cond_pp = (true_func_pp + false_func_pp +
                   (self.pp_vars_decl(var_names, pp_str(
                     "cond_ge(") >> cond_args >> pp_str(")"))))
        body.append(cond_pp)
        env = env.add_vars(var_names)
        env = env.add_func(true_func_name, nr_true_ops, nr_res)
        env = env.add_func(false_func_name, nr_false_ops, nr_res)
      else:
        assert False

    # Now call all the functions we defined (to ensure that we use them)
    this_body_env = env.difference(start_env)
    this_body_results = this_body_env.get_vars()
    for f, nr_args, nr_res in this_body_env.get_funcs():
      var_names = [self.new_jax_name("v") for _ in range(nr_res)]
      args = [self.draw_expr_atom(data, env) for _ in range(nr_args)]
      body.append(self.pp_vars_decl(var_names, self.pp_func_call(f, args)))
      env = env.add_vars(var_names)
      this_body_results.extend(var_names)

    decls_pp = pp_list(body, vertical=True)

    # Add a _result = sum(all vars)

    # If we do not have enough results for the body, replicate them
    orig_results = this_body_results
    # It is possible that we got nothing, e.g., if we had to generate func_call
    # but we had no functions
    if not this_body_results:
      this_body_results = [pp_str("1.")] * body_nr_res
    while len(this_body_results) < body_nr_res:
      this_body_results += orig_results

    vars_per_result = len(this_body_results) // body_nr_res
    results = []
    for res_idx in range(body_nr_res):
      results.append(pp_str("sum(") >>
                     pp_list(this_body_results[res_idx * vars_per_result:(
                                                                         res_idx + 1) * vars_per_result],
                             hsep=", ") >> pp_str(")"))
    results_pp = pp_list(results, hsep=", ")
    if env.depth == 0:
      put_result = "_result = "
    else:
      put_result = "return "
    return decls_pp + (pp_str(put_result) >> results_pp)

  def draw_expr_atom(self, data, env: Environment) -> PrettyPrint:
    strategies = [
      # Constants
      st.integers(min_value=-5, max_value=5).map(lambda c: str(float(c))),
    ]
    env_vars = env.get_vars()
    if env_vars:
      strategies += [
        # Variables
        st.sampled_from(env_vars)
      ]
    x = st.sampled_from(strategies).do_draw(data)
    y = x.do_draw(data)  # Why do we have to draw twice?
    return pp_str(y)

  def draw_expr_op(self, data, env: Environment) -> PrettyPrint:
    op = st.sampled_from(["+", "-", "*", "**"]).do_draw(data)
    if op == "**":
      pow = st.integers(min_value=1, max_value=5).map(str).do_draw(data)
      return pp_list([self.draw_expr_atom(data, env), "**", pow])
    else:
      return pp_list([self.draw_expr_atom(data, env), op,
                      self.draw_expr_atom(data, env)])

  def draw_func_decl(self, data, nr_args, nr_res, env):
    func_name = self.new_jax_name("f")
    arg_names = [self.new_jax_name("v") for _ in range(nr_args)]
    func_env = env.add_vars(arg_names, depth_incr=1)
    data.start_example("func@{}".format(env.depth))
    func_body = self.draw_body(data, nr_res, func_env)
    data.stop_example()
    return func_name, arg_names, func_body

  def pp_vars_decl(self, var_names, var_init: PrettyPrint) -> PrettyPrint:
    return pp_list(var_names, hsep=", ") >> pp_str(" = ") >> var_init

  def pp_func_decl(self, func_name, arg_names,
                   func_body: PrettyPrint) -> PrettyPrint:
    header = (
        pp_str("def {}(".format(func_name)) >>
        pp_list(arg_names, hsep=", ") >> pp_str("):"))
    decls_pp = func_body  # Leaves the result in _result
    return header + decls_pp.indent(2)

  def pp_func_call(self, func_name, func_args) -> PrettyPrint:
    return (pp_str(func_name) >> pp_str("(") >>
            pp_list(func_args, hsep=", ") >> pp_str(")"))

  def calc_has_reusable_values(self, recur):
    return True


@cacheable
@defines_strategy_with_reusable_values
def jax_examples(**kwargs):
  return JaxExampleStrategy(**kwargs)


class JaxGenTest(jtu.JaxTestCase):

  @given(body=jax_examples(max_body_depth=1))
  @settings(deadline=None,
            max_examples=HYPOTHESIS_EXAMPLES,
            verbosity=HYPOTHESIS_VERBOSITY
            )
  def test_tree(self, body: PrettyPrint):
    if not USE_HYPOTHESIS:
      self.skipTest("Must pass JAX_HYPOTHESIS_EXAMPLES=NNN to enable this test")
    check_code_example(body)

  def test_fake_jvp_discountinuity(self):
    """Test the fake JVP, at discontinuity"""
    code = """
def f0(v2):
  return cond_ge(v2, lambda tv: 1., (0.,), lambda fv: 0., (0.,))
v17, v18 = jvp(f0)(0.0, 3.0)
_result = (v18, v17)
"""
    check_code_example(code)

  def test_fake_jvp_discountinuity_2(self):
    """Test the fake JVP, at discontinuity"""
    code = """
def f0(v2):
  return cond_ge(0. - v2, lambda tv: 1., (0.,), lambda fv: 0., (0.,))
v17, v18 = jvp(f0)(0.0, 3.0)
_result = (v18, v17)
"""
    check_code_example(code)


  def test_repro(self):
    """Bug found with hypothesis."""
    code = """
def f0(v1):
  def f5(v6):
    return v6, v6
  def f8(v9):
    return 0., 0.
  v3, v4 = cond_ge(5.0, f5, (v1,), f8, (0.,))
  return v3
v16 = grad(f0)(0.0)
_result = v16
"""
    check_code_example(code)


if __name__ == '__main__':
  absltest.main()
