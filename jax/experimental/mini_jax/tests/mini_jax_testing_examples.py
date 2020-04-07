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

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import inspect
from typing import Any, Callable, Dict, Iterator, List, Optional, Tuple, Set, Sequence
import sys

from jax import test_util as jtu
from jax.experimental import mini_jax as mj
from jax.experimental.mini_jax.mini_jax import (
  const_like
)
import numpy as np

"""Defines examples that can be used to test transformations.

The union of these examples should cover all primitives, and all
interesting control flow. There are a bunch of tests that 
are called xxx_all_examples() that use `iterate_examples` to cover
all defined examples. In order to see test coverage, run 

  rm -rf htmlcov && \
      coverage run --branch --source jax/experimental/mini_jax \
        -m pytest --durations=10 -r a --verbosity 1 -k "all_examples" \
        jax/experimental/mini_jax/tests/ && \
      coverage html && coverage report && open htmlcov/index.html

"""
class Example(object):
  """Abstract superclass for examples."""
  NR_ARGS = 1  # By default 1 argument

  # Argument constraints
  CONSTRAINT_SCALAR = "scalar"
  CONSTRAINT_RANK_AT_LEAST_1 = "rank_at_least_1"
  CONSTRAINT_RANK_AT_LEAST_2 = "rank_at_least_2"

  ARG_CONSTRAINTS: Optional[Tuple[Optional[str], ...]] = None  # Constraints for the arguments

  # Can define extra args, in addition to the one generated by `arange_args`.
  # Each tuple element is a pair of a name and a tuple of arguments.
  # If none of the arguments have constraints, we will also generate a tensor version.
  EXTRA_ARGS: Tuple[Tuple[str, Tuple], ...] = ()

  @classmethod
  def name(cls):
    return cls.__name__

  def args_constraints(self):
    """Returns a tuple of argument constraints, one for each argument"""
    cons = self.ARG_CONSTRAINTS if isinstance(self.ARG_CONSTRAINTS, tuple) else (self.ARG_CONSTRAINTS,) * self.NR_ARGS
    assert len(cons) == self.NR_ARGS
    return cons

  def _default_args(self):
    """A default set of arguments, based on arange."""
    def _make_arg(i: int, cons: Optional[str] = None):
      if cons is None or cons == Example.CONSTRAINT_SCALAR:
        return float(i)
      elif cons == Example.CONSTRAINT_RANK_AT_LEAST_1:
        return np.array([float(i) + 1.])
      elif cons == Example.CONSTRAINT_RANK_AT_LEAST_2:
        return np.full((2, 2), float(i) + 1.)
      else:
        assert False, f"{cons}"
    cons = self.args_constraints()
    return tuple([_make_arg(i, a_cons)
                  for i, a_cons in zip(range(self.NR_ARGS), cons)])

  def generate_args(self) -> Sequence[Tuple[str, Tuple]]:
    """Generates a number of sets of arguments
    Returns: a tuple with each element a pair of a name and a
      sequence of arguments
    """
    # Type check EXTRA_ARGS, it is easy to make mistakes
    def _typecheck_one_EXTRA_ARGS(a):
      assert isinstance(a, tuple), f"{a}"
      assert len(a) == 2, f"{a}"
      assert isinstance(a[0], str), f"{a}"
      assert isinstance(a[1], tuple), f"{a}"
    [_typecheck_one_EXTRA_ARGS(a) for a in self.EXTRA_ARGS]
    res = [("", self._default_args())] + list(self.EXTRA_ARGS)
    # Now tensorize these args, each scalar that can be a tensor
    # is lifted to shape (2, 3)
    tensor_res: List[Tuple[str, Tuple]] = []
    def _tensorize(name_and_args: Tuple[str, Tuple]) -> None:
      name, args = name_and_args
      if all(a is None and not np.shape(a)
             for a in self.args_constraints()):  # Scalars with constraints
        new_name = f"{name}_tensor"
        new_args = [a * np.ones((2, 3)) for a in args]
        tensor_res.append((new_name, tuple(new_args)))
    for r in res: _tensorize(r)
    return tuple(res + tensor_res)

  def apply(self, *_):
    """Applies the function and returns the result."""
    raise NotImplementedError(f"Must implement apply for {type(self)}")

  def get_lambda(self):
    """Get a lambda that can be used for transformations."""
    def func(*args):
      assert len(args) == self.NR_ARGS
      cons = self.args_constraints()
      assert all(a_cons != Example.CONSTRAINT_SCALAR or np.shape(a) == ()
                 for a, a_cons in zip(args, cons))
      return self.apply(*args)
    return func


class ExampleInstance(object):
  """Objects returned by iterate_examples."""
  def __init__(self, name, example, func, args):
    self.name = name
    self._example = example
    self.args = args
    self.func = func

  def args_constraints(self):
    return self._example.args_constraints()


def iterate_examples() -> Iterator[ExampleInstance]:
  """Iterator over Example instances"""
  clsmembers = inspect.getmembers(sys.modules[__name__], inspect.isclass)
  # Make a bunch of instances
  examples = [exCls() for _, exCls in clsmembers
              if issubclass(exCls, Example) and exCls not in [Example, Example2]]

  for ex in examples:
    func = ex.get_lambda()
    for args_name, args in ex.generate_args():
      yield ExampleInstance(ex.name()+args_name, ex, func, args)


class Example2(Example):
  """An example with 2 arguments."""
  NR_ARGS = 2

class Add0(Example2):
  """A simple addition"""
  def apply(self, x, y):
    return x + y

class AddConstant(Example):
  def apply(self, x):
    return x + const_like(5., x)

class Arithmetic(Example2):
  """A bunch of arithmetic"""
  def apply(self, x, y):
    return x * const_like(3., x) + y - x + x ** 3 - y ** 1 + x ** 0

class MultipleResults(Example2):
  """Returns two results: first depends on both args, second only on second."""
  def apply(self, x, y):
    return x * y, y * const_like(3., y)

class InnerJit0(Example):
  def apply(self, x):
    def func(y):
      return y * const_like(2., y)
    return mj.jit(func)(x + const_like(2., x))

class Conditional0(Example2):
  def apply(self, x, y):
      return mj.Ops.cond_ge(x - 2.,
                            lambda x_true: x_true + y,
                            lambda x_false0: x_false0 + 4.,
                            (x + 5.,))
  ARG_CONSTRAINTS = (Example.CONSTRAINT_SCALAR, None)
  EXTRA_ARGS = (
    ("ex1", (2., 4.)),
    ("ex2", (-2., 4.))
  )

class ConditionalTupleArg(Example):
  """One of the branches take a tuple of arguments."""
  def apply(self, x):
    return mj.Ops.cond_ge(x - 2.,
                          lambda x_true0, x_true1: x_true0 + 3.,
                          lambda x_false0, x_false1: x_false0 + x_false1 * 3.,
                          (x + 4., x + 5.))

  ARG_CONSTRAINTS = (Example.CONSTRAINT_SCALAR,)
  EXTRA_ARGS = (
    ("ex1", (2.,)),
    ("ex2", (-2.,))
  )

class ConditionalTupleRes(Example2):
  """The branches return tuples"""
  def apply(self, x, y):
    return mj.Ops.cond_ge(x - 2.,
                          lambda x_true0, x_true1: (x_true0 + y, y),
                          lambda x_false0, x_false1: (x_false0 + x_false1 * 3., y + 6.),
                          (x + 4., x + 5.))

  ARG_CONSTRAINTS = (Example.CONSTRAINT_SCALAR, None)
  EXTRA_ARGS = (
    ("ex1", (2., 4.)),
    ("ex2", (-2., 4.))
  )

class PowerOp3(Example):
  def apply(self, x):
    return mj.customPowerOp.invoke_single(x, exp=3)


class CallbackOp(Example):
  def apply(self, x):
    def acc_callback_fun(*args, transforms=()):
      print("callback = ", tuple([*args, transforms]))

    r, = mj.callback(acc_callback_fun)(x)
    return r


class LibraryTest(jtu.JaxTestCase):

  def test_one(self):
    add0 = Add0().get_lambda()
    res = add0(5., 7.)
    self.assertEqual(12., res)

    self.assertMultiLineStrippedEqual("""
{lambda v0 v1.
  # v0: float, v1: float
  n0 = add v0 v1
  in n0}
        """, str(mj.trace(add0)(3., 5.).pp()))

  def test_run_all_examples(self):
    """Invoke all examples, to make sure they do not crash"""
    for ex in iterate_examples():
      #if ex.name != "PowerOp3_tensor": continue
      print(f"Invoking {ex.name}")
      ex.func(*ex.args)
