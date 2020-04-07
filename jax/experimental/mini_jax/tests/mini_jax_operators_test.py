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

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import re
from typing import Callable, Dict, Sequence, Tuple

from jax import test_util as jtu
from jax.experimental import mini_jax as mj
from jax.config import config
from jax.experimental.mini_jax.mini_jax import (
  Expr, ExprType, Operator, Function, Tracer,
  Value, PrettyPrint, pp_str
)
import numpy as np

config.parse_flags_with_absl()
FLAGS = config.FLAGS


# A multi-output custom op
class _TwiceThrice(mj.CustomOperator):
  """A custom op returning (x * 2., x * 3.). """
  def __init__(self):
    super().__init__("2n3")

  def eval_concrete(self, params: Dict, args: Sequence[Value]) -> Sequence[Value]:
    assert isinstance(args[0], float)
    return (args[0] * 2., args[0] * 3.)

  def type_check(self, params: Dict, args_t: Sequence[ExprType]) -> Sequence[ExprType]:
    if not(len(args_t) == 1 and args_t[0].dtype is float):
      raise TypeError(f"Unexpected argument types for {self}: {args_t}")
    return (args_t[0], args_t[0])

  def compile_assigned(self, params: Dict, args_s: Sequence[str],
                       e_types: Sequence[ExprType],
                       name: str) -> PrettyPrint:
    return pp_str(f"{name} = {args_s[0]} ** {params['exp']}")

  def eval_jvp(self, params: Dict, args_v: Sequence[Value], args_tan: Sequence[Value]) -> Sequence[Value]:
    arg = args_v[0]
    # Return 4 values, the primal(s) and the tangent(s)
    return (*self.invoke(arg), args_tan[0] * 2., args_tan[0] * 3.)

  def eval_vjp(self, params: Dict, args: Sequence['Expr'], out_adj: Sequence[Value],
               eval_std_expr: Callable[['Expr'], Value]) -> Sequence[Value]:
    return (out_adj[0] * 2. + out_adj[1] * 3., )

  def eval_count_flops(self, params: Dict, args: Sequence['Expr'],
                       eval_std_expr: Callable[['Expr'], Value]) -> Value:
    return 2.

twice_thrice_op = _TwiceThrice()

# A custom op whose compilation lifts additional parameters outside

class CustomPowerOpTest(jtu.JaxTestCase):

  def test_eval(self):
    def fun(x):
      return mj.customPowerOp.invoke_single(x, exp=3)

    self.assertAllClose(8., fun(2.), check_dtypes=True)

  def test_eval_multi(self):
    def fun(x):
      y, z = twice_thrice_op.invoke(x)
      return y + z

    self.assertAllClose(10., fun(2.), check_dtypes=True)

  def test_trace(self):
    def fun(x):
      return mj.customPowerOp.invoke_single(x, exp=3)

    self.assertMultiLineStrippedEqual("""
{lambda v0.
  # v0: float
  n0 = Op[cpow][ exp=3 ] v0
  n1 = proj[ idx=0 ] n0
  in n1}
          """, str(mj.trace(fun)(4.).pp()))

  def test_trace_multi(self):
    def fun(x):
      y, z = twice_thrice_op.invoke(x)
      return y + z

    self.assertMultiLineStrippedEqual("""
{lambda v0.
  # v0: float
  n0 = Op[2n3] v0
  n1 = proj[ idx=0 ] n0
  n2 = proj[ idx=1 ] n0
  n3 = add n1 n2
  in n3}
          """, str(mj.trace(fun)(4.).pp()))

  def test_invoke_error(self):
    with self.assertRaisesRegexp(TypeError,
                                 re.escape("Op[cpow] expects parameters {'exp'}")):
      mj.customPowerOp.invoke(1., bad_params=2)

    with self.assertRaisesRegexp(TypeError,
                                 re.escape("Unexpected argument types for Op[cpow]: (float, float)")):
      mj.customPowerOp.invoke(1., 2., exp=2)


  def test_jit(self):
    def fun(x):
      return mj.customPowerOp.invoke_single(x, exp=3)

    self.assertAllClose(8., mj.jit(fun)(2.), check_dtypes=True)

  def test_jvp(self):
    def func(x):
      return mj.customPowerOp.invoke_single(x, exp=3)

    # Check that the grad uses the custom op
    self.assertMultiLineStrippedEqual("""
{lambda v0 v1.
  # v0: float, v1: float
  n0 = Op[cpow][ exp=3 ] v0
  n1 = proj[ idx=0 ] n0
  n2 = Op[cpow][ exp=2 ] v0
  n3 = proj[ idx=0 ] n2
  n4 = mul 3.0 n3
  n5 = mul n4 v1
  in (n1 n5,)}
          """, str(mj.trace(mj.jvp(func))(2., 5.).pp()))

    self.assertEqual((2. ** 3, 3. * 2. ** 2 * 5.), mj.jvp(func)(2., 5.))

    # Special cases for exponent 0 and 1
    self.assertMultiLineStrippedEqual("""
{lambda v0 v1.
  # v0: float, v1: float
  in (1.0 0.0,)}
              """, str(mj.trace(mj.jvp(lambda x: mj.customPowerOp.invoke_single(x, exp=0)))(2., 5.).pp()))
    self.assertMultiLineStrippedEqual("""
{lambda v0 v1.
  # v0: float, v1: float
  in (v0 v1,)}
          """, str(mj.trace(mj.jvp(lambda x: mj.customPowerOp.invoke_single(x, exp=1)))(2., 5.).pp()))

  def test_jvp_multi(self):
    def func(x):
      y, z = twice_thrice_op.invoke(x)
      return y + z

    self.assertMultiLineStrippedEqual("""
{lambda v0 v1.
  # v0: float, v1: float
  n0 = Op[2n3] v0
  n1 = proj[ idx=0 ] n0
  n2 = proj[ idx=1 ] n0
  n3 = add n1 n2
  n4 = mul v1 2.0
  n5 = mul v1 3.0
  n6 = add n4 n5
  in (n3 n6,)}
            """, str(mj.trace(mj.jvp(func))(2., 5.).pp()))

    self.assertEqual((2. * 5., 5. * 5), mj.jvp(func)(2., 5.))

  def test_grad(self):
      def func(x):
        return mj.customPowerOp.invoke_single(x, exp=3)

      # Check that the grad uses the custom op
      self.assertMultiLineStrippedEqual("""
{lambda v0.
  # v0: float
  n0 = Op[cpow][ exp=2 ] v0
  n1 = proj[ idx=0 ] n0
  n2 = mul 3.0 n1
  in n2}
            """, str(mj.trace(mj.grad(func))(2.).pp()))

      self.assertEqual(3. * 2. ** 2, mj.grad(func)(2.))

      # Special cases for exponent 0 and 1
      self.assertMultiLineStrippedEqual("""
  {lambda v0.
    # v0: float
    in 0.0}
                """, str(mj.trace(mj.grad(lambda x: mj.customPowerOp.invoke_single(x, exp=0)))(2.).pp()))
      self.assertMultiLineStrippedEqual("""
  {lambda v0.
    # v0: float
    in 1.0}
            """, str(mj.trace(mj.grad(lambda x: mj.customPowerOp.invoke_single(x, exp=1)))(2.).pp()))

  def test_grad_twice(self):
    def func(x):
      return mj.customPowerOp.invoke_single(x, exp=3)

    # Check that the grad uses the custom op
    self.assertMultiLineStrippedEqual("""
{lambda v0.
  # v0: float
  n0 = mul 2.0 v0
  n1 = mul n0 3.0
  in n1}
          """, str(mj.trace(mj.grad(mj.grad(func)))(2.).pp()))

    self.assertEqual(3. * 2. * 2., mj.grad(mj.grad(func))(2.))


  def test_grad_multi(self):
    def func(x):
      y, z = twice_thrice_op.invoke(x)
      return y + z

      # Check that the grad uses the custom op
    self.assertMultiLineStrippedEqual("""
{lambda v0.
  # v0: float
  in 5.0}
            """, str(mj.trace(mj.grad(func))(2.).pp()))

    self.assertEqual(5., mj.grad(func)(2.))

  def test_flops(self):
    def func(x):
      return mj.customPowerOp.invoke_single(x + 1., exp=3)

    # Check that the grad uses the custom op
    self.assertEqual(1. + 2., mj.count_flops(func)(2.))

  def test_flops_multi(self):
    def func(x):
      y, z = twice_thrice_op.invoke(x)
      return y + z

    # Check that the grad uses the custom op
    self.assertEqual(1. + 2., mj.count_flops(func)(2.))
