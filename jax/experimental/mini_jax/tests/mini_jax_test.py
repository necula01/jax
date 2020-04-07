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

import math
import re

from jax import api
from jax import lax
from jax import test_util as jtu
from jax.experimental import mini_jax as mj
from jax.experimental.mini_jax.mini_jax import Cache
from jax.experimental.mini_jax.tests import mini_jax_testing_examples as testex

from jax.config import config
import numpy as np
from typing import Any, Dict, Callable, Tuple, Sequence, List, Optional, Union

config.parse_flags_with_absl()
FLAGS = config.FLAGS

class MiniJaxTest(jtu.JaxTestCase):


  def test_trace_all_examples(self):
    """Trace all examples, to make sure they do not crash"""
    for ex in testex.iterate_examples():
      print(f"Tracing {ex.name}")
      mj.trace(ex.func)(*ex.args)

  def test_jit_all_examples(self):
    """Trace all examples, to make sure they return the same value on jit"""
    for ex in testex.iterate_examples():
      #if ex.name != "SumDim0": continue  #TODO
      print(f"JITing {ex.name}")
      args = ex.args
      res = ex.func(*args)
      jit_res = mj.jit(ex.func)(*args)
      self.assertAllClose(res, jit_res, check_dtypes=True)

  def test_trace_const_eval(self):
    """Constants are evaluated by Python."""

    def func(x):
      return 1. + 2.

    func_tr = mj.trace(func)
    res = func_tr(5.)
    self.assertMultiLineStrippedEqual("""
{lambda v0.
  # v0: float
  in 3.0}
    """, str(res.pp()))

  def test_trace_sharing(self):
    """Ensure that we maintain sharing of Expr (no exp blowout)."""

    def func(x):
      y = x + x
      z = y * y
      w = z * y
      return w + w

    func_tr = mj.trace(func)
    res = func_tr(5.)
    self.assertMultiLineStrippedEqual("""
{lambda v0.
  # v0: float
  n0 = add v0 v0
  n1 = mul n0 n0
  n2 = mul n1 n0
  n3 = add n2 n2
  in n3}
    """, str(res.pp()))

  def test_trace_captured_const(self):
    """Capture a constant from outside the scope."""
    outside = 1.

    def func(x, y):
      return x + y + outside

    func_tr = mj.trace(func)
    res = func_tr(5., 6.)
    self.assertMultiLineStrippedEqual("""
{lambda v0 v1.
  # v0: float, v1: float
  n0 = add v0 v1
  n1 = add n0 1.0
  in n1}
    """, str(res.pp()))

  def test_jit_const(self):
    """JIT a function returning a constant"""
    outside = 10.

    def func(x):
      x += 1.

      def inner(z):
        return outside

      return mj.jit(inner)(x)

    func_tr = mj.trace(func)(5.)
    self.assertMultiLineStrippedEqual("""
{lambda v0.
  # v0: float
  n0 = add v0 1.0
  n1 = jit_call[ func={lambda v1.
                        # v1: float
                        in 10.0} ] n0
  in n1}
    """, str(func_tr.pp()))
    # Run the jit
    self.assertEqual(10., func(5.))

  def test_jit_captured_const(self):
    outside = 10.

    def func(x, y):
      x += 1.

      def inner(z):
        return x + z + outside

      return mj.jit(inner)(x)

    func_tr = mj.trace(func)(5., 6.)
    self.assertMultiLineStrippedEqual("""
{lambda v0 v1.
  # v0: float, v1: float
  n0 = add v0 1.0
  n1 = jit_call[ func={lambda v2 v3.
                        # v2: float, v3: float
                        n0 = add v3 v2
                        n1 = add n0 10.0
                        in n1} ] n0 n0
  in n1}
    """, str(func_tr.pp()))
    # Run the jit
    self.assertEqual(22., func(5., 6.))

  def test_jit_captured_computation(self):
    """Should not capture computations from shallower scope depths"""

    def fun1(x1):
      z11 = x1 * 11.  # We multiply with 11. to mean using x1 in fun1

      def fun2(x2):
        return z11 + x1 * 12. + x2 * 22.

      return mj.jit(fun2)(x1)

    self.assertMultiLineStrippedEqual("""
{lambda v0.
  # v0: float
  n0 = jit_call[ func={lambda v1.
                        # v1: float
                        n0 = mul v1 11.0
                        n1 = jit_call[ func={lambda v2 v1 v3.
                                              # v2: float, v1: float, v3: float
                                              n0 = mul v1 12.0
                                              n1 = add v3 n0
                                              n2 = mul v2 22.0
                                              n3 = add n1 n2
                                              in n3} ] v1 v1 n0
                        in n1} ] v0
  in n0}
    """, str(mj.trace(mj.jit(fun1))(5.)))

  def test_jit_captured_computation_2(self):
    """Should not capture computations from shallower scope depths"""

    def fun1(x1):
      z11 = x1 * 11.  # We multiply with 11. to mean using x1 in fun1

      def fun2(x2):
        z22 = z11 + x1 * 12. + x2 * 22.

        def fun3(x3):
          # Capture twice x2, ensure that we only close over it once
          z33 = z11 * 13. + z22 * 23. + x2 * 23. + x2 * 23. + x1 * x1
          return z33

        return mj.jit(fun3)(x2)

      return mj.jit(fun2)(x1)

    fun1_jit = mj.jit(fun1)
    self.assertMultiLineStrippedEqual("""
{lambda v0.
  # v0: float
  n0 = jit_call[ func={lambda v1.
                        # v1: float
                        n0 = mul v1 11.0
                        n1 = jit_call[ func={lambda v2 v1 v3 v7.
                                              # v2: float, v1: float, v3: float, v7: float
                                              n0 = mul v1 12.0
                                              n1 = add v3 n0
                                              n2 = mul v2 22.0
                                              n3 = add n1 n2
                                              n4 = jit_call[ func={lambda v4 v1 v2 v5 v6.
                                                                    # v4: float, v1: float, v2: float, v5: float, v6: float
                                                                    n0 = mul v5 13.0
                                                                    n1 = mul v6 23.0
                                                                    n2 = add n0 n1
                                                                    n3 = mul v2 23.0
                                                                    n4 = add n2 n3
                                                                    n5 = mul v2 23.0
                                                                    n6 = add n4 n5
                                                                    n7 = mul v1 v1
                                                                    n8 = add n6 n7
                                                                    in n8} ] v2 v1 v2 v7 n3
                                              in n4} ] v1 v1 n0 n0
                        in n1} ] v0
  in n0}

    """, str(mj.trace(fun1_jit)(5.)))

  def test_jit_return_tuple(self):
    """JIT function returning a tuple."""

    def func(x):
      return (x + 1., x * 2., 5.)

    func_jit = mj.jit(func)
    func_tr = mj.trace(func_jit)(5.)
    self.assertMultiLineStrippedEqual("""
{lambda v0.
  # v0: float
  n0 = jit_call[ func={lambda v1.
                        # v1: float
                        n0 = add v1 1.0
                        n1 = mul v1 2.0
                        in (n0 n1 5.0,)} ] v0
  n1 = proj[ idx=0 ] n0
  n2 = proj[ idx=1 ] n0
  n3 = proj[ idx=2 ] n0
  in (n1 n2 n3,)}
    """, str(func_tr.pp()))
    self.assertEqual((6., 10., 5.), func(5.))
    # Call the jitted function
    self.assertEqual((6., 10., 5.), func_jit(5.))

  def test_jit_nested(self):
    outside = 1.

    def func(x, y):
      y = x + 10.

      def inner(z):
        return z + x + outside

      return mj.jit(inner)(y)

    func_tr = mj.trace(mj.jit(func))(5., 6.)
    self.assertMultiLineStrippedEqual("""
{lambda v0 v1.
  # v0: float, v1: float
  n0 = jit_call[ func={lambda v2 v3.
                        # v2: float, v3: float
                        n0 = add v2 10.0
                        n1 = jit_call[ func={lambda v4 v2.
                                              # v4: float, v2: float
                                              n0 = add v4 v2
                                              n1 = add n0 1.0
                                              in n1} ] n0 v2
                        in n1} ] v0 v1
  in n0}
    """, str(func_tr.pp()))

    self.assertEqual(21., func(5., 6.))
    self.assertEqual(21., mj.jit(func)(5., 6.))

  def test_jit_two_calls(self):
    enable_jit = False

    def func(x):
      x += 1.

      def inner(z):
        return z + x

      inner_jitted = mj.jit(inner) if enable_jit else inner
      return inner_jitted(x) + inner_jitted(3.)

    # Without jit
    enable_jit = False
    self.assertEqual(21., func(5.))
    func_tr = mj.trace(func)(5.)
    self.assertMultiLineStrippedEqual("""
{lambda v0.
  # v0: float
  n0 = add v0 1.0
  n1 = add n0 n0
  n2 = add 3.0 n0
  n3 = add n1 n2
  in n3}
    """, str(func_tr.pp()))

    # With jit
    enable_jit = True
    func_tr = mj.trace(func, cache=False)(5.)
    self.assertMultiLineStrippedEqual("""
{lambda v0.
  # v0: float
  n0 = add v0 1.0
  n1 = jit_call[ func={lambda v1 v2.
                        # v1: float, v2: float
                        n0 = add v1 v2
                        in n0} ] n0 n0
  n2 = jit_call[ func={lambda v3 v4.
                        # v3: float, v4: float
                        n0 = add v3 v4
                        in n0} ] 3.0 n0
  n3 = add n1 n2
  in n3}
    """, str(func_tr.pp()))

  def test_cond(self):
    def func(x1):
      z = x1 * 2.
      return mj.Ops.cond_ge(x1,
                            lambda tv: z + tv + x1 * 3.,
                            lambda fv: z + x1 * 13., (x1 * 4.,))

    def func_equiv(x1):
      if x1 >= 0.:
        return x1 * 2. + x1 * 4. + x1 * 3.
      else:
        return x1 * 2. + x1 * 4. + x1 * 13.

    self.assertMultiLineStrippedEqual("""
{lambda v0.
  # v0: float
  n0 = mul v0 4.0
  n1 = mul v0 2.0
  n2 = cond_ge[ args=('n0', v0, 'n1', v0, 'n1')
                false_func={lambda v3 v7 v8 v0 v4.
                             # v3: float, v7: float, v8: float, v0: float, v4: float
                             n0 = mul v0 13.0
                             n1 = add v4 n0
                             in n1}
                pred_arg=v0
                true_func={lambda v1 v0 v2 v5 v6.
                            # v1: float, v0: float, v2: float, v5: float, v6: float
                            n0 = add v2 v1
                            n1 = mul v0 3.0
                            n2 = add n0 n1
                            in n2} ] 
  in n2}
      """, str(mj.trace(func)(3.).pp()))

    self.assertAllClose(func_equiv(5.), func(5.), check_dtypes=True)

  def test_jit_cond(self):
    def func(x1):
      z = x1 * 2.
      return mj.Ops.cond_ge(x1,
                            lambda tv: z + tv + x1 * 3.,
                            lambda fv: z + x1 * 13., (x1 * 4.,))

    def func_equiv(x1):
      if x1 >= 0.:
        return x1 * 2. + x1 * 4. + x1 * 3.
      else:
        return x1 * 2. + x1 * 4. + x1 * 13.

    self.assertAllClose(func_equiv(5.), mj.jit(func)(5.), check_dtypes=True)

  def test_trace_if_ge(self):
    """Test tracing through "if" """

    def func(x):
      z = x * 2.
      if z >= 4.:
        return z + 3.
      else:
        return z - 3.

    self.assertMultiLineStrippedEqual("""
{lambda v0.
  # v0: float
  n0 = mul v0 2.0
  n1 = add n0 3.0
  in n1}
    """, str(mj.trace(func, abstract=False)(2.).pp()))

    self.assertMultiLineStrippedEqual("""
{lambda v0.
  # v0: float
  n0 = mul v0 2.0
  n1 = sub n0 3.0
  in n1}
    """, str(mj.trace(func, abstract=False)(-1.9).pp()))

  def test_trace_if_gt(self):
    """Test tracing through "if" """

    def func(x):
      z = x * 2.
      if z > 4.:
        return z + 3.
      else:
        return z - 3.

    self.assertMultiLineStrippedEqual("""
{lambda v0.
  # v0: float
  n0 = mul v0 2.0
  n1 = add n0 3.0
  in n1}
    """, str(mj.trace(func, abstract=False)(2.01).pp()))

    self.assertMultiLineStrippedEqual("""
{lambda v0.
  # v0: float
  n0 = mul v0 2.0
  n1 = sub n0 3.0
  in n1}
    """, str(mj.trace(func, abstract=False)(-2.).pp()))

  def test_trace_cond_if(self):
    """Test tracing through "if" under "cond" """

    def func(x):
      z = x * 2.

      def true_f(xt):
        if xt >= 0.:
          return xt + 3.
        else:
          return xt - 3.

      return mj.Ops.cond_ge(z, true_f, lambda _: 0., (z,))

    self.assertMultiLineStrippedEqual("""
{lambda v0.
  # v0: float
  n0 = mul v0 2.0
  n1 = cond_ge[ args=('n0',)
                false_func={lambda v2.
                             # v2: float
                             in 0.0}
                pred_arg=n0
                true_func={lambda v1.
                            # v1: float
                            n0 = add v1 3.0
                            in n0} ] 
  in n1}
    """,
                                      str(mj.trace(func, abstract=False)(
                                        3.).pp()))

    self.assertMultiLineStrippedEqual("""
{lambda v0.
  # v0: float
  n0 = mul v0 2.0
  n1 = cond_ge[ args=('n0',)
                false_func={lambda v2.
                             # v2: float
                             in 0.0}
                pred_arg=n0
                true_func={lambda v1.
                            # v1: float
                            n0 = sub v1 3.0
                            in n0} ] 
  in n1}
    """,
                                      str(mj.trace(func, abstract=False)(
                                        -3.).pp()))

  def test_trace_jit_if_static(self):
    """Test tracing through "if" on static args under "jit" """

    def func(x):
      z = x * 2.

      def inner(y):
        if z >= 0.:  # Conditional is on the concrete parameter
          return y + 3.
        else:
          return y - 3.

      return mj.jit(inner)(z)

    self.assertMultiLineStrippedEqual("""
{lambda v0.
  # v0: float
  n0 = mul v0 2.0
  n1 = jit_call[ func={lambda v1.
                        # v1: float
                        n0 = add v1 3.0
                        in n0} ] n0
  in n1}
    """, str(mj.trace(func, abstract=False)(3.).pp()))

    self.assertMultiLineStrippedEqual("""
{lambda v0.
  # v0: float
  n0 = mul v0 2.0
  n1 = jit_call[ func={lambda v1.
                        # v1: float
                        n0 = sub v1 3.0
                        in n0} ] n0
  in n1}
    """, str(mj.trace(func, abstract=False)(-3.).pp()))

  def test_trace_jit_if_dynamic(self):
    """Test tracing through "if" on dynamic args under "jit" """

    def func(x):
      z = x * 2.

      def inner(y):
        if y >= 0.:  # Conditional is on the abstract parameter
          return y + 3.
        else:
          return y - 3.

      return mj.jit(inner)(z)

    with self.assertRaisesRegex(TypeError,
                                "Boolean test not supported on abstract values"):
      mj.trace(func)(3.).pp()

  def test_jit_cache(self):
    def func(x):
      return x * 2.

    jvp_func = mj.jvp(func)  # Save it, to attach the cache to it
    self.assertAllClose((2., 2.), mj.jit(jvp_func)(1., 1.), check_dtypes=True)
    self.assertAllClose((2., 2.), mj.jit(jvp_func)(1., 1.), check_dtypes=True)
    self.assertEqual(dict(hits=1, misses=1), Cache.get_info(jvp_func))
    # The func trace cache is never hit, because the jvp_fund cache is hit
    self.assertEqual(dict(hits=0, misses=1), Cache.get_info(func))

    self.assertAllClose((2., 2.), jvp_func(1., 1.), check_dtypes=True)
    self.assertEqual(dict(hits=1, misses=1), Cache.get_info(func))

  def test_jit_cache_2(self):
    def func(x):
      return x * 2.

    self.assertAllClose((2., 2.), mj.jvp(mj.jit(func))(1., 1.),
                        check_dtypes=True)
    self.assertAllClose((2., 2.), mj.jvp(mj.jit(func))(1., 1.),
                        check_dtypes=True)
    # We still hit the `func` cache for mj.jit
    self.assertEqual(dict(hits=1, misses=1), Cache.get_info(func))

    self.assertAllClose(2., mj.jit(func)(1.), check_dtypes=True)
    self.assertEqual(dict(hits=2, misses=1), Cache.get_info(func))


#   def test_breakpoint_grad(self):
#     """Try out debugging for grad."""
#
#     def identity(z):
#       return z
#
#     def func(x, y):
#       return x * mj.jit(identity)(x * y)
#
#     self.assertEqual(3. * (3. * 4.), func(3., 4.))
#     self.assertEqual((2. * 3. * 4., 3. * 3.), mj.grad(func, abstract=False)(3., 4.))
#     self.assertMultiLineStrippedEqual("""
# {lambda v0 v1.
#   # v0: float, v1: float
#   n0 = jit_call[ func={lambda v2 v3.
#                         # v2: float, v3: float
#                         n0 = mul v2 v3
#                         n1 = Op[myop] n0
#                         n2 = mul 2.0 n0
#                         n3 = mul n2 v2
#                         n4 = mul n3 v3
#                         n5 = add n1 n4
#                         n6 = mul v2 n3
#                         in (n5 n6,)} ] v0 v1
#   n1 = proj[ idx=0 ] n0
#   n2 = proj[ idx=1 ] n0
#   in (n1 n2,)}
#         """, str(mj.trace(mj.jit(mj.grad(func)))(3., 4.)))
#     self.assertAllClose(((3. * 4.) ** 2 + 3. * 2. * 3. * 4. * 4.,
#                          3. * 2. * 3. * 4. * 3.),
#                         mj.jit(mj.grad(func))(3., 4.),
#                         check_dtypes=True)


class ComparativeJaxTest(jtu.JaxTestCase):
  """Tests using regular JAX, for comparison."""

  def test_captured_computation(self):
    def func1(x):
      z = x * 2.

      def inner(y):
        return y + x * 4. + z

      return api.jit(inner)(x * 3.)

    print(api.make_jaxpr(func1)(5.))

  def test_grad_sharing(self):
    def func(x, y):
      z = x * y
      z = 2. * z + 3. * z
      z = 2. * z + 3. * z
      z = 2. * z + 3. * z
      z = 2. * z + 3. * z
      z = 2. * z + 3. * z
      return z

    print(api.make_jaxpr(api.grad(func, argnums=(0, 1)))(3., 4.))

  def test_grad_jit_multiple_results(self):
    def func(x1, y1):
      def inner(x2, y2, z3):
        z = x2 * y2
        return (2. * x2 + z, z, z * 6.)

      res1, res2, res3 = api.jit(inner)(3. * x1 + 4. * y1, 5. * x1, 8. * x1)
      return res1 + res2 + res3

    print(api.make_jaxpr(api.grad(func, argnums=(0, 1)))(3., 4.))

  def test_grad_jit_swap(self):
    """Compare the result if we swap grad and jit."""

    def func(x1, y1):
      return x1 * y1

    print(api.make_jaxpr(api.jit(api.grad(func, argnums=(0, 1))))(3., 4.))
    print(api.make_jaxpr(api.grad(api.jit(func), argnums=(0, 1)))(3., 4.))

  def test_grad_jit_swap_mj(self):
    """Compare the result if we swap grad and jit."""
    def func(x1, y1):
      return x1 * y1

    print(mj.trace(mj.jit(mj.grad(func)))(3., 4.))
    print(mj.trace(mj.grad(mj.jit(func)))(3., 4.))

  def test_grad_concrete(self):
    """Gradient with concrete control-flow", and jit"""

    def func(x1, y1):
      if x1 >= 0:
        return x1 * y1
      else:
        return x1 + y1

    self.assertAllClose((4., 3.), api.grad(func, argnums=(0, 1))(3., 4.),
                        check_dtypes=True)
    self.assertAllClose((1., 1.), api.grad(func, argnums=(0, 1))(-3., 4.),
                        check_dtypes=True)
    # Cannot jit through control flow
    with self.assertRaisesRegex(TypeError, "Abstract value passed to `bool`"):
      api.jit(func)(3., 4.)

    # cannot even jit after applying grad (why???)
    with self.assertRaisesRegex(TypeError, "Abstract value passed to `bool`"):
      api.jit(api.grad(func, argnums=(0, 1)))(3., 4.)

    # cannot make_jaxpr after grad
    with self.assertRaisesRegex(TypeError, "Abstract value passed to `bool`"):
      print(api.make_jaxpr(api.grad(func, argnums=(0, 1)))(3., 4.))

  def test_jit_cache(self):
    def func(x):
      def inner(y):
        return y * x

      res1 = api.jit(inner)(4.)
      x = -2.
      res2 = api.jit(inner)(4.)
      return res1 + res2

    self.assertAllClose(40., func(5.), check_dtypes=True)
    self.assertAllClose(-40., func(-5.), check_dtypes=True)





