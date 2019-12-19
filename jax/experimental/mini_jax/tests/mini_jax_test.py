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


from jax import api
from jax import lax
from jax import test_util as jtu
from jax.experimental import mini_jax as mj

from jax.config import config
config.parse_flags_with_absl()
FLAGS = config.FLAGS

class MiniJaxTest(jtu.JaxTestCase):

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

    fun1_jit = mj.jit(fun1)
    self.assertMultiLineStrippedEqual("""
    {lambda v0.
      # v0: float
      n0 = jit_call[ func={lambda v1.
                            # v1: float
                            n0 = mul v1 11.0
                            n1 = jit_call[ func={lambda v2 v3 v1.
                                                  # v2: float, v3: float, v1: float
                                                  n0 = mul v1 12.0
                                                  n1 = add v3 n0
                                                  n2 = mul v2 22.0
                                                  n3 = add n1 n2
                                                  in n3} ] v1 n0 v1
                            in n1} ] v0
      in n0} 
    """, str(mj.trace(fun1_jit)(5.)))

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
                            n1 = jit_call[ func={lambda v2 v7 v3 v1.
                                                  # v2: float, v7: float, v3: float, v1: float
                                                  n0 = mul v1 12.0
                                                  n1 = add v3 n0
                                                  n2 = mul v2 22.0
                                                  n3 = add n1 n2
                                                  n4 = jit_call[ func={lambda v4 v5 v6 v2 v1.
                                                                        # v4: float, v5: float, v6: float, v2: float, v1: float
                                                                        n0 = mul v5 13.0
                                                                        n1 = mul v6 23.0
                                                                        n2 = add n0 n1
                                                                        n3 = mul v2 23.0
                                                                        n4 = add n2 n3
                                                                        n5 = mul v2 23.0
                                                                        n6 = add n4 n5
                                                                        n7 = mul v1 v1
                                                                        n8 = add n6 n7
                                                                        in n8} ] v2 v7 n3 v2 v1
                                                  in n4} ] v1 n0 n0 v1
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
      x += 10.
      def inner(z):
        return z + x + outside
      return mj.jit(inner)(x)

    self.assertEqual(31., func(5., 6.))

    func_jit = mj.jit(func)
    func_tr = mj.trace(func_jit)(5., 6.)
    self.assertMultiLineStrippedEqual("""
{lambda v0 v1.
  # v0: float, v1: float
  n0 = jit_call[ func={lambda v2 v3.
                        # v2: float, v3: float
                        n0 = add v2 10.0
                        n1 = jit_call[ func={lambda v4 v5.
                                              # v4: float, v5: float
                                              n0 = add v4 v5
                                              n1 = add n0 1.0
                                              in n1} ] n0 n0
                        in n1} ] v0 v1
  in n0}
    """, str(func_tr.pp()))

    self.assertEqual(31., func_jit(5., 6.))

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
    func_tr = mj.trace(func)(5.)
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
                 lambda tv: z + tv + x1 * 3., (x1 * 4.,),
                 lambda fv: z + x1 * 13., (x1 * 14.,))
    def func_equiv(x1):
      if x1 >= 0.:
        return x1 * 2. + x1 * 4. + x1 * 3.
      else:
        return x1 * 2. + x1 * 14. + x1 * 13.
    self.assertMultiLineStrippedEqual("""
{lambda v0.
  # v0: float
  n0 = mul v0 4.0
  n1 = mul v0 2.0
  n2 = mul v0 14.0
  n3 = cond_ge[ false_args=('n2', 'n1', v0)
                false_func={lambda v3 v4 v0.
                             # v3: float, v4: float, v0: float
                             n0 = mul v0 13.0
                             n1 = add v4 n0
                             in n1}
                pred_arg=v0
                true_args=('n0', 'n1', v0)
                true_func={lambda v1 v2 v0.
                            # v1: float, v2: float, v0: float
                            n0 = add v2 v1
                            n1 = mul v0 3.0
                            n2 = add n0 n1
                            in n2} ] 
  in n3}
      """, str(mj.trace(func)(3.).pp()))

    self.assertAllClose(func_equiv(5.), func(5.), check_dtypes=True)

  def test_jit_cond(self):
    def func(x1):
      z = x1 * 2.
      return mj.Ops.cond_ge(x1,
                 lambda tv: z + tv + x1 * 3., (x1 * 4.,),
                 lambda fv: z + x1 * 13., (x1 * 14.,))
    def func_equiv(x1):
      if x1 >= 0.:
        return x1 * 2. + x1 * 4. + x1 * 3.
      else:
        return x1 * 2. + x1 * 14. + x1 * 13.

    self.assertAllClose(func_equiv(5.), mj.jit(func)(5.), check_dtypes=True)



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
    def func(x1, y1):
      return x1 * y1

    print(api.make_jaxpr(api.jit(api.grad(func, argnums=(0, 1))))(3., 4.))
    print(api.make_jaxpr(api.grad(api.jit(func), argnums=(0, 1)))(3., 4.))

  def test_grad_cond(self):
    """Gradient with conditional control-flow"""
    def func(x1):
      z = x1 * 2.
      return lax.cond(x1 >= 0.,
                      x1 * 4., lambda tv: z + tv + x1 * 3.,
                      x1 * 14., lambda fv: z + fv + x1 * 13.)

    def func_equiv(x1):
      if x1 >= 0.:
        return x1 * 2. + x1 * 4. + x1 * 3.
      else:
        return x1 * 2. + x1 * 14. + x1 * 13.
    self.assertAllClose(func_equiv(5.), func(5.), check_dtypes=True)
    self.assertAllClose(func_equiv(-5.), func(-5.), check_dtypes=True)
    with self.assertRaisesRegex(NotImplementedError,
                                "Forward-mode differentiation rule for 'cond' not implemented"):
      self.assertAllClose(2. + 4. + 3.,
                          api.grad(func)(5.),
                          check_dtypes=True)


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

  def test_grad_concrete_cond(self):
    """Gradient with concrete control-flow and cond"""
    def func(x1, y1):
      return lax.cond(x1 >= 0, y1, lambda tv: tv * 2., y1, lambda fv: fv * 3.)

    with self.assertRaisesRegex(
        NotImplementedError,
        "Forward-mode differentiation rule for 'cond' not implemented"):
      api.grad(func, argnums=(0, 1))(3., 4.)

