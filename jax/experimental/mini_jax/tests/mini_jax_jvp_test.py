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


from jax import test_util as jtu
from jax.experimental import mini_jax as mj

from jax.config import config
config.parse_flags_with_absl()
FLAGS = config.FLAGS

class JvpTest(jtu.JaxTestCase):

  def test_jvp_simple(self):
    def func(x):
      return 1. + x + 2. * x + x * x

    #res = mj.jvp(func)(3., 5.)
    #self.assertEqual((19., 45.), res)
    self.assertMultiLineStrippedEqual("""
{lambda v0 v1.
  # v0: float, v1: float
  n0 = add 1.0 v0
  n1 = mul 2.0 v0
  n2 = add n0 n1
  n3 = mul v0 v0
  n4 = add n2 n3
  n5 = mul 2.0 v1
  n6 = add v1 n5
  n7 = mul v1 v0
  n8 = mul v0 v1
  n9 = add n7 n8
  n10 = add n6 n9
  in (n4 n10,)}
    """, str(mj.trace(mj.jvp(func))(3., 5.).pp()))

  def test_jvp_twice(self):
    def func(x):
      return 1. + x + 2. * x + x ** 3

    def diff_once(x):
      _, res_tan = mj.jvp(func)(x, 1.)
      return res_tan
    def diff_twice(x):
      return mj.jvp(diff_once)(x, 1.)[1]

    # First derivative should be (1. + 2. + 3. * x**2) * x_tan
    self.assertMultiLineStrippedEqual("""
    {lambda v0.
      # v0: float
      n0 = pow[ pow=2 ] v0
      n1 = mul 3.0 n0
      n2 = add 3.0 n1
      in n2}
    """, str(mj.trace(diff_once)(3.).pp()))

    # Second derivative should be (3. * 2. * x) * x_tan
    self.assertMultiLineStrippedEqual("""
{lambda v0.
  # v0: float
  n0 = mul 2.0 v0
  n1 = mul 3.0 n0
  in n1}
    """, str(mj.trace(diff_twice)(3.).pp()))

    res = diff_twice(3.)
    self.assertEqual(18., res)


  def test_jvp_nested_function(self):
    def func(x, y):
      z = 2. * x
      def inner(w):
        return z + y + w
      return inner(x)
    def func_equiv(x, y):
      return 2. * x + y + x
    self.assertEqual(func_equiv(3., 5.), func(3., 5.))

    # Derivative: 2. * x_tan + y_tan + x_tan
    self.assertMultiLineStrippedEqual("""
    {lambda v0 v1 v2 v3.
      # v0: float, v1: float, v2: float, v3: float
      n0 = mul 2.0 v0
      n1 = add n0 v1
      n2 = add n1 v0
      n3 = mul 2.0 v2
      n4 = add n3 v3
      n5 = add n4 v2
      in (n2 n5,)}
    """, str(mj.trace(mj.jvp(func))(3., 5., 30., 50.).pp()))
    self.assertEqual((14., 140.), mj.jvp(func)(3., 5., 30., 50.))

  def test_jvp_nested_jvp_freevars(self):
    # func(x, y) = (2 * x + y) * 7
    def func(x, y):
      z = 2. * x

      def inner(w):
        return z * w + y * w

      _, res_tan = mj.jvp(inner)(x, 7.)
      return res_tan

    def func_equiv(x, y):
      return (2. * x + y) * 7.

    self.assertEqual(func_equiv(3., 5.), func(3., 5.))

    # Derivative: jvp_func(x, y, x_tan, y_tan) = ((2 * x + y) * 7,   (2. * x_tan + y_tan) * 7)
    self.assertMultiLineStrippedEqual("""
{lambda v0 v1 v2 v3.
  # v0: float, v1: float, v2: float, v3: float
  n0 = mul 2.0 v0
  n1 = mul n0 7.0
  n2 = mul v1 7.0
  n3 = add n1 n2
  n4 = mul 2.0 v2
  n5 = mul n4 7.0
  n6 = mul v3 7.0
  n7 = add n5 n6
  in (n3 n7,)}
    """, str(mj.trace(mj.jvp(func))(3., 5., 30., 50.).pp()))
    self.assertEqual((func_equiv(3., 5.), (2. * 30. + 50.) * 7.),
                     mj.jvp(func)(3., 5., 30., 50.))

  def test_jvp_nested_jvp_3(self):
    def func1(x):
      s = 2. * x
      def func2(y):
        u = 3. * y + 4. * x + 5. * s
        # u == 3y + 14x
        def func3(z):
          # 6z + 21y + 98x + 8x == 6z + 21y + 106x
          return 6. * z + 7. * u + 8. * x
        v, vt = mj.jvp(func3)(1., u)
        # v == 6 + 21y + 106x
        # vt == 6u == 18y + 84x
        # v + vt = 6 + 39y + 190x
        return v + vt

      w, wt = mj.jvp(func2)(x, s)
      # w = 6 + 39x + 190x = 6 + 229x
      # wt = 39s = 78x
      # w + wt = 6 + 307x
      return w + wt

    def func_equiv(x):
      return 6 + 307. * x
    self.assertEqual(func_equiv(3.), func1(3.))


  def test_second_derivative(self):
    def func1(y):
      def func0(x):
        return x * x
      return mj.jvp(func0)(y, 1.)[1]
    second_derivative_at_3 = mj.jvp(func1)(3., 1.)[1]
    self.assertAllClose(2., second_derivative_at_3, check_dtypes=True)

  def test_jvp_multiple_results(self):
    def func(x):
      return (1. + 4. * x, x * x)

    res = mj.trace(mj.jvp(func))(3., 5.)
    self.assertEqual((13., 9., 4. * 5., 2. * 3. * 5.), mj.jvp(func)(3., 5.))
    self.assertMultiLineStrippedEqual("""
{lambda v0 v1.
  # v0: float, v1: float
  n0 = mul 4.0 v0
  n1 = add 1.0 n0
  n2 = mul v0 v0
  n3 = mul 4.0 v1
  n4 = mul v1 v0
  n5 = mul v0 v1
  n6 = add n4 n5
  in (n1 n2 n3 n6,)}
      """, str(mj.trace(mj.jvp(func))(3., 5.).pp()))

  def test_jit_jvp(self):
    def func(x):
      return 1. + x + 2. * x + x * x

    self.assertEqual((19., 45.), mj.jvp(func)(3., 5.))
    jit_jvp = mj.jit(mj.jvp(func))
    res = mj.trace(jit_jvp)(3., 5.)
    print(res.pp())

    self.assertEqual((19., 45.), jit_jvp(3., 5.))

  def test_jvp_jit(self):
    def func(x):
      y = 2. * x

      def inner(z):
        return y * z

      return 1. + y + mj.jit(inner)(5.)

    self.assertEqual((37., 60.), mj.jvp(func)(3., 5.))
    self.assertMultiLineStrippedEqual("""
{lambda v0 v1.
  # v0: float, v1: float
  n0 = mul 2.0 v0
  n1 = add 1.0 n0
  n2 = mul 2.0 v1
  n3 = jit_call[ func={lambda v5 v6 v7 v8.
                        # v5: float, v6: float, v7: float, v8: float
                        n0 = mul v6 v5
                        n1 = mul v8 v5
                        n2 = mul v6 v7
                        n3 = add n1 n2
                        in (n0 n3,)} ] 5.0 n0 0.0 n2
  n4 = proj[ idx=0 ] n3
  n5 = add n1 n4
  n6 = proj[ idx=1 ] n3
  n7 = add n2 n6
  in (n5 n7,)}
              """,
                                      str(mj.trace(mj.jvp(func, cache=False))(3., 5.).pp()))



  def test_jvp_jit_2(self):
    """This example used to throw an error in real-Jax"""

    def func(x):
      def inner(y):
        return y * x

      # Must have two calls to the inner jit (which is cached)
      res1 = mj.jit(inner)(4.)
      res2 = mj.jit(inner)(5.)
      return res1 + res2

    self.assertAllClose((45., 9.), mj.jvp(func)(5., 1.,), check_dtypes=True)

  def test_jvp_jit_multiple_results(self):
    def func(x):
      x += 2. * x
      def inner(z):
        return (x * z, z)
      inner_res = mj.jit(inner)(5.)
      return 1. + x + inner_res[0] + inner_res[1] * 2.
    def func_equiv(x):
      return 1. + (3. * x) + (3. * x) * 5. + 5. * 2.

    self.assertEqual(func_equiv(3.), func(3.))

    jvp_func = mj.jvp(func, cache=False)
    self.assertEqual((func_equiv(3.), 5. * (3. + 15.)), jvp_func(3., 5.))
    self.assertMultiLineStrippedEqual("""
{lambda v0 v1.
  # v0: float, v1: float
  n0 = mul 2.0 v0
  n1 = add v0 n0
  n2 = add 1.0 n1
  n3 = mul 2.0 v1
  n4 = add v1 n3
  n5 = jit_call[ func={lambda v5 v6 v7 v8.
                        # v5: float, v6: float, v7: float, v8: float
                        n0 = mul v6 v5
                        n1 = mul v8 v5
                        n2 = mul v6 v7
                        n3 = add n1 n2
                        in (n0 v5 n3 v7,)} ] 5.0 n1 0.0 n4
  n6 = proj[ idx=0 ] n5
  n7 = add n2 n6
  n8 = proj[ idx=1 ] n5
  n9 = mul n8 2.0
  n10 = add n7 n9
  n11 = proj[ idx=2 ] n5
  n12 = add n4 n11
  n13 = proj[ idx=3 ] n5
  n14 = mul n13 2.0
  n15 = add n12 n14
  in (n10 n15,)}
  """, str(mj.trace(jvp_func)(3., 5.).pp()))


  def test_jvp_cond(self):
    def func(x1):
      z = x1 * 2.
      return mj.Ops.cond_ge(x1,
                 lambda tv: z + tv + x1 * 3.,
                 lambda fv: z + fv + x1 * 13., (x1 * 4.,))
    def func_equiv(x1):
      if x1 >= 0.:
        return x1 * 2. + x1 * 4. + x1 * 3.
      else:
        return x1 * 2. + x1 * 4. + x1 * 13.

    self.assertAllClose(func_equiv(5.), func(5.), check_dtypes=True)
    self.assertAllClose(func_equiv(-5.), func(-5.), check_dtypes=True)
    self.assertAllClose((func_equiv(5.), 2. + 4. + 3.),
                        mj.jvp(func)(5., 1.),
                        check_dtypes=True)
    self.assertAllClose((func_equiv(-5.), 2. + 4. + 13.),
                        mj.jvp(func)(-5., 1.),
                        check_dtypes=True)
    self.assertMultiLineStrippedEqual("""
{lambda v0 v1.
  # v0: float, v1: float
  n0 = mul v0 4.0
  n1 = mul v0 2.0
  n2 = mul v1 4.0
  n3 = mul v1 2.0
  n4 = cond_ge[ args=('n0', v0, 'n1', v0, 'n1', 'n2', v1, 'n3', v1, 'n3')
                false_func={lambda v21 v22 v23 v24 v25 v26 v27 v28 v29 v30.
                             # v21: float, v22: float, v23: float, v24: float, v25: float, v26: float, v27: float, v28: float, v29: float, v30: float
                             n0 = add v25 v21
                             n1 = mul v24 13.0
                             n2 = add n0 n1
                             n3 = add v30 v26
                             n4 = mul v29 13.0
                             n5 = add n3 n4
                             in (n2 n5,)}
                pred_arg=v0
                true_func={lambda v11 v12 v13 v14 v15 v16 v17 v18 v19 v20.
                            # v11: float, v12: float, v13: float, v14: float, v15: float, v16: float, v17: float, v18: float, v19: float, v20: float
                            n0 = add v13 v11
                            n1 = mul v12 3.0
                            n2 = add n0 n1
                            n3 = add v18 v16
                            n4 = mul v17 3.0
                            n5 = add n3 n4
                            in (n2 n5,)} ] 
  n5 = proj[ idx=0 ] n4
  n6 = proj[ idx=1 ] n4
  in (n5 n6,)}
      """, str(mj.trace(mj.jvp(func, cache=False))(5., 1.).pp()))


  def test_jvp_cond_0(self):
    def func(x):
      return mj.Ops.cond_ge(x, lambda tv: 1., lambda fv: 0., (0.,))

    res = mj.jvp(func)(0., 3.)
    self.assertAllClose((1., 0.), res, check_dtypes=True)
    print(mj.trace(mj.jvp(func))(0., 3.).pp())

  def test_jvp_cond_multiple_results(self):
    def func(x1):
      z = x1 * 2.
      return mj.Ops.cond_ge(x1,
                 lambda tv: (z + tv + x1 * 3., tv),
                 lambda fv: (z + fv + x1 * 13., fv), (x1 * 4.,))
    def func_equiv(x1):
      if x1 >= 0.:
        return (x1 * 2. + x1 * 4. + x1 * 3., x1 * 4.)
      else:
        return (x1 * 2. + x1 * 4. + x1 * 13., x1 * 4.)

    self.assertAllClose(func_equiv(5.), func(5.), check_dtypes=True)
    self.assertAllClose(func_equiv(-5.), func(-5.), check_dtypes=True)
    self.assertAllClose((func_equiv(5.)[0], func_equiv(5.)[1],
                         2. + 4. + 3., 4.),
                        mj.jvp(func)(5., 1.),
                        check_dtypes=True)
    self.assertAllClose((func_equiv(-5.)[0], func_equiv(-5.)[1],
                         2. + 4. + 13., 4.),
                        mj.jvp(func)(-5., 1.),
                        check_dtypes=True)

  def test_jvp_if(self):
    """Test jvp through "if" """
    def func(x):
      z = x * 2.
      if z >= 0.:
        return z + 3.
      else:
        return z * 3.

    self.assertEqual((9., 5. * 2.), mj.jvp(func, abstract=False)(3., 5.))
    self.assertEqual((-18., 5. * 2. * 3.), mj.jvp(func, abstract=False)(-3., 5.))

  def test_jvp_nested_if(self):
    """Test jvp nested through "if" """
    def func(x):
      z = x * 2.
      def inner(y):
        if y >= 0.:
          return y + 3.
        else:
          return y * 3.
      r, r_t = mj.jvp(inner, abstract=False)(z, 2.)
      return r + r_t

    self.assertEqual((11., 5. * 2.), mj.jvp(func, abstract=False)(3., 5.))
    self.assertEqual((-12., 5. * 2. * 3.), mj.jvp(func, abstract=False)(-3., 5.))

  def test_jvp_nested_if_freevar(self):
    """Test jvp nested through "if" """
    def func(x):
      z = x * 2.
      def inner(y):
        if z >= 0.:  # Conditional on a freevar
          return y + 3.
        else:
          return y * 3.
      r, r_t = mj.jvp(inner, abstract=False)(x, 2.)
      return r + r_t

    self.assertEqual((8., 5.), mj.jvp(func, abstract=False)(3., 5.))
    self.assertEqual((-3., 5. * 3.), mj.jvp(func, abstract=False)(-3., 5.))