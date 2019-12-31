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


class GradTest(jtu.JaxTestCase):

  def test_grad_simple(self):
    def func(x, y):
      return 2. * x + x * y
    grad_func = mj.grad(func, cache=False)
    self.assertMultiLineStrippedEqual("""
    {lambda v0 v1.
      # v0: float, v1: float
      n0 = add 2.0 v1
      in (n0 v0,)}
        """, str(mj.trace(grad_func)(3., 5.).pp()))

    self.assertEqual((7., 3.), grad_func(3., 5.))

    jit_grad_func = mj.jit(grad_func, cache=False)
    self.assertMultiLineStrippedEqual("""
    {lambda v0 v1.
      # v0: float, v1: float
      n0 = jit_call[ func={lambda v2 v3.
                            # v2: float, v3: float
                            n0 = add 2.0 v3
                            in (n0 v2,)} ] v0 v1
      n1 = proj[ idx=0 ] n0
      n2 = proj[ idx=1 ] n0
      in (n1 n2,)}
        """, str(mj.trace(jit_grad_func, cache=False)(3., 5.).pp()))

    self.assertEqual((7., 3.), jit_grad_func(3., 5.))

  def test_grad_unused(self):
    """An used parameter"""
    def f31(x):
      return 0.
    self.assertAllClose(0., mj.grad(f31)(0.), check_dtypes=True)

  def test_grad_jit(self):
    def func(x, y):
      return 2. * x + x * y
    grad_func = mj.grad(mj.jit(func))
    self.assertMultiLineStrippedEqual("""
    {lambda v0 v1.
      # v0: float, v1: float
      n0 = jit_call[ func={lambda v6 v7 v8.
                            # v6: float, v7: float, v8: float
                            n0 = mul 2.0 v8
                            n1 = mul v8 v7
                            n2 = add n0 n1
                            n3 = mul v6 v8
                            in (n2 n3,)} ] v0 v1 1.0
      n1 = proj[ idx=0 ] n0
      n2 = proj[ idx=1 ] n0
      in (n1 n2,)}
      """, str(mj.trace(grad_func)(3., 5.).pp()))

    self.assertEqual((7., 3.), grad_func(3., 5.))

  def test_grad_jit_multiple_results(self):
    def func(x1, y1):
      def inner(x2, y2):
        z = x2 * y2
        return (2. * x2 + z, z)
      res1, res2 = mj.jit(inner)(3. * x1 + 4. * y1, 5. * x1)
      return res1 + 6. * res2
    def func_equiv(x1, y1):
      x2 = 3. * x1 + 4. * y1
      y2 = 5. * x1
      z = x2 * y2
      return (2. * x2 + z) + 6. * z
    self.assertAllClose(func_equiv(3., 5.), func(3., 5.), check_dtypes=True)

    grad_func = mj.grad(func)
    self.assertMultiLineStrippedEqual("""
    {lambda v0 v1.
      # v0: float, v1: float
      n0 = mul 3.0 v0
      n1 = mul 4.0 v1
      n2 = add n0 n1
      n3 = mul 5.0 v0
      n4 = jit_call[ func={lambda v6 v7 v8 v9.
                            # v6: float, v7: float, v8: float, v9: float
                            n0 = mul 2.0 v8
                            n1 = add v9 v8
                            n2 = mul n1 v7
                            n3 = add n0 n2
                            n4 = mul v6 n1
                            in (n3 n4,)} ] n2 n3 1.0 6.0
      n5 = proj[ idx=0 ] n4
      n6 = mul 3.0 n5
      n7 = proj[ idx=1 ] n4
      n8 = mul 5.0 n7
      n9 = add n6 n8
      n10 = mul 4.0 n5
      in (n9 n10,)}
      """, str(mj.trace(grad_func)(3., 5.).pp()))

    self.assertEqual((1336., 428.), grad_func(3., 5.))

  def test_grad_sharing_1(self):
    """The grad computations is as big as the forward one"""
    def func(x, y, z):
      tmp = x * y * z
      tmp = tmp * 2.
      tmp = tmp * 3.
      tmp = tmp * 4.
      return tmp

    grad_func = mj.grad(func)
    self.assertMultiLineStrippedEqual("""
    {lambda v0 v1 v2.
      # v0: float, v1: float, v2: float
      n0 = mul 24.0 v2
      n1 = mul n0 v1
      n2 = mul v0 n0
      n3 = mul v0 v1
      n4 = mul n3 24.0
      in (n1 n2 n4,)}
      """, str(mj.trace(grad_func)(3., 4., 5.).pp()))

  def test_grad_sharing_2(self):
    """The grad computations is as big as the forward one"""
    def func(x, y):
      z = x * y
      z = 0.5 * (z + z)
      z = 0.5 * (z + z)
      z = 0.5 * (z + z)
      z = 0.5 * (z + z)
      z = 0.5 * (z + z)
      return z

    grad_func = mj.grad(func)
    self.assertMultiLineStrippedEqual("""
    {lambda v0 v1.
      # v0: float, v1: float
      in (v1 v0,)}
      """, str(mj.trace(grad_func)(3., 4.).pp()))

  def test_grad_sharing_3(self):
    """The grad computations is as big as the forward one"""
    size = [3]
    def func(x, y, v2, v3):
      def inner(x, y):
        z = x * y
        for _ in range(size[0]):
          z = z * v2 + z * v3
        return z
      dx, dy = mj.grad(inner)(x, y)
      return dx, dy

    size[0] = 3
    self.assertMultiLineStrippedEqual("""
    {lambda v0 v1 v2 v3.
      # v0: float, v1: float, v2: float, v3: float
      n0 = add v2 v3
      n1 = mul n0 v2
      n2 = mul n0 v3
      n3 = add n1 n2
      n4 = mul n3 v2
      n5 = mul n3 v3
      n6 = add n4 n5
      n7 = mul n6 v1
      n8 = mul v0 n6
      in (n7 n8,)}
      """, str(mj.trace(func)(3., 4., 5., 6.).pp()))

    # Now test that the size of gradient grows linearly
    def get_grad_size(sz: int) -> int:
      size[0] = sz
      grad_sz = str(mj.trace(func)(3., 4., 5., 6.).pp())
      print("For computation size {} grad size {}".format(sz, len(grad_sz)))
      return len(grad_sz)

    sizes = [5, 10, 15, 20]
    grad_sizes = [get_grad_size(sz) for sz in sizes]
    growth_factor = (grad_sizes[1] - grad_sizes[0])/(sizes[1] - sizes[0])
    for idx in range(2, len(sizes)):
      self.assertAllClose(growth_factor,
                          (grad_sizes[idx] - grad_sizes[0]) / (sizes[idx] - sizes[0]),
                          check_dtypes=True,
                          rtol=0.1)


  def test_jit_grad_nested(self):
    def func(x1):
      z11 = x1 * 11.
      def inner(x2):
        return z11 + x1 * 12. + x2 * 22.
      return mj.grad(inner)(x1)
    func_jit = mj.jit(func, cache=False)
    def func_equiv(x1):
      return 22.
    self.assertAllClose(func_equiv(5.), func_jit(5.), check_dtypes=True)

    self.assertMultiLineStrippedEqual("""
    {lambda v0.
      # v0: float
      n0 = jit_call[ func={lambda v1.
                            # v1: float
                            in 22.0} ] v0
      in n0}
      """, str(mj.trace(func_jit)(3.).pp()))

    self.assertEqual(22., func_jit(5.))

  def test_grad_jit2(self):
    """Grad over a sequence of two jit calls, with dependency on forward values."""
    def func(x1):
      def inner1(x2):
        return x2 * x2
      def inner2(x2):
        return x2 ** 3
      r1 = mj.jit(inner1)(x1)
      return mj.jit(inner2)(r1 * r1 * 2.)

    self.assertMultiLineStrippedEqual("""
    {lambda v0.
      # v0: float
      n0 = jit_call[ func={lambda v2.
                            # v2: float
                            n0 = mul v2 v2
                            in n0} ] v0
      n1 = mul n0 n0
      n2 = mul n1 2.0
      n3 = jit_call[ func={lambda v4 v5.
                            # v4: float, v5: float
                            n0 = mul v5 3.0
                            n1 = pow[ pow=2 ] v4
                            n2 = mul n0 n1
                            in n2} ] n2 1.0
      n4 = mul n3 2.0
      n5 = mul n4 n0
      n6 = mul n0 n4
      n7 = add n5 n6
      n8 = jit_call[ func={lambda v6 v7.
                            # v6: float, v7: float
                            n0 = mul v7 v6
                            n1 = mul v6 v7
                            n2 = add n0 n1
                            in n2} ] v0 n7
      in n8}
      """, str(mj.trace(mj.grad(func))(3.).pp()))

  def test_grad_jit_swap(self):
    def func(x1, y1):
      return x1 * y1

    print(mj.trace(mj.jit(mj.grad(func)))(3., 4.).pp())
    print(mj.trace(mj.grad(mj.jit(func)))(3., 4.).pp())

  def test_grad_cond(self):
    def func(x1):
      z = x1 * 2.
      return mj.Ops.cond_ge(x1,
                 lambda tv: z + tv + x1 * 3., (x1 * 4.,),
                 lambda fv: z + fv + x1 * 13., (x1 * 14.,))
    def func_equiv(x1):
      if x1 >= 0.:
        return x1 * 2. + x1 * 4. + x1 * 3.
      else:
        return x1 * 2. + x1 * 14. + x1 * 13.

    self.assertAllClose(func_equiv(5.), func(5.), check_dtypes=True)
    self.assertAllClose(func_equiv(-5.), func(-5.), check_dtypes=True)
    self.assertAllClose(2. + 4. + 3.,
                        mj.grad(func)(5.),
                        check_dtypes=True)
    self.assertAllClose(2. + 14. + 13.,
                        mj.grad(func)(-5.),
                        check_dtypes=True)
    self.assertMultiLineStrippedEqual("""
{lambda v0.
  # v0: float
  n0 = mul v0 4.0
  n1 = mul v0 2.0
  n2 = mul v0 14.0
  n3 = cond_ge[ false_args=('n2', 'n1', v0, 1.0)
                false_func={lambda v10 v11 v12 v13.
                             # v10: float, v11: float, v12: float, v13: float
                             n0 = mul v13 13.0
                             in (0.0 0.0 0.0 v13 v13 n0,)}
                pred_arg=v0
                true_args=('n0', 'n1', v0, 1.0)
                true_func={lambda v6 v7 v8 v9.
                            # v6: float, v7: float, v8: float, v9: float
                            n0 = mul v9 3.0
                            in (v9 v9 n0 0.0 0.0 0.0,)} ] 
  n4 = proj[ idx=2 ] n3
  n5 = proj[ idx=5 ] n3
  n6 = add n4 n5
  n7 = proj[ idx=0 ] n3
  n8 = mul n7 4.0
  n9 = add n6 n8
  n10 = proj[ idx=1 ] n3
  n11 = proj[ idx=4 ] n3
  n12 = add n10 n11
  n13 = mul n12 2.0
  n14 = add n9 n13
  n15 = proj[ idx=3 ] n3
  n16 = mul n15 14.0
  n17 = add n14 n16
  in n17}
      """, str(mj.trace(mj.grad(func, cache=False))(5.).pp()))

  def test_grad_shared_body(self):
    def func(x):
      def f5(y):
        # We use the same body twice, we have to count the adjoints from each
        z = 3. * y
        return z, z

      v3, v4 = mj.jit(f5)(x)
      return v3 + v4
    def func_equiv(v1):
      return 2. * v1 * 3.

    self.assertAllClose(func_equiv(0.), func(0.), check_dtypes=True)
    self.assertAllClose(func_equiv(1.), func(1.), check_dtypes=True)

    self.assertAllClose(6.,
                        mj.grad(func)(0.),
                        check_dtypes=True)
    self.assertMultiLineStrippedEqual("""
{lambda v0.
  # v0: float
  n0 = jit_call[ func={lambda v3 v4 v5.
                        # v3: float, v4: float, v5: float
                        n0 = add v4 v5
                        n1 = mul 3.0 n0
                        in n1} ] v0 1.0 1.0
  in n0}
      """,
                                      str(mj.trace(mj.grad(func, cache=False))(0.).pp()))

  def test_grad_jvp(self):
    def f0(v1):
      def inner(y):
        return y

      _, v17 = mj.jvp(inner)(0., v1)
      return v17

    _result = mj.grad(f0)(0.)
    self.assertMultiLineStrippedEqual("""
{lambda v0.
  # v0: float
  in 1.0}
      """, str(mj.trace(mj.grad(f0))(0.).pp()))

  def test_grad_if(self):
    """Test grad through "if" """
    def func(x):
      z = x * 2.
      if z >= 0.:
        return z + 3.
      else:
        return z * 3.

    self.assertEqual(2., mj.grad(func, abstract=False)(3.))
    self.assertEqual(2. * 3., mj.grad(func, abstract=False)(-3.))