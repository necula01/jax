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

class FlopsTest(jtu.JaxTestCase):

  def test_flops_simple(self):
    def func(x):
      z = x * 2.  # Cost: 1
      z = z ** 8  # Cost: 3
      z = z + 1. - x  # Cost: 2
      return z

    self.assertAllClose(6., mj.count_flops(func)(3.),
                        check_dtypes=True)

  def test_flops_sharing(self):
    """Count each operation once, even if we share sub-expressions."""
    def func(x):
      x = x + x
      x = x + x
      x = x + x
      x = x + x
      x = x + x
      return x

    self.assertAllClose(5., mj.count_flops(func)(3.),
                        check_dtypes=True)

  def test_flops_sharing_across_tuple(self):
    """Check for handling of sharing across multiple results."""
    def func(x):
      x = x + x
      x = x + x
      y = x
      x = x + x
      x = x + x
      return (x, y)

    self.assertAllClose(4., mj.count_flops(func)(3.),
                        check_dtypes=True)

  def test_flops_inner_call(self):
    def func(x):
      z = x * 2.  # Counts as 1
      def inner(y):  # Total = 7
        for _ in range(3):
          y = y + y  # Counts as 3
        y = y + y ** 8  # Counts as 1 + 3 (log2(8))
        return y
      return inner(z) + inner(z)  # 7 + 1 + 7

    self.assertMultiLineStrippedEqual("""
    {lambda v0.
      # v0: float
      in 16.0}
    """, str(mj.trace(mj.count_flops(func))(3.).pp()))

    self.assertAllClose(16., mj.count_flops(func)(3.),
                        check_dtypes=True)

  def test_flops_cond_same_flops(self):
    """In conditionals with same-flops branches, the count is lifted."""
    def func(x):
      return mj.Ops.cond_ge(x, # The conditional costs 1
                        # Both branches cost 1
                        lambda tv: tv + tv, (x,),
                        lambda fv: fv * fv, (x * 2.,) # x * 2. costs 1
                        )

    self.assertMultiLineStrippedEqual("""
{lambda v0.
  # v0: float
  n0 = mul v0 2.0
  n1 = cond_ge[ false_args=('n0',)
                false_func={lambda v2.
                             # v2: float
                             n0 = mul v2 v2
                             in n0}
                pred_arg=v0
                true_args=(v0,)
                true_func={lambda v1.
                            # v1: float
                            n0 = add v1 v1
                            in n0} ] 
  in n1}
    """, str(mj.trace(func)(3.).pp()))

    self.assertAllClose(3., mj.count_flops(func)(3.),
                        check_dtypes=True)

  def test_flops_cond_dep_flops(self):
    """In conditionals with data dependent flops, keep the conditional."""
    def func(x):
      return mj.Ops.cond_ge(x, # The conditional costs 1
                        # True branch costs 1 and false costs 2
                        lambda tv: tv + tv, (x,),
                        lambda fv: fv * fv * fv, (x * 2.,) # x * 2. costs 1
                        )

    self.assertMultiLineStrippedEqual("""
{lambda v0.
  # v0: float
  n0 = mul v0 2.0
  n1 = cond_ge[ false_args=('n0',)
                false_func={lambda v2.
                             # v2: float
                             n0 = mul v2 v2
                             n1 = mul n0 v2
                             in n1}
                pred_arg=v0
                true_args=(v0,)
                true_func={lambda v1.
                            # v1: float
                            n0 = add v1 v1
                            in n0} ] 
  in n1}
    """, str(mj.trace(func)(3.).pp()))

    self.assertMultiLineStrippedEqual("""
{lambda v0.
  # v0: float
  n0 = mul v0 2.0
  n1 = cond_ge[ false_args=('n0',)
                false_func={lambda v5.
                             # v5: float
                             in 2.0}
                pred_arg=v0
                true_args=(v0,)
                true_func={lambda v4.
                            # v4: float
                            in 1.0} ] 
  n2 = add 1.0 n1
  n3 = add 1.0 n2
  in n3}
    """, str(mj.trace(mj.count_flops(func))(3.).pp()))

    self.assertAllClose(3., mj.count_flops(func)(3.),
                        check_dtypes=True)
    self.assertAllClose(4., mj.count_flops(func)(-1.),
                        check_dtypes=True)


  def test_flops_jit(self):
    def func(x):
      z = x * 2.  # Counts as 1
      def inner(y):  # Total = 7
        for _ in range(3):
          y = y + y  # Counts as 3
        y = y + y ** 8  # Counts as 1 + 3 (log2(8))
        return y
      return mj.jit(inner)(z) + inner(z)  # 7 + 1 + 7 + 2 (cost of jit call with 1 arg)

    self.assertMultiLineStrippedEqual("""
    {lambda v0.
      # v0: float
      n0 = mul v0 2.0
      n1 = jit_call[ func={lambda v1.
                            # v1: float
                            n0 = add v1 v1
                            n1 = add n0 n0
                            n2 = add n1 n1
                            n3 = pow[ pow=8 ] n2
                            n4 = add n2 n3
                            in n4} ] n0
      n2 = add n0 n0
      n3 = add n2 n2
      n4 = add n3 n3
      n5 = pow[ pow=8 ] n4
      n6 = add n4 n5
      n7 = add n1 n6
      in n7}
    """, str(mj.trace(func)(3.).pp()))

    self.assertMultiLineStrippedEqual("""
    {lambda v0.
      # v0: float
      in 18.0}
    """, str(mj.trace(mj.count_flops(func))(3.).pp()))
    self.assertAllClose(18., mj.count_flops(func)(3.),
                        check_dtypes=True)

  def test_flops_jit_data_dep(self):
    """A conditional with data-dependent flops inside jit."""
    def func(x):
      z = x * 2.  # Counts as 1
      def inner(y):  # Cost: 1 + 1 + if y >= 0 then 1 else 2"
        return mj.Ops.cond_ge(y,  # The conditional costs 1
                   # True branch costs 1 and false costs 2
                   lambda tv: tv + tv, (y,),
                   lambda fv: fv * fv * fv, (y * 2.,)
                   )
      return (mj.jit(inner)(z)  # Cost: 2 + (if z >= 0 then 3 else 4)
              +   # Cost: 1
              inner(z)  # Cost: if z >= 0 then 3 else 4
              )

    self.assertMultiLineStrippedEqual("""
{lambda v0.
  # v0: float
  n0 = mul v0 2.0
  n1 = jit_call[ func={lambda v1.
                        # v1: float
                        n0 = mul v1 2.0
                        n1 = cond_ge[ false_args=('n0',)
                                      false_func={lambda v3.
                                                   # v3: float
                                                   n0 = mul v3 v3
                                                   n1 = mul n0 v3
                                                   in n1}
                                      pred_arg=v1
                                      true_args=(v1,)
                                      true_func={lambda v2.
                                                  # v2: float
                                                  n0 = add v2 v2
                                                  in n0} ] 
                        in n1} ] n0
  n2 = mul n0 2.0
  n3 = cond_ge[ false_args=('n2',)
                false_func={lambda v5.
                             # v5: float
                             n0 = mul v5 v5
                             n1 = mul n0 v5
                             in n1}
                pred_arg=n0
                true_args=('n0',)
                true_func={lambda v4.
                            # v4: float
                            n0 = add v4 v4
                            in n0} ] 
  n4 = add n1 n3
  in n4}
    """, str(mj.trace(func)(3.).pp()))

    self.assertMultiLineStrippedEqual("""
{lambda v0.
  # v0: float
  n0 = mul v0 2.0
  n1 = jit_call[ func={lambda v7.
                        # v7: float
                        n0 = mul v7 2.0
                        n1 = cond_ge[ false_args=('n0',)
                                      false_func={lambda v9.
                                                   # v9: float
                                                   in 2.0}
                                      pred_arg=v7
                                      true_args=(v7,)
                                      true_func={lambda v8.
                                                  # v8: float
                                                  in 1.0} ] 
                        n2 = add 1.0 n1
                        n3 = add 1.0 n2
                        in n3} ] n0
  n2 = add 2.0 n1
  n3 = add 1.0 n2
  n4 = add n3 1.0
  n5 = mul n0 2.0
  n6 = cond_ge[ false_args=('n5',)
                false_func={lambda v11.
                             # v11: float
                             in 2.0}
                pred_arg=n0
                true_args=('n0',)
                true_func={lambda v10.
                            # v10: float
                            in 1.0} ] 
  n7 = add 1.0 n6
  n8 = add n4 n7
  n9 = add n8 1.0
  in n9}
    """, str(mj.trace(mj.count_flops(func))(3.).pp()))
    self.assertAllClose(10., mj.count_flops(func)(3.),
                       check_dtypes=True)
    self.assertAllClose(12., mj.count_flops(func)(-3.),
                        check_dtypes=True)
