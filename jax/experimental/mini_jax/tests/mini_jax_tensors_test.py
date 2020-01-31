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


from jax import test_util as jtu
from jax.experimental import mini_jax as mj
from jax.experimental.mini_jax.mini_jax import const_like
from jax import api
from jax import numpy as jnp
from jax import lax

import numpy as np

from jax.config import config
config.parse_flags_with_absl()
FLAGS = config.FLAGS

class TensorsTest(jtu.JaxTestCase):

  def test_simple(self):
    shape = (3, 2)
    def func(x):
      y = np.arange(6, dtype=np.float32).reshape(shape)
      return x * y + y - y

    ones = np.ones(shape, dtype=np.float32)
    self.assertMultiLineStrippedEqual("""
{lambda v0.
  # v0: float[3,2]
  n0 = mul v0 array([[0., 1.],
                  [2., 3.],
                  [4., 5.]], dtype=float32)
  n1 = add n0 array([[0., 1.],
                  [2., 3.],
                  [4., 5.]], dtype=float32)
  n2 = sub n1 array([[0., 1.],
                  [2., 3.],
                  [4., 5.]], dtype=float32)
  in n2}
  """,
         str(mj.trace(func)(ones)))

    self.assertAllClose(np.arange(6, dtype=np.float32).reshape(shape),
                        func(ones),
                        check_dtypes=True)

    self.assertAllClose(np.arange(6, dtype=np.float32).reshape(shape),
                        mj.jit(func)(ones),
                        check_dtypes=True)

  def test_cond(self):
    shape = (3, 2)
    def func(x_test, x):
      y = x * const_like(2., x)

      return mj.Ops.cond_ge(x_test,
                            lambda tv: tv + const_like(3., tv),
                            (y * const_like(4., y),),
                            lambda fv: x * const_like(13., x),
                            (x * const_like(14., x),))

    def func_equiv(x_test, x):
      if x_test >= 0.:
        return x * 2. * 4. + 3.
      else:
        return x * 13.

    def func_jvp_equiv(x_test, x, x_test_tan, x_tan):
      if x_test >= 0.:
        return (func_equiv(x_test, x), x_tan * 2. * 4.)
      else:
        return (func_equiv(x_test, x), x_tan * 13.)

    def func_grad_equiv(x_test, x):
      if x_test >= 0.:
        return 0., const_like(8., x)
      else:
        return 0., const_like(13., x)


    ones = np.ones(shape, dtype=np.float32)

    self.assertMultiLineStrippedEqual("""
{lambda v0 v1.
  # v0: float, v1: float[3,2]
  n0 = mul v1 array([[2., 2.],
                  [2., 2.],
                  [2., 2.]], dtype=float32)
  n1 = mul n0 array([[4., 4.],
                  [4., 4.],
                  [4., 4.]], dtype=float32)
  n2 = mul v1 array([[14., 14.],
                  [14., 14.],
                  [14., 14.]], dtype=float32)
  n3 = cond_ge[ false_args=('n2', v1)
                false_func={lambda v3 v1.
                             # v3: float[3,2], v1: float[3,2]
                             n0 = mul v1 array([[13., 13.],
                                             [13., 13.],
                                             [13., 13.]], dtype=float32)
                             in n0}
                pred_arg=v0
                true_args=('n1',)
                true_func={lambda v2.
                            # v2: float[3,2]
                            n0 = add v2 array([[3., 3.],
                                            [3., 3.],
                                            [3., 3.]], dtype=float32)
                            in n0} ] 
  in n3}
          """, str(mj.trace(func)(3., ones).pp()))

    self.assertAllClose(func_equiv(3., ones),
                        func(3., ones),
                        check_dtypes=True)
    self.assertAllClose(func_equiv(3., ones),
                        mj.jit(func)(3., ones),
                        check_dtypes=True)

    jvp_args_true = (3., ones, 0., 2. * ones)
    self.assertMultiLineStrippedEqual("""
{lambda v0 v1 v2 v3.
  # v0: float, v1: float[3,2], v2: float, v3: float[3,2]
  n0 = mul v1 array([[2., 2.],
                  [2., 2.],
                  [2., 2.]], dtype=float32)
  n1 = mul n0 array([[4., 4.],
                  [4., 4.],
                  [4., 4.]], dtype=float32)
  n2 = mul v3 array([[2., 2.],
                  [2., 2.],
                  [2., 2.]], dtype=float32)
  n3 = mul n2 array([[4., 4.],
                  [4., 4.],
                  [4., 4.]], dtype=float32)
  n4 = mul v1 array([[14., 14.],
                  [14., 14.],
                  [14., 14.]], dtype=float32)
  n5 = mul v3 array([[14., 14.],
                  [14., 14.],
                  [14., 14.]], dtype=float32)
  n6 = cond_ge[ false_args=('n4', v1, 'n5', v3)
                false_func={lambda v6 v7 v8 v9.
                             # v6: float[3,2], v7: float[3,2], v8: float[3,2], v9: float[3,2]
                             n0 = mul v7 array([[13., 13.],
                                             [13., 13.],
                                             [13., 13.]], dtype=float32)
                             n1 = mul v9 array([[13., 13.],
                                             [13., 13.],
                                             [13., 13.]], dtype=float32)
                             in (n0 n1,)}
                pred_arg=v0
                true_args=('n1', 'n3')
                true_func={lambda v4 v5.
                            # v4: float[3,2], v5: float[3,2]
                            n0 = add v4 array([[3., 3.],
                                            [3., 3.],
                                            [3., 3.]], dtype=float32)
                            in (n0 v5,)} ] 
  n7 = proj[ idx=0 ] n6
  n8 = proj[ idx=1 ] n6
  in (n7 n8,)}
""",
                                      str(mj.trace(mj.jvp(func))(*jvp_args_true).pp()))
    self.assertAllClose(func_jvp_equiv(*jvp_args_true),
                        mj.jvp(func)(*jvp_args_true),
                        check_dtypes=True)

    jvp_args_false = (-3., ones, 0., 2. * ones)
    self.assertAllClose(func_jvp_equiv(*jvp_args_false),
                    mj.jvp(func)(*jvp_args_false),
                    check_dtypes=True)

    self.assertMultiLineStrippedEqual("""
{lambda v0 v1.
  # v0: float, v1: float[3,2]
  n0 = mul v1 array([[2., 2.],
                  [2., 2.],
                  [2., 2.]], dtype=float32)
  n1 = mul n0 array([[4., 4.],
                  [4., 4.],
                  [4., 4.]], dtype=float32)
  n2 = mul v1 array([[14., 14.],
                  [14., 14.],
                  [14., 14.]], dtype=float32)
  n3 = cond_ge[ false_args=('n2', v1, array([[1., 1.],
                                  [1., 1.],
                                  [1., 1.]], dtype=float32))
                false_func={lambda v4 v5 v6.
                             # v4: float[3,2], v5: float[3,2], v6: float[3,2]
                             n0 = mul v6 array([[13., 13.],
                                             [13., 13.],
                                             [13., 13.]], dtype=float32)
                             in (array([[0., 0.],
                                        [0., 0.],
                                        [0., 0.]], dtype=float32) array([[0., 0.],
                                                                         [0., 0.],
                                                                         [0., 0.]], dtype=float32) n0,)}
                pred_arg=v0
                true_args=('n1', array([[1., 1.],
                                 [1., 1.],
                                 [1., 1.]], dtype=float32))
                true_func={lambda v2 v3.
                            # v2: float[3,2], v3: float[3,2]
                            in (v3 array([[0., 0.],
                                          [0., 0.],
                                          [0., 0.]], dtype=float32) array([[0., 0.],
                                                                           [0., 0.],
                                                                           [0., 0.]], dtype=float32),)} ] 
  n4 = proj[ idx=2 ] n3
  n5 = proj[ idx=0 ] n3
  n6 = mul n5 array([[4., 4.],
                  [4., 4.],
                  [4., 4.]], dtype=float32)
  n7 = mul n6 array([[2., 2.],
                  [2., 2.],
                  [2., 2.]], dtype=float32)
  n8 = add n4 n7
  n9 = proj[ idx=1 ] n3
  n10 = mul n9 array([[14., 14.],
                   [14., 14.],
                   [14., 14.]], dtype=float32)
  n11 = add n8 n10
  in (0.0 n11,)}
""",
                                      str(mj.trace(mj.grad(func))(3., ones).pp()))
    self.assertAllClose(func_grad_equiv(3., ones), mj.grad(func)(3., ones),
                        check_dtypes=True)


  def testShape(self):
    ones = np.ones((3, 4), dtype=np.float32)
    def func_tensor(x):
      self.assertEqual((3, 4), x.shape)
      self.assertEqual((3, 4), np.shape(x))
      return x
    mj.trace(func_tensor)(ones)

    def func_scalar(x):
      self.assertEqual((), x.shape)
      self.assertEqual((), np.shape(x))
      return x
    mj.trace(func_scalar)(1.)


class ComparativeJaxTest(jtu.JaxTestCase):
  """Tests using regular JAX, for comparison."""


  def test_vmap_constant(self):
    def func(x):
      return (x, 2.)

    res = api.vmap(func)(np.arange(6, dtype=np.float))
    self.assertAllClose((np.arange(6, dtype=np.float),
                         2. * np.ones((6,), dtype=np.float)),
                        res, check_dtypes=True)

  def test_vmap_nested(self):
    def diff(x, y):
      return x - y

    xs = np.array([10., 11., 12., 13.])
    ys = np.array([0., 1., 2.])
    def all_dist(xs, ys):
      return api.vmap(lambda x: api.vmap(lambda y: diff(x, y))(ys))(xs)
    print(api.make_jaxpr(all_dist)(xs, ys))
    res = all_dist(xs, ys)
    self.assertAllClose(np.array([
      [10., 9., 8.,],
      [11., 10., 9.],
      [12., 11., 10.],
      [13., 12., 11.]
    ]), res, check_dtypes=True)

  def test_numpy_expand(self):
    a = np.array([[0., 1.], [2., 3.], [4., 5.]])
    res1 = np.broadcast_to(np.expand_dims(a, axis=1), (3, 5, 2))
    print(res1)

  def test_numpy_mul(self):
    a = np.array([[0., 1.], [2., 3.], [4., 5.]], dtype=np.float32)
    res1 = a * a
    print(res1)

  def test_numpy_overload(self):
    a = np.array([[0., 1.], [2., 3.], [4., 5.]])
    def func(x):
      y = a + 3.
      return a + x
    print(mj.trace(func)(a))
    print(api.make_jaxpr(func)(a))

  def test_layer(self):
    def layer(weights, bias, inputs):
      # weights: f32[3,5]; bias: f32[3]; inputs: f32[5]
      z = jnp.dot(weights, inputs) + bias
      # z : f32[3]
      return z

    def layer_loops(weights, bias, inputs):
      M, N = weights.shape
      assert bias.shape == (M,)
      assert inputs.shape == (N,)
      res = np.zeros((M,), dtype=np.float32)
      for i in range(M):
        acc = 0.
        for j in range(N):
          acc += weights[i, j] * inputs[j]
        res[i] = acc + bias[i]
      return res


    def layer_batched_naive(weights, bias, vinputs):
      M, N = weights.shape
      assert bias.shape == (M,)
      B, N1 = vinputs.shape
      assert N == N1
      # weights: f32[3,5]; bias: f32[3]; inputs: f32[7,5]
      return np.stack([layer(weights, bias, inp) for inp in vinputs])

    def layer_batched_manual(weights, bias, vinputs):
      O, I = weights.shape
      assert bias.shape == (O,)
      B, I1 = vinputs.shape
      assert I == I1
      vz = jnp.dot(vinputs, weights.T) + jnp.broadcast_to(bias, (B, O))
      return vz

    def layer_batched_vmap(weights, bias, vinputs):
      return api.vmap(lambda inputs: layer(weights, bias, inputs))(vinputs)

    from jax import random
    key = random.PRNGKey(0)
    x = random.normal(key, (10,))

    weights = random.normal(key, (3, 5))
    inputs = random.normal(key, (5,))
    bias = random.normal(key, (3,))

    # Manually batch
    print("layer", layer(weights, bias, inputs))
    print("layer_loops", layer_loops(weights, bias, inputs))

    vinputs = random.normal(key, (7, 5,))
    print("layer_batched_naive", layer_batched_naive(weights, bias, vinputs))
    print("layer_batched_manual", layer_batched_manual(weights, bias, vinputs))
    print("layer_batched_vmap", layer_batched_vmap(weights, bias, vinputs))

    res = np.dot(random.normal(key, (3,3,4,5)), random.normal(key, (7, 9, 5, 8)))
    print(res.shape)