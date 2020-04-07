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

from absl.testing import absltest
import os
from typing import List


from jax import test_util as jtu
from jax.experimental import mini_jax as mj
from jax.experimental.mini_jax import mini_jax_callback
from jax.config import config

import numpy as np

config.parse_flags_with_absl()
FLAGS = config.FLAGS


class CallbackTest(jtu.JaxTestCase):
  def setUp(self):
    # Setup an accumulator and a callback
    self.acc: List[str] = []  # tuples with the arguments, ending with a tuple of transformations
    def acc_callback_fun(*args, transforms=()):
      self.acc.append(tuple([*args, transforms]))
    self.acc_callback = mj.callback(acc_callback_fun)
    mini_jax_callback._reset_counters()

  def testEval(self):
    def fun(x):
      y = x * x
      z, = self.acc_callback(y)
      return z * z

    self.assertAllClose(4. ** 4, fun(4.), check_dtypes=True)
    self.assertEqual([(4. ** 2, ())], self.acc)

  def testEvalConst(self):
    def fun(x):
      y = x * x
      z, = self.acc_callback(3.)  # Called even for constants
      return z * y

    self.assertAllClose(48., fun(4.), check_dtypes=True)
    self.assertEqual([(3., ())], self.acc)

  def testEvalCond(self):
    def multiply_if_positive(x):
      return 3. * mj.Ops.cond_ge(x,
          lambda x: self.acc_callback(x * 2.)[0],
          lambda _: 0., (x,))

    self.assertAllClose(4. * 6, multiply_if_positive(4.), check_dtypes=True)
    self.assertEqual([(4. * 2, ())], self.acc)


  def testTrace(self):
    def fun(x):
      y = x * x
      z, = self.acc_callback(y)
      return z * z

    self.assertMultiLineStrippedEqual("""
{lambda v0.
  # v0: float
  n0 = mul v0 v0
  n1 = Op[callback][ func=cb0
                     transforms=() ] n0
  n2 = proj[ idx=0 ] n1
  n3 = mul n2 n2
  in n3}
          """, str(mj.trace(fun)(4.).pp()))
    self.assertEqual([], self.acc)

  def testJit(self):
    os.environ["MINI_JAX_LOG_COMPILES"] = "1"
    def fun(x):
      y = x * x
      z, = self.acc_callback(y)
      return z * z

    res = mj.jit(fun)(3.)
    self.assertEqual(3. ** 4, res)
    self.assertEqual([(3. ** 2, ())], self.acc)

  def testJvp(self):
    os.environ["MINI_JAX_LOG_COMPILES"] = "1"
    def fun(x):
      y = x * x
      z, = self.acc_callback(y)
      return z * z

    self.assertMultiLineStrippedEqual("""
{lambda v0 v1.
  # v0: float, v1: float
  n0 = mul v0 v0
  n1 = mul v1 v0
  n2 = mul v0 v1
  n3 = add n1 n2
  n4 = Op[callback][ func=cb0
                     transforms=('jvp',) ] n0 n3
  n5 = proj[ idx=0 ] n4
  n6 = mul n5 n5
  n7 = proj[ idx=1 ] n4
  n8 = mul n7 n5
  n9 = mul n5 n7
  n10 = add n8 n9
  in (n6 n10,)}
              """, str(mj.trace(mj.jvp(fun))(5., 6.).pp()))
    self.assertEqual((5. ** 4, 6. * 4. * 5. ** 3), mj.jvp(fun)(5., 6.))
    # We logged the value of `y` and its tangent
    self.assertEqual([(5. ** 2, 6. * 2. * 5., ("jvp",))], self.acc)

  def testGrad(self):
    os.environ["MINI_JAX_LOG_COMPILES"] = "1"
    def fun(x):
      y = x * x
      z, = self.acc_callback(y)
      # The result of the callback is needed in the grad computation
      return z * z

    self.assertMultiLineStrippedEqual("""
{lambda v0.
  # v0: float
  n0 = mul v0 v0
  n1 = Op[callback][ func=cb0
                     transforms=() ] n0
  n2 = proj[ idx=0 ] n1
  n3 = add n2 n2
  n4 = Op[callback][ func=cb0
                     transforms=('vjp',) ] n3
  n5 = proj[ idx=0 ] n4
  n6 = mul n5 v0
  n7 = mul v0 n5
  n8 = add n6 n7
  in n8}
              """, str(mj.trace(mj.grad(fun))(5.).pp()))
    self.assertEqual((4. * 5. ** 3), mj.grad(fun)(5.))
    # We logged the value of `y` on the primal computation, then the z_adjoint
    self.assertEqual([(5. ** 2, ()),
                      (2. * 5. ** 2, ("vjp",))], self.acc)

    self.acc = []
    # Second gradient
    self.assertMultiLineStrippedEqual("""
{lambda v0.
  # v0: float
  n0 = mul v0 v0
  n1 = Op[callback][ func=cb0
                     transforms=() ] n0
  n2 = proj[ idx=0 ] n1
  n3 = add n2 n2
  n4 = Op[callback][ func=cb0
                     transforms=('vjp',) ] n3
  n5 = proj[ idx=0 ] n4
  n6 = add n5 n5
  n7 = add v0 v0
  n8 = Op[callback][ func=cb0
                     transforms=('vjp', 'vjp') ] n7
  n9 = proj[ idx=0 ] n8
  n10 = add n9 n9
  n11 = Op[callback][ func=cb0
                      transforms=('vjp',) ] n10
  n12 = proj[ idx=0 ] n11
  n13 = mul n12 v0
  n14 = add n6 n13
  n15 = mul v0 n12
  n16 = add n14 n15
  in n16}
                  """, str(mj.trace(mj.grad(mj.grad(fun)))(5.).pp()))

    self.assertEqual((4. * 3. * 5. ** 2), mj.grad(mj.grad(fun))(5.))
    # We logged the value of `y` on the primal computation: 5 * 5
    # We logger the z_adjoint
    self.assertCountEqual([(25., ()),
                           (50., ("vjp",)),
                           (10., ("vjp", "vjp")),
                           (20., ("vjp",))], self.acc)

  def testGrad_linear(self):
    os.environ["MINI_JAX_LOG_COMPILES"] = "1"
    def fun(x):
      y = x * x
      z, = self.acc_callback(y)
      # The result of the callback is NOT needed in the grad computation
      return z + z

    self.assertMultiLineStrippedEqual("""
{lambda v0.
  # v0: float
  n0 = Op[callback][ func=cb0
                     transforms=('vjp',) ] 2.0
  n1 = proj[ idx=0 ] n0
  n2 = mul n1 v0
  n3 = mul v0 n1
  n4 = add n2 n3
  in n4}
              """, str(mj.trace(mj.grad(fun))(5.).pp()))
    self.assertEqual((4. * 5.), mj.grad(fun)(5.))
    # We logged the value of `y` on the primal computation, then the z_adjoint
    self.assertEqual([(2., ("vjp",))], self.acc)

if __name__ == "__main__":
  absltest.main()