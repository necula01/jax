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
from typing import Callable, List, Optional, Tuple, Sequence

from jax import test_util as jtu
from jax.experimental import mini_jax as mj
from jax.experimental.mini_jax.tests import mini_jax_testing_examples as testex

from jax.experimental.mini_jax.mini_jax import (
  const_like, zero_like, unzip, Globals
)
from jax.experimental.mini_jax.mini_jax_operators import (
  broadcast_value, broadcast_values
)
from jax.config import config

import numpy as np

config.parse_flags_with_absl()
FLAGS = config.FLAGS


class VmapTest(jtu.JaxTestCase):
  def setUp(self):
    Globals.reset()

  def tearDown(self) -> None:
    Globals.reset()

  def helperTestVmap(self, func: Callable, args_has_batch: Sequence[bool],
                     vargs: Sequence,
                     expected_trace: Optional[str] = None):
    """Test the vmap of the function, against an iteration.

    Args:
      func: a function to vmap
      batched: a tuple with booleans denoting whether to batch arg_i
      vargs: a tuple of a batched arguments
      expected_trace: the expected trace string for the vmapped function
    """
    tr = mj.trace(mj.vmap(func, args_has_batch=args_has_batch))(*vargs)
    if expected_trace is not None:
      self.assertMultiLineStrippedEqual(expected_trace, str(tr.pp()))
    batch_sizes = {a.shape[0]
                   for a, a_hasb in zip(vargs, args_has_batch) if a_hasb}
    assert len(batch_sizes) == 1, "Found multiple batch sizes"
    batch_size, = batch_sizes
    out = mj.vmap(func, args_has_batch=args_has_batch)(*vargs)
    all_bargs, _ = broadcast_values(batch_size, vargs, args_has_batch)
    all_res = list(map(func, *all_bargs))  # type: ignore[arg-type]
    # If the function returns a tuple
    if isinstance(all_res[0], tuple):
      all_res_stacked = tuple(np.stack(one_res) for one_res in unzip(all_res))
    else:
      all_res_stacked = np.stack(all_res)
    self.assertAllClose(all_res_stacked, out, check_dtypes=True)

  def test_vmap_all_examples(self):
    """Test VMAP for all examples, numerically."""
    for ex in testex.iterate_examples():
      args = ex.args
      args_cons = ex.args_constraints()
      if not all(a_cons is None for a_cons in args_cons):
        print(f"Skip {ex.name} because it has arg constraints")
        continue
      if not all(not np.shape(a) for a in args):
        print(f"Skip {ex.name} because it has tensor args")
        continue
      # For each example vectorize all combination of args
      for i in range(2 ** len(args)):
        # Which args to batch?
        args_has_batch = tuple(i & (2 ** j) == 0 for j in range(len(args)))
        args_b = tuple(a if not a_hasb else np.arange(3.)
                       for a, a_hasb in zip(args, args_has_batch))
        name = f"{ex.name}_{args_has_batch}"
        if all(not a_hasb for a_hasb in args_has_batch):
          print(f"Skip {args_has_batch} to ensure at least one arg must be batched")
          continue

        #if name != "WhereOp0_(True,)": continue
        print(f"Testing VMAP {name}")
        self.helperTestVmap(ex.func, args_has_batch, args_b)


  def testAdd0(self):
    self.helperTestVmap(testex.Add0().get_lambda(), (True, True), (np.arange(3.), np.ones(3)),
                        expected_trace="""
{lambda v0 v1.
  # v0: float[3], v1: float[3]
  n0 = add v0 v1
  in n0}
  """)

  def testAdd0Unbatched1(self):
    self.helperTestVmap(testex.Add0().get_lambda(), (True, False), (np.arange(3.), 5.),
                        expected_trace="""
{lambda v0 v1.
  # v0: float[3], v1: float
  n0 = Op[bcast][ dim=0
                  dim_sz=3 ] v1
  n1 = proj[ idx=0 ] n0
  n2 = add v0 n1
  in n2}
    """)

  def testAddUnbatchedResult(self):
    self.helperTestVmap(testex.MultipleResults().get_lambda(), (True, False), (np.arange(3.), 5.),
                        expected_trace="""
{lambda v0 v1.
  # v0: float[3], v1: float
  n0 = Op[bcast][ dim=0
                  dim_sz=3 ] v1
  n1 = proj[ idx=0 ] n0
  n2 = mul v0 n1
  n3 = mul v1 3.0
  n4 = Op[bcast][ dim=0
                  dim_sz=3 ] n3
  n5 = proj[ idx=0 ] n4
  in (n2 n5,)}
      """)

  def testArithmetic(self):
    self.helperTestVmap(testex.Arithmetic().get_lambda(), (True, True), (np.arange(3.), np.ones(3)),
                        expected_trace="""
{lambda v0 v1.
  # v0: float[3], v1: float[3]
  n0 = mul v0 array([3., 3., 3.])
  n1 = add n0 v1
  n2 = sub n1 v0
  n3 = pow[ pow=3 ] v0
  n4 = add n2 n3
  n5 = sub n4 v1
  n6 = add n5 array([1., 1., 1.])
  in n6}
  """)

  def testWithEnv(self):
    def func(x):
      def inner(z):
        return x + z

      return mj.vmap(inner, args_has_batch=(True,))(np.arange(3.))

    self.assertMultiLineStrippedEqual("""
{lambda v0.
  # v0: float
  n0 = Op[bcast][ dim=0
                  dim_sz=3 ] v0
  n1 = proj[ idx=0 ] n0
  n2 = add n1 array([0., 1., 2.])
  in n2}""", str(mj.trace(func)(5.).pp()))

  def testDoubleVmap(self):
    def add_all_pairs(xb, yb):
      self.assertEqual((3,), np.shape(xb))
      self.assertEqual((2,), np.shape(yb))

      def add_xb_to_y(y):
        def add_x_to_y(x):
          return x + y

        return mj.vmap(add_x_to_y, args_has_batch=(True,))(xb)

      return mj.vmap(add_xb_to_y, args_has_batch=(True,))(yb)

    self.assertMultiLineStrippedEqual("""
{lambda v0 v1.
  # v0: float[3], v1: float[2]
  n0 = Op[bcast][ dim=0
                  dim_sz=2 ] v0
  n1 = proj[ idx=0 ] n0
  n2 = Op[bcast][ dim=1
                  dim_sz=3 ] v1
  n3 = proj[ idx=0 ] n2
  n4 = add n1 n3
  in n4}""", str(mj.trace(add_all_pairs)(np.arange(3.), np.array([10., 20.])).pp()))
    self.assertAllClose(np.array([[10., 11., 12.], [20., 21., 22.], ]),
                        add_all_pairs(np.arange(3.), np.array([10., 20.])),
                        check_dtypes=True)

  def testVmapJit(self):
    self.helperTestVmap(testex.InnerJit0().get_lambda(), (True,), (np.arange(3.),),
                        expected_trace="""
{lambda v0.
  # v0: float[3]
  n0 = add v0 array([2., 2., 2.])
  n1 = jit_call[ func={lambda v3.
                        # v3: float[3]
                        n0 = mul v3 array([2., 2., 2.])
                        in n0} ] n0
  in n1}
      """)

  def testCond0PredScalar(self):
    self.helperTestVmap(testex.Conditional0().get_lambda(), (False, True,), (5., np.arange(3.),),
                        expected_trace="""
{lambda v0 v1.
  # v0: float, v1: float[3]
  n0 = sub v0 2.0
  n1 = add v0 5.0
  n2 = cond_ge[ args=('n1', v1)
                false_func={lambda v13 v14.
                             # v13: float, v14: float[3]
                             n0 = add v13 4.0
                             n1 = Op[bcast][ dim=0
                                             dim_sz=3 ] n0
                             n2 = proj[ idx=0 ] n1
                             in n2}
                pred_arg=n0
                true_func={lambda v11 v12.
                            # v11: float, v12: float[3]
                            n0 = Op[bcast][ dim=0
                                            dim_sz=3 ] v11
                            n1 = proj[ idx=0 ] n0
                            n2 = add n1 v12
                            in n2} ] 
  in n2}
        """)

  def testCond0PredBatched(self):
    self.helperTestVmap(testex.Conditional0().get_lambda(), (True, False), (np.arange(3.), 5.),
                        expected_trace="""
{lambda v0 v1.
  # v0: float[3], v1: float
  n0 = sub v0 array([2., 2., 2.])
  n1 = add v0 array([5., 5., 5.])
  n2 = jit_call[ func={lambda v7 v8.
                        # v7: float[3], v8: float
                        n0 = Op[bcast][ dim=0
                                        dim_sz=3 ] v8
                        n1 = proj[ idx=0 ] n0
                        n2 = add v7 n1
                        in n2} ] n1 v1
  n3 = jit_call[ func={lambda v9 v10.
                        # v9: float[3], v10: float
                        n0 = add v9 array([4., 4., 4.])
                        in n0} ] n1 v1
  n4 = Op[where_ge] n0 n2 n3
  n5 = proj[ idx=0 ] n4
  in n5}
          """)

  def testCondTupleArg(self):
    self.helperTestVmap(testex.ConditionalTupleArg().get_lambda(), (True,), (np.arange(3.),),
                        expected_trace="""
{lambda v0.
  # v0: float[3]
  n0 = sub v0 array([2., 2., 2.])
  n1 = add v0 array([4., 4., 4.])
  n2 = add v0 array([5., 5., 5.])
  n3 = jit_call[ func={lambda v6 v7.
                        # v6: float[3], v7: float[3]
                        n0 = add v6 array([3., 3., 3.])
                        in n0} ] n1 n2
  n4 = jit_call[ func={lambda v8 v9.
                        # v8: float[3], v9: float[3]
                        n0 = mul v9 array([3., 3., 3.])
                        n1 = add v8 n0
                        in n1} ] n1 n2
  n5 = Op[where_ge] n0 n3 n4
  n6 = proj[ idx=0 ] n5
  in n6}
            """)

  def testVmapCondPredBatchedMultipleRes(self):
    self.helperTestVmap(testex.ConditionalTupleRes().get_lambda(), (True, False), (np.arange(3.), 5.),
                        expected_trace="""
{lambda v0 v1.
  # v0: float[3], v1: float
  n0 = sub v0 array([2., 2., 2.])
  n1 = add v0 array([4., 4., 4.])
  n2 = add v0 array([5., 5., 5.])
  n3 = jit_call[ func={lambda v10 v11 v12 v13.
                        # v10: float[3], v11: float[3], v12: float, v13: float
                        n0 = Op[bcast][ dim=0
                                        dim_sz=3 ] v12
                        n1 = proj[ idx=0 ] n0
                        n2 = add v10 n1
                        n3 = Op[bcast][ dim=0
                                        dim_sz=3 ] v12
                        n4 = proj[ idx=0 ] n3
                        in (n2 n4,)} ] n1 n2 v1 v1
  n4 = proj[ idx=0 ] n3
  n5 = jit_call[ func={lambda v14 v15 v16 v17.
                        # v14: float[3], v15: float[3], v16: float, v17: float
                        n0 = mul v15 array([3., 3., 3.])
                        n1 = add v14 n0
                        n2 = add v17 6.0
                        n3 = Op[bcast][ dim=0
                                        dim_sz=3 ] n2
                        n4 = proj[ idx=0 ] n3
                        in (n1 n4,)} ] n1 n2 v1 v1
  n6 = proj[ idx=0 ] n5
  n7 = Op[where_ge] n0 n4 n6
  n8 = proj[ idx=0 ] n7
  n9 = proj[ idx=1 ] n3
  n10 = proj[ idx=1 ] n5
  n11 = Op[where_ge] n0 n9 n10
  n12 = proj[ idx=0 ] n11
  in (n8 n12,)}
          """)


if __name__ == "__main__":
  absltest.main()
