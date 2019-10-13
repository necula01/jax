
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import collections
from functools import partial
import unittest
import warnings

from absl.testing import absltest
import numpy as onp
import six

if six.PY3:
  import concurrent.futures

import jax
import jax.numpy as jnp
import numpy as np

# from jax import jit, grad, device_put, jacfwd, jacrev, hessian
# from jax import api, lax
# from jax.core import Primitive
# from jax.interpreters import ad
# from jax.interpreters.xla import DeviceArray
# from jax.abstract_arrays import concretization_err_msg
# from jax.lib import xla_bridge as xb
from jax import test_util as jtu
# from jax import tree_util

from jax.config import config
config.parse_flags_with_absl()
FLAGS = config.FLAGS

# From CL/266377511
class DeepFried(object):
  """Corresponds to the deepfried.py module."""
  def conv2d(self, x, w, **kw):
    """Convolves `x` with `w` using some default options."""
    # Most of these are the defaults already, but I leave them here for
    # reference and ease of fiddling.
    kw.setdefault("window_strides", (1, 1))  # Not default.
    kw.setdefault("padding", "VALID")  # Not default.
    kw.setdefault("lhs_dilation", (1, 1))  # Default, image dilation.
    kw.setdefault("rhs_dilation", (1, 1))  # Default, filter dilation.
    kw.setdefault("feature_group_count", 1)  # Default.
    kw.setdefault("precision", None)  # Default.

    # This one is not the default, but chosen this way to be consistent with
    # the order used in TF. And, according to chat, is most efficient on TPU.
    kw.setdefault("dimension_numbers", ("NHWC", "OIHW", "NHWC"))

    return jax.lax.conv_general_dilated(x, w, **kw)

  def softmax_cls(self, x, axis=-1):
    ex = jnp.exp(x - jnp.max(x, axis=axis, keepdims=True))
    return ex / jnp.sum(ex, axis=axis, keepdims=True)

df = DeepFried()

class Render(object):
  """Corresponds to the render.py module"""


  def render1(self, presence, types, templates, background):
    # Extract and verify input shapes. pylint: disable=invalid-name
    # In the comments, I use TH->h, TW->w ("small h") for clarity.
    inf = 1e5  # Such that inf*0 = 0 and not nan, but fits into float32 with +1

    B, H, W = presence.shape
    B2, H2, W2, T = types.shape
    assert B == B2 and H == H2 and W == W2, "Bad types shape"
    T, TH, TW, C = templates.shape
    B3, H3, W3, C2 = background.shape
    assert B3 == B2 and H3 == H2 and W3 == W2 and C2 == C, "Bad background shape"

    # Shortcut for padding.
    dont = (0, 0, 0)

    # Compute `a`: of each output pixel, which input pixels influence it.
    # shape will be (H+h-1)(W+w-1)hw
    w = np.eye(TH*TW).reshape(TH*TW, 1, TH, TW)  # OIHW.
    # aka "FULL" padding in old-school frameworks.
    x = jax.lax.pad(presence, -inf, (dont, (TH-1, TH-1, 0), (TW-1, TW-1, 0)))
    a_logits = df.conv2d(x[..., None], w, padding="VALID")

    # Append a default background-claim of 0.0 (logits) via 0-padding channels.
    a_logits = jax.lax.pad(a_logits, 0.0, (dont, dont, dont, (0, 1, 0)))
    a_combined = df.softmax_cls(a_logits, axis=-1)
    a_bg = a_combined[..., -1]
    a = a_combined[..., :-1].reshape(B, H+TH-1, W+TW-1, TH, TW)

    # We now have the background probabilities in `a_bg` as B(H+h-1)(W+w-1)
    # and the influence probabilities in `a` as B(H+h-1)(W+w-1)hw

    # What follows, is something very similar to `a`, but not about
    # presence/claim anymore, but this time about which template to use, hence
    # we call this one `t`, and the shape will be (H+h-1)(W+w-1)Thw.
    x = df.softmax_cls(types, axis=-1)  # Softmax along classes (T).
    w = np.eye(T*TW*TH).reshape(T*TW*TH, T, TH, TW)  # OIHW.
    t = df.conv2d(x, w, padding=[(TH-1, TH-1), (TW-1, TW-1)])
    t = t.reshape(B, H+TH-1, W+TW-1, T, TH, TW)

    # We now have both `a` (claims of output pixels by objects) and
    # `t` (similar, for each individual template), next, combine them.

    # First, we mix the templates together according to the type compositions
    # in `t`. Note that this still ignores `a` ("presence").
    # Combine: (H+h-1)(W+w-1)Thw  and  ThwC  along  T  to get  (H+h-1)(W+w-1)hwC
    t_mixed = t[..., None] * templates[None, None, ...]
    t_mixed = jnp.sum(t_mixed, axis=-4)

    # Second, combine this with the presences `a` to collect only those pixels
    # which should actually be drawn, according to the claims.
    # Combine: (H+h-1)(W+w-1)hwC  and  (H+h-1)(W+w-1)hw  to get  (H+h-1)(W+w-1)C
    p = jnp.sum(t_mixed * a[..., None], axis=[-2, -3])

    # Finally, remove the padding to get HWC, but note a[0:-0] needs special case.
    none_if_zero = lambda x: (x if x != 0 else None)
    Hlo, Hhi = TH//2, none_if_zero(-(TH//2))
    Wlo, Whi = TW//2, none_if_zero(-(TW//2))
    p = p[:, Hlo:Hhi, Wlo:Whi]

    # Combine with the given background, which is also HWC,
    # using the background-claim mask after removing its padding.
    a_bg = a_bg[:, Hlo:Hhi, Wlo:Whi, None]
    img = p * (1.0 - a_bg) + background * a_bg

    # return a bit more than just the image, for debug purposes.
    return img, (a, a_bg, t)


class RenderTest(jtu.JaxTestCase):

  def test_render1(self):
    # 1. Create the test data.
    inf = 1e5  # Such that inf*0 = 0 and not nan, but fits into float32 with +1
               # and we get 1/0 when softmaxing with logits using this.

    H, W = 1, 5  # pylint: disable=invalid-name
    # Template types:
    T = 2  # pylint: disable=invalid-name
    # We assume an image 1x5 with two template instances aligned as follows:
    #  Pixel: 0  1  2  3  4
    #  T1:    -  T1 T1 T1 -
    #  T2:    -  -   - T2 T2
    #
    # Presence logits: HW
    # For each pixel, we use 'inf' if the image contains an instance
    # of some template centered at that pixel. The 'types' tensor below
    # says which template it is.
    # We use -inf for a pixel if it does not contain the center of
    # a template instance.
    presence = np.array([-inf, -inf, inf, -inf, inf], np.float32).reshape(H, W)

    # Type logits: HWT
    # For each pixel, and each template, whether the pixel contains the center
    # of that template (+inf). And -inf ???
    types = np.array([[0., 0., +inf, 0., -inf],
                      [0., 0., -inf, 0., +inf]], np.float32).T.reshape(H, W, T)

    # Templates: ThwC
    TH, TW, C = 1, 3, 1  # pylint: disable=invalid-name
    # NOTE: there is currently a mirroring happening (conv vs filt). If there
    # was no mirroring, the first one is what we'd want to use for this test,
    # but in order to account for the mirroring, using the second one for now.
    # (maybe we actually shouldn't do mirroring in the renderer since it might
    # be counter-intuitive? Let's change that later.)
    #templates = np.array([[0.3, 0.5, 1.0],
    #                      [3.0, 5.0, 10.0]], np.float32).reshape(T, TH, TW, C)
    templates = np.array([[1.0, 0.5, 0.3],
                          [10.0, 5.0, 3.0]], np.float32).reshape(T, TH, TW, C)

    # Background: HWC
    background = np.full((H, W, C), 9.9, np.float32)

    # Desired painting: HWC
    desired = np.array([9.9,
                        1.0*0.3 + 0.0*3.0,
                        1.0*0.5 + 0.0*5.0,
                        0.5*1.0 + 0.5*3.0,
                        0.0*1.0 + 1.0*5.0], np.float32).reshape(H, W, C)

    render = Render()
    # Add the batch-dimension where necessary
    actual, extra = render.render1(
        presence[None], types[None], templates, background[None])
    print(actual, extra)
    np.testing.assert_allclose(actual[0], desired)