
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
    """
    Render template instances into a batch of images.

    The dimensions are: batch of B images, of size (H,W) with C channels. There are T templates,
    each described by an image (TH,TW) with C channels.

    Assumptions: At each pixel there is at most one template instance center.

    :param presence: a tensor[B, H, W], for each batch and each pixel, an entry +inf if at that
      pixel we have the center of some template instantion. And -inf if there is no template
      whose center is at the pixel. TODO: I think this can be derived from "types".
    :param types: a tensor[B, H, W, T], for each batch and each pixel and each template type,
      we have +inf if there is an instance of that template type at that pixel. We put -inf
      for all the other templates at that pixel.
    :param templates: a tensor[T, TH, TW, C], for each template type, its image.
    :param background: a tensor[B, H, W, C] with the background
    :return: a tensor[B, H, W, C], for each batch and pixel, the average of the contributions
      of each template instance that overlap the pixel. For each template instance we
      use the corresponding pixel from the template. If no template instance overlaps the
      we use the background for the pixel.
    """

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
    # t: tensor[B, H+2*(h-1), W+2*(w-1), T]
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

  def render_no_batch(self, presence, types, templates, background):
    """
    Render template instances into a batch of images.

    The dimensions are: batch of B images, of size (H,W) with C channels. There are T templates,
    each described by an image (TH,TW) with C channels.

    Assumptions: At each pixel there is at most one template instance center.

    :param presence: a tensor[B, H, W], for each batch and each pixel, an entry +inf if at that
      pixel we have the center of some template instantion. And -inf if there is no template
      whose center is at the pixel. TODO: I think this can be derived from "types".
    :param types: a tensor[B, H, W, T], for each batch and each pixel and each template type,
      we have +inf if there is an instance of that template type at that pixel. We put -inf
      for all the other templates at that pixel.
    :param templates: a tensor[T, TH, TW, C], for each template type, its image.
    :param background: a tensor[B, H, W, C] with the background
    :return: a tensor[B, H, W, C], for each batch and pixel, the average of the contributions
      of each template instance that overlap the pixel. For each template instance we
      use the corresponding pixel from the template. If no template instance overlaps the
      we use the background for the pixel.
    """

    # Extract and verify input shapes. pylint: disable=invalid-name
    # In the comments, I use TH->h, TW->w ("small h") for clarity.
    inf = 1e5  # Such that inf*0 = 0 and not nan, but fits into float32 with +1

    H, W = presence.shape
    H2, W2, T = types.shape
    assert H == H2 and W == W2, "Bad types shape"
    T, TH, TW, C = templates.shape
    H3, W3, C2 = background.shape
    assert H3 == H2 and W3 == W2 and C2 == C, "Bad background shape"

    # Shortcut for padding.
    dont = (0, 0, 0)

    # Compute `a`: of each output pixel, which input pixels influence it.
    # shape will be (H+h-1)(W+w-1)hw
    w = np.eye(TH*TW).reshape(TH*TW, 1, TH, TW)  # OIHW.
    # aka "FULL" padding in old-school frameworks.
    x = jax.lax.pad(presence, -inf, ((TH-1, TH-1, 0), (TW-1, TW-1, 0)))
    a_logits = df.conv2d(x[None, ..., None], w, padding="VALID")
    # Remove the batch dimension introduced by convolution
    a_logits = a_logits[0]

    # Append a default background-claim of 0.0 (logits) via 0-padding channels.
    a_logits = jax.lax.pad(a_logits, 0.0, (dont, dont, (0, 1, 0)))
    a_combined = df.softmax_cls(a_logits, axis=-1)
    a_bg = a_combined[..., -1]
    a = a_combined[..., :-1].reshape(H+TH-1, W+TW-1, TH, TW)

    # We now have the background probabilities in `a_bg` as B(H+h-1)(W+w-1)
    # and the influence probabilities in `a` as B(H+h-1)(W+w-1)hw

    # What follows, is something very similar to `a`, but not about
    # presence/claim anymore, but this time about which template to use, hence
    # we call this one `t`, and the shape will be (H+h-1)(W+w-1)Thw.
    x = df.softmax_cls(types, axis=-1)  # Softmax along classes (T).
    w = np.eye(T*TW*TH).reshape(T*TW*TH, T, TH, TW)  # OIHW.
    t = df.conv2d(x[None, ...], w, padding=[(TH-1, TH-1), (TW-1, TW-1)])
    t = t.reshape(H+TH-1, W+TW-1, T, TH, TW)

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
    p = p[Hlo:Hhi, Wlo:Whi]

    # Combine with the given background, which is also HWC,
    # using the background-claim mask after removing its padding.
    a_bg = a_bg[Hlo:Hhi, Wlo:Whi, None]
    img = p * (1.0 - a_bg) + background * a_bg

    # return a bit more than just the image, for debug purposes.
    return img, (a, a_bg, t)

  def render_loops(self, presence, types, templates, background):
    """Render template instances into a batch of images.

    This is a version written with loops, trying to validate the idea that
    it is easier to write the loops code, and results in more readable code.
    Virtually no attention has been paid to performance.

    See doc string for render1 above.
    """
    # Extract and verify input shapes. pylint: disable=invalid-name
    # In the comments, I use TH->h, TW->w ("small h") for clarity.
    inf = 1e5  # Such that inf*0 = 0 and not nan, but fits into float32 with +1

    B, H, W = presence.shape
    B2, H2, W2, T = types.shape
    assert B == B2 and H == H2 and W == W2, "Bad types shape"
    T, TH, TW, C = templates.shape
    assert TH % 2 == 1 and TW % 2 == 1  # For now the simpler case of odd-size dimensions,
                                        # for which the center is clear
    B3, H3, W3, C2 = background.shape
    assert B3 == B2 and H3 == H2 and W3 == W2 and C2 == C, "Bad background shape"

    # The output: BHW
    output = np.zeros((B, H, W, C))

    # For each batch
    for b in range(B):
      # For each pixel in the image
      for i in range(H):
        for j in range(W):
          # Compute how many templates contribute to this pixel. At the same time
          # add up their contributions.
          count = 0.
          sum = np.zeros(C)  # The total contribution to each channel
          # For each template type
          for t in range(T):
            # A template contributes to this pixel if its center is not too far. Iterate
            # over the possible centers that can contribute to this pixel
            for ci in range(i - TH//2, i + TH//2 + 1):
              for cj in range(j - TW//2, j + TW//2 + 1):
                # If this center is still in the image
                if (ci >= 0 and ci < H) and (cj >= 0 and cj < W):
                  if types[b, ci, cj, t] > 0.0:
                    # An instance of this template is present here
                    count += 1.
                    # Compute the indices into the template
                    ti = i - (ci - TH//2)
                    tj = j - (cj - TW//2)
                    # Due to the mirroring in the input templates
                    ti = TH - 1 - ti
                    tj = TW - 1 - tj
                    print("At [{},{}] template {}[{},{}] contributes {}".format(
                      i, j, t, ti, tj, templates[t, ti, tj, 0]
                    ))
                    for c in range(C):
                      sum[c] += templates[t, ti, tj, c]
          # Now divide the sum to get the average
          if count > 0.:
            for c in range(C):
              output[b, i, j, c] = sum[c] / count
          else:
            # It's a background pixel
            for c in range(C):
              output[b, i, j, c] = background[b, i, j, c]

    return output, (None, None, None)


class RenderTest(jtu.JaxTestCase):

  def check_all_implementations(self, presence, types, templates, background, desired):
    """Test helper, pass a single image (B=1).

    Uses multiple version of the code: render1, render1_no_batch, vmap(render1_no_batch),
    and loops. Assert that the result is the desired image.

    Params:
      presence: tensor[H,W]
      types: tensor[H,W,T]
      templates: tensor[T,TW,TH,C]
      background: tensor[H,W,C]
      desired: tensor[H,W,C]

    """
    def check_one_implementation(name, func, add_batch=True):
      """Check one implementation."""
      img, (a, a_bg, t) = func(
        presence[np.newaxis] if add_batch else presence,
        types[np.newaxis] if add_batch else types,
        templates,  # We never batch the templates
        background[np.newaxis] if add_batch else background
      )
      print(("#################################\n"+
            "Using {}\nimg={}\na={}\na_bg={}\nt={}\n").format(
        name, img, a, a_bg, t
      ))
      np.testing.assert_allclose(img[0] if add_batch else img, desired)

    render = Render()
    check_one_implementation("render1", render.render1, add_batch=True)

    check_one_implementation("render_no_batch", render.render_no_batch, add_batch=False)

    # Now apply vmap. Add a batch axis, except for templates
    render_batched = jax.vmap(render.render_no_batch, in_axes=(0, 0, None, 0))
    check_one_implementation("vmap(render_no_batch)", render_batched, add_batch=True)

    check_one_implementation("loops", render.render_loops, add_batch=True)

  def test_render1(self):
    # 1. Create the test data.
    # See the docstring of the render1 function.
    #
    inf = 1e5  # Such that inf*0 = 0 and not nan, but fits into float32 with +1
               # and we get 1/0 when softmaxing with logits using this.

    H, W = 1, 5  # pylint: disable=invalid-name
    # Template types:
    T = 2  # pylint: disable=invalid-name

    # We assume an image 1x5 with two template instances (one instance of T1
    # and one instance of T2), aligned as follows:
    #  Pixel: 0  1   2   3   4
    #  0:     -  T1 T1 T1+T2 T2
    #
    # Presence logits: HW
    presence = np.array([-inf, -inf, inf, -inf, inf], np.float32).reshape(H, W)

    # Type logits: HWT
    types = np.array([[0., 0., +inf, 0., -inf],  # Template 0
                      [0., 0., -inf, 0., +inf]], # Template 1
                     np.float32).T.reshape(H, W, T)

    # Templates: ThwC
    # We have two templates, with shape 1x3 and 1 channel.
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
                        1.0*0.3 + 0.0*3.0,  # T1_0
                        1.0*0.5 + 0.0*5.0,  # T1_1
                        0.5*1.0 + 0.5*3.0,  # T1_2 + T2_0
                        0.0*1.0 + 1.0*5.0], # T2_1
                       np.float32).reshape(H, W, C)

    self.check_all_implementations(presence, types, templates, background, desired)

  def test_render2(self):
    # 1. Create the test data.
    inf = 1e5  # Such that inf*0 = 0 and not nan, but fits into float32 with +1
               # and we get 1/0 when softmaxing with logits using this.

    H, W = 1, 5  # pylint: disable=invalid-name
    # Template types:
    T = 2  # pylint: disable=invalid-name

    # We assume an image 1x5 with three template instances (two instances of T1
    # and one instance of T2), aligned as follows:
    #  Pixel: 0  1    2       3      4
    #  0:     -  T1 T1+T1 T1+T1+T2 T1+T2
    #
    # Presence logits: HW
    presence = np.array([-inf, -inf, inf, inf, inf], np.float32).reshape(H, W)

    # Type logits: HWT
    types = np.array([[0., 0., +inf, +inf, -inf],  # Template 0 (2 instances)
                      [0., 0., -inf, -inf, +inf]], # Template 1 (1 instance)
                     np.float32).T.reshape(H, W, T)

    # Templates: ThwC
    # We have two templates, with shape 1x3 and 1 channel.
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
                        1.0*0.3,            # T1_0
                        0.5*0.5 + 0.5*0.3,  # T1_1 + T1_0
                        1./3.*1.0 + 1./3.*0.5 + 1./3.*3.0,  # T1_2 + T1_1 + T2_0
                        0.5*1.0 + 0.5*5.],  # T1_2 + T2_1
                       np.float32).reshape(H, W, C)

    self.check_all_implementations(presence, types, templates, background, desired)