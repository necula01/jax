"""
JAX implements certain transformations of Python functions, e.g., jax.jit, jax.grad, jax.vmap, or jax.pmap.
The Python functions to be transformed must be JAX-traceable, which means that as the Python function executes
the only operations it applies to the data are either inspections of data attributes such as shape or type,
or special operations called JAX primitives.
In particular, a JAX-traceable function is sometimes invoked by JAX with abstract arguments.
Jax primitives know how to operate on both concrete data values and on the JAX abstract values.
An example of a JAX abstract value is ShapedArray(float32[2,2]), which has the type and the
shape of values, but not the data values.
The JAX-transformed functions must themselves be JAX-traceable functions, to ensures that these transformations
can be composed, e.g., jit(jacfwd(grad(f))).

There are pre-defined JAX primitives corresponding to most XLA operations, e.g., add, matmul, sin, cos, indexing.
JAX comes with an implementation of numpy functions in terms of JAX primitives, which means that Python programs
using JAX’s implementation of numpy are JAX-traceable and therefore transformable.
Other libraries can be made JAX-traceable by implementing them in terms of JAX primitives.

The set of JAX primitives is extensible. Instead of reimplementing a function in terms of pre-defined JAX primitives,
one can define a new primitive that encapsulates the behavior of the function.

*The goal of this document is to explain the interface that a JAX primitive must support in order to allow JAX to
perform all its transformations.*

Consider that we want to add to JAX support for a multiply-add function with three arguments defined mathematically
as "ma(x, y, z) = x * y + z". This function can operate on 3 tensors of floating point values and performs the
opertions pointwise.

== Defining functionality in terms of existing primitives ==

The easiest way to define new functions is to write them in terms of JAX primitives, or in terms of other
functions that are themselves written using JAX primitives.

"""

from jax import lax
from jax import api

from jax.interpreters import partial_eval
_orig_trace_to_subjaxpr = partial_eval.trace_to_subjaxpr
def _new_trace_to_subjaxpr(*args):
    res = _orig_trace_to_subjaxpr(*args)
    return res
partial_eval.trace_to_subjaxpr = _new_trace_to_subjaxpr

def ma_lax(x, y, z):
  return lax.add(lax.mul(x, y), z)


def sq_add_lax(a, b):
  return ma_lax(a, a, b)


print("sq_add_lax = ", sq_add_lax(2., 10.))
print("grad(sq_add_lax) = ", api.grad(sq_add_lax)(2.0, 10.))

"""
Now we add some simple function tracing helpers.
"""
import functools

_indentation = 0


def _trace(msg=None):
  """Print a message at current indentation."""
  if msg is not None:
    print("  " * _indentation + msg)


def _trace_indent(msg=None):
  """Print a message and then indent the rest."""
  global _indentation
  _trace(msg)
  _indentation = 1 + _indentation


def _trace_unindent(msg=None):
  """Unindent then print a message."""
  global _indentation
  _indentation = _indentation - 1
  _trace(msg)


def trace(name):
  """A decorator for functions to trace arguments and results."""

  def trace_func(func):  # pylint: disable=missing-docstring
    def pp(v):
        """Print certain values more succinctly"""
        vtype = str(type(v))
        if "jax.lib.xla_bridge._JaxComputationBuilder" in vtype:
            return "<JaxComputationBuilder>"
        elif "jaxlib.xla_extension.XlaOp" in vtype:
            return "<XlaOp at 0x{:x}>".format(id(v))
        elif ("partial_eval.JaxprTracer" in vtype or
              "batching.BatchTracer" in vtype or
              "ad.JVPTracer" in vtype):
            return "Traced<{}>".format(v.aval)
        elif isinstance(v, tuple):
            return "({})".format(pp_values(v))
        else:
            return str(v)
    def pp_values(args):
        return ", ".join([pp(arg) for arg in args])

    @functools.wraps(func)
    def func_wrapper(*args):
      _trace_indent("Call {}({})".format(name, pp_values(args)))
      res = func(*args)
      _trace_unindent("|-> {} = {}".format(name, pp(res)))
      return res

    return func_wrapper

  return trace_func


"""
We can conveniently use jax's implementation of numpy
"""
print("numpy")

import jax.numpy as jnp
import numpy as onp


@trace("ma_numpy")
def ma_numpy(x, y, z):
  return jnp.add(jnp.multiply(x, y), z)


@trace("sq_add_numpy")
def sq_add_numpy(a, b):
  return ma_numpy(a, a, b)


sq_add_numpy(2., 10.)
api.grad(sq_add_numpy)(2.0, 10.)

"""
Notice that in the process of computing api.grad, JAX invokes both sq_add_numpy, and therefore 
ma_numpy,
with special arguments ConcreteArray(...). These special argument types are described below. The 
main point for now is that a *JAX traceable function* must be able to operate not only on 
concrete arguments but also on special arguments that JAX may use to trace the function. This
property is satisfied as long as the function is written in terms of JAX primitives. 
"""

"""
= Defining new primitives =

To demonstrate how JAX primitives work let us pretend that we want to add a new primitive to JAX
for the multiply-add functionality.
"""
print("*** PRIMITIVES ***")
from jax import core

ma_p = core.Primitive("ma")


@trace("ma_prim")
def ma_prim(x, y, z):
  return ma_p.bind(x, y, z)


@trace("sq_add_prim")
def sq_add_prim(a, b):
  return ma_prim(a, a, b)


"""
Now we try to use it:
"""
# sq_add_prim(2., 10.)

"""We get
NotImplementedError: Evaluation rule for 'ma' not implemented

We have not told JAX anything about the semantics of the 'ma' primitive. 
"""


# The primal evaluation rules
@trace("ma_impl")
def ma_impl(x, y, z):
  """Primal implementation of the primitive.

  This function does not need to be JAX traceable.
  Args:
      x, y, z: the primal arguments of the primitive. Will only be concrete values.
  Returns:
      the concrete result of the primitive.
  """
  # Note that we use the original numpy, which is not JAX traceable
  return onp.add(onp.multiply(x, y), z)


# Now we register the primal implementation with JAX
ma_p.def_impl(ma_impl)

assert sq_add_prim(2., 10.) == 14.

"""
== JIT
"""
# api.jit(sq_add_prim)(2., 10.)

"""
We get: 
NotImplementedError: Abstract evaluation for 'ma' not implemented

In order to JIT the function, JAX first evaluates it abstractly using only the 
shape and type of the arguments. 
"""

# The abstract evaluation rules
from jax import abstract_arrays


@trace("ma_abstract_eval")
def ma_abstract_eval(xs, ys, zs):
  """Abstract evaluation of the primitive.

  This function does not need to be JAX traceable. It will be invoked with
  abstractions of the actual arguments. For example, the abstraction of a vector with 3 elements
  may be ShapedArray(float32[3]), or ConcreteArray([1., 2., 3.]). In the latter case, JAX uses
  the actual concrete value. Nevertheless, the abstract evaluation is only supposed to use the
  shapes and types obtained from the abstract value.

  Args:
      xs, ys, zs: abstractions of the arguments.
  Result:
      a ShapedArray for the result of the primitive.
  """
  assert xs.shape == ys.shape
  assert xs.shape == zs.shape
  return abstract_arrays.ShapedArray(xs.shape, xs.dtype)


# Now we register the abstract evaluation
ma_p.def_abstract_eval(ma_abstract_eval)

# api.jit(sq_add_prim)(2., 10.)

"""
We get
NotImplementedError: XLA translation rule for primitive 'ma' not found


"""

print("*** XLA translation")


@trace("ma_xla_translation")
def ma_xla_translation(c, xc, yc, zc):
  """The compilation to XLA of the primitive.

  Given an XlaBuilder and XlaOps for each argument, return the XlaOp for the
  result of the function.

  Does not need to be a JAX-traceable function.
  Implementing this function is the major obstacle to JAX extensibility, because we need to be able to
  express the semantics of the primitive in XLA.
  """
  return c.Add(c.Mul(xc, yc), zc)


# Now we register the XLA compilation rule with JAX
# TODO: for GPU? and TPU?
from jax import xla

xla.backend_specific_translations['cpu'][ma_p] = ma_xla_translation

"""Now we succedd to compile and run the sq_add_prim function."""
assert api.jit(sq_add_prim)(2., 10.) == 14.

"""Now we succedd to compile and run the sq_add_prim function.

Notice how only the dynamic arg (0) is abstracted. When the ma_abstract_eval is called
the concrete value of the static argument is wrapped as a ConcreteArray. 
"""
print("\n*** JIT only first arg")
assert api.jit(sq_add_prim, static_argnums=1)(2., 10.) == 14.

"""
== Forward differentiation ==


"""

print("*** JVP")
sq_add_prim_jvp = lambda xs, xts: api.jvp(sq_add_prim, xs, xts),

# api.jvp(sq_add_prim, (2., 10.), (1., 1.))

"""
NotImplementedError: Forward-mode differentiation rule for 'ma' not implemented
"""

from jax import ad


@trace("ma_value_and_jvp")
def ma_value_and_jvp(arg_values, arg_tangents):
  """Evaluates the primal output and the tangents (Jacobian-vector product) at that point.

  Given values of the arguments and perturbation of the arguments (tangents), compute the output of
  the primitive and the perturbation of the output.
  Tan[A] is the tangent (aka perturbation) corresponding to type A. For a tensor of
  floats, this is the same as A.

  This method must be JAX-traceable. JAX may invoke it with abstract values for the arguments
  and tangents.

  Args:
      arg_values: a tuple of the arguments
      arg_tangents: a tuple with the tangents of the arguments. Tuple has the same arity as the
          arg_values. These could also be the special value ad.Zero to specify a zero-tangent.
  Returns:
      a pair of the primal output and the tangent.
  """
  x, y, z = arg_values
  xt, yt, zt = arg_tangents
  _trace("Primal evaluation:")
  # Use the ma_prim to compute the primal output. This must be a JAX-traceable function.
  primal_out = ma_prim(x, y, z)
  _trace("Tangent evaluation:")
  _trace("Tangent evaluation:")

  # We must have a JAX-traceable way to compute the tangent. It turns out that
  # the output tangent can be computed as (xt * y + x * yt + zt),
  # which we can implement in a JAX-traceable way using the same "ma_prim" primitive.

  # We do need to deal specially with Zero. Here we just turn it into a
  # proper tensor of 0s (of the same shape as 'x').
  # An alternative would be to check for Zero and perform algebraic
  # simplification of the output tangent computation.
  def make_zero(tan):
      return lax.zeros_like_array(x) if tan is ad.zero else tan

  output_tangent = ma_prim(make_zero(xt), y, ma_prim(x, make_zero(yt), make_zero(zt)))
  return (primal_out, output_tangent)


ad.primitive_jvps[ma_p] = ma_value_and_jvp

assert api.jvp(sq_add_prim, (2., 10.), (1., 1.)) == (14., 5.)

"""
We can also JIT the JVP

"""
print("\n*** JIT the JVP")
assert api.jit(lambda arg_values, arg_tangents: api.jvp(sq_add_prim, arg_values, arg_tangents))((2., 10.),
                                                                                                (1., 1.)) == (14., 5.)

"""
== Reverse differentiation ==

"""
# api.grad(sq_add_prim)(2., 10.)

"""We get
NotImplementedError: Reverse-mode differentiation rule for 'ma' not implemented

What this actually means is that we have not implemented a companion transpose function
"""


@trace("ma_transpose")
def ma_transpose(ct, x, y, z):
  """Evaluates the transpose of a linear primitive.

  This method is only used when computing the backward gradient following value_and_jvp,
  and is only needed for primitives that are used in the JVP calculation for some
  other primitive. We need transposition for ma_prim, because we have used ma_prim in the
  computation of the output_tangent in ma_value_and_jvp.

  Such primitives occurring in JVP calculations express linear computations in terms of
  some other tangents. Intuitively, the transpose for a linear function returns the
  coefficients for the linear arguments, e.g., if f(ct1, ct2) is a linear function
  computing ct1*l1 + ct2*l2, then f_transpose(ct1, ct2) = (l1, l2).

  In our case, ma is not a linear primitive. However, it is used linearly w.r.t. tangents
  in ma_value_and_jvp:
       output_tangent(xt, yt, zt) = ma_prim(xt, y, ma_prim(x, yt, zt)
  ,meaning that at least one of the first two multiplicative arguments are constants (not tangents).


  Consider that the current primitive is a function of three arguments, and it is linear
  in the values of the last two arguments: f(x,l1,l2) computes c1*l1+c2*l2, for some
  c1 and c2 that depend only on the first argument x. For such a primitive,

      eval_transpose(ct, x, None, None) = (None, c1 * ct, c2 * ct)

  The arguments to eval_transpose are the cotangent of the output (ct), and one
  additional argument for each argument of function f. In JAX, eval_transpose will
  be passed None for the linear arguments, since they should not be needed.
  There are three return values, each a cotangent corresponding to an argument of
  function f. Only the cotangent of the linear arguments are actually needed.

  Args:
      ct: the cotangent of the output of the primitive.
      args: tuple of
  Returns:
      the cotangent of the inputs
  """
  if x is not ad.undefined_primal:
    # This use of ma is with a constant "x"
    assert y is ad.undefined_primal
    ct_y = ad.zero if ct is ad.zero else ma_prim(x, ct, onp.zeros(onp.shape(x), dtype=onp.float32))
    res = None, ct_y, ct
  else:
    # This use of ma is with a constant "y"
    assert x is ad.undefined_primal
    ct_x = ad.zero if ct is ad.zero else ma_prim(ct, y, onp.zeros(onp.shape(y), dtype=onp.float32))
    res = ct_x, None, ct
  return res


ad.primitive_transposes[ma_p] = ma_transpose

print("\n*** GRAD")
assert api.grad(sq_add_prim)(2., 10.) == 4.

print("\n*** JIT OF GRAD")
assert api.jit(api.grad(sq_add_prim))(2., 10.) == 4.

"""
== Batching ==

"""
# api.vmap(sq_add_prim, in_axes=0, out_axes=0)(onp.array([2., 3.]),
#                                             onp.array([10., 20.]))
"""
We get
NotImplementedError: Batching rule for 'ma' not implemented
"""

print("\n*** BATCH")
from jax import batching


@trace("ma_batch")
def ma_batch(vector_arg_values, batch_axes):
  # Cheating, we use the fact that the primitive works on vectors as well
  assert batch_axes[0] == batch_axes[1]
  assert batch_axes[0] == batch_axes[2]
  _trace("Using ma to compute the batch:")
  res = ma_prim(*vector_arg_values)
  return res, batch_axes[0]


batching.primitive_batchers[ma_p] = ma_batch

assert onp.allclose(api.vmap(sq_add_prim, in_axes=0, out_axes=0)(
  onp.array([2., 3.]),
  onp.array([10., 20.])),
  [14., 29.])

print("\n*** JIT of VMAP")
assert onp.allclose(api.jit(api.vmap(sq_add_prim, in_axes=0, out_axes=0))
                    (onp.array([2., 3.]),
                     onp.array([10., 20.])),
                    [14., 29.])
