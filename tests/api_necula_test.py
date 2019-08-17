

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import collections
import functools
from functools import partial
import itertools
from unittest import skip, SkipTest

from absl.testing import absltest
from absl.testing import parameterized

import numpy as onp
import numpy.random as npr
import six

from jax import api
from jax import core
from jax import numpy as jnp
from jax import lax
from jax import test_util as jtu
from jax import ad_util
from jax import lax_reference
from jax.test_util import check_grads
from jax.interpreters import xla
from jax.lib import xla_bridge
from jax.lib import xla_client

from jax.config import config
config.parse_flags_with_absl()
FLAGS = config.FLAGS


class PrimitiveInterface(object):
    """The interface that a primitive must implement.

    We assume that the primitive implements a function with
    arguments of type "a" and return value of type "b".
    """
    def __init__(self, print_name, multiple_results=False):
        self.print_name = print_name
        self.multiple_results = multiple_results

    def __str__(self):
        return self.print_name

    def eval(self, *arg_values):
        """Evaluates the primitive function.

        This is used when there is no tracing.
        Args:
            arg_values: the array of primitive arguments (concrete values).

        Returns:
            the primal result.
        """
        raise NotImplemented

    def abstract_eval(self, *arg_shapes):
        """Evaluates the shape of the custom function.

        This is used as part of jax.jit ...

        Args:
            arg_shapes: the shapes of the custom function
                arguments, e.g., ShapedArray.
        Returns:
            the shape of the result.
        """
        raise NotImplemented


    def xla_translation(self, c, *arg_ops):
        """Produces the XLA operand corresponding to the result.

        Args:
            c: a _JaxComputationBuilder
            args_ops: a tuple with the XlaOp operands of the
                custom function.

        Returns:
            an XlaOp corresponding to the result.
        """
        raise NotImplemented

    def jvp_instance(self):
        """The custom function representing the JVP for this custom function instance.

        This represents another custom function that computes JVP for current function.
        If the current function has type "a -> b", then the VJP instance has type
        "(a, b, Tan a) -> Tan b", where "Tan a" is the tangent of "a".

        Returns:
            a PrimitiveInterface that represents the computation of the JVP.
        """
        raise NotImplemented

    def transpose_instance(self):
        """The custom function representing the transpose for this custom function instance.


        TODO: This represents another custom function that computes the transpose for current function.
        If the current function has type "a -> b", then the transpose instance has type
        "(Tan a, a, b, ???) -> Tan b", where "Tan a" is the tangent of "a".

        Returns:
            a PrimitiveInterface that represents the computation of the transpose.
        """
        raise NotImplemented

    def vjp_instance(self, argnums):
        """The custom function representing the VJP for this custom function instance.

        This represents another custom function that computes VJP for current function.
        If the current function has type "a -> b", then the VJP instance has type
        "(a, b, CT b) -> CT a", where "CT a" is the cotangent of "a".

        Args:
            argnums: a tuple of arguments w.r.t. which to differentiate, e.g., (0, 2)
                to differentiate w.r.t. first and third argument.

        Returns:
            a PrimitiveInterface that represents the computation of the VJP.
        """
        raise NotImplemented

    def batch_instance(self, batch_axes):
        """The custom function representing the batch version for this custom function instance.

        This represents another custom function that computes the current function
        applied vectorized arguments.

        If the current function has type "[a] -> [b]" and batch_axes=1, then the batch instance has type
        "[a, c] -> [c, b]", i.e., it can be applied to a vectorized argument where the axis 1 is the
        vectorized dimension. The result is also vectorized, always on dimension 0.

        Args:
            batch_axes: a tuple of arguments, one element per argument, specifying on what
                position in each vectorized argument is the vectorized dimension.

        Returns:
            a PrimitiveInterface that represents the computation of the batch operation.
        """
        raise NotImplemented

# Define a generic primitive that we can customize
from jax.interpreters import ad
from jax import batching

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
    """Print a message and then indent the rest."""
    global _indentation
    _indentation = _indentation - 1
    _trace(msg)


def trace(name):
  """A decorator for functions to log arguments and results."""
  def trace_func(func):        # pylint: disable=missing-docstring
    @functools.wraps(func)
    def func_wrapper(*args):
        _trace_indent("Call {}({})".format(name, args))
        res = func(*args)
        _trace_unindent("|-> {} = {}".format(name, res))
        return res
    return func_wrapper
  return trace_func

# Whether to define the VJP. If True, then JVP is disabled. If False, then
# VJP is done by means of JVP.
_DEF_VJP = False

def setupCustomFunction():
    """Defines the JAX genericCustomPrimitive.

    Does all the magic to hook into JAX the functionality needed from
    a custom function.

    Returns a Python function that can be used to invoke a custom function,
    e.g.,
    >> customFunction = setupCustomFunction()
    >> customFunction(args, instance=PrimitiveInterface())
    """
    genericCustomPrimitive = core.Primitive("primitive")
    # This custom primitive always returns multiple arguments
    genericCustomPrimitive.multiple_results = True
    def eval(*arg_values, instance=None):
        _trace_indent("Calling {}.eval with {}".format(instance, arg_values))
        res = instance.eval(*arg_values)
        _trace_unindent("{}.eval returned {}".format(instance, res))
        if isinstance(res, tuple):
            return res
        else:
            return (res, )
    genericCustomPrimitive.def_impl(eval)

    def abstract_eval(*arg_shapes, instance=None):
        _trace_indent("Calling {}.abstract_eval with {}".format(instance, arg_shapes))
        res = instance.abstract_eval(*arg_shapes)
        _trace_unindent("{}.abstract_eval returned {}".format(instance, res))
        if isinstance(res, tuple):
            return res
        else:
            return (res, )
    genericCustomPrimitive.def_abstract_eval(abstract_eval)

    def xla_translation(c, *arg_ops, instance=None):
        _trace_indent("Calling {}.xla_translation with {}".format(instance, arg_ops))
        res = instance.xla_translation(c, *arg_ops)
        _trace_unindent()
        if c.GetShape(res).is_tuple():
            return res
        else:
            return c.Tuple(res)
    xla.backend_specific_translations['cpu'][genericCustomPrimitive] = xla_translation

    def value_and_jvp(arg_values, arg_tangents, instance=None):
        """
        Args:
            arg_values: a list of concrete or abstract values corresponding to the arguments
                of the primitive.
            arg_tangents: a list of concrete or abstract values corresponding to the
                arguments of the primitive.

        Returns:
            a pair of the output and its tangent.
        """
        _trace_indent("Calling {}.value_and_jvp with {} tangents={}".format(instance, arg_values, arg_tangents))
        # Run the custom function through JAX to get the primal output value
        primal_out = genericCustomPrimitive.bind(*arg_values, instance=instance)
        # Construct the custom function for the JVP
        jvp_instance = instance.jvp_instance()
        # arg_values and tangents are lists of values or of tracers. But .bind wants them
        # to be tracers, so we destructure the list when passing to .bind
        output_tangent = genericCustomPrimitive.bind(*arg_values, *primal_out, *arg_tangents, instance=jvp_instance)
        _trace_unindent("{}.value_and_jvp returned {} and {}".format(instance, primal_out, output_tangent))
        return (primal_out, output_tangent)

    ad.primitive_jvps[genericCustomPrimitive] = value_and_jvp

    def eval_transpose(ct, *args, instance=None):
        """
        Args:
            ct: the cotangent of the output of the primitive.
            args: tuple of
        Returns:
            the cotangent of the inputs
        """
        _trace_indent("Calling {}.eval_transpose with ct={}, args={}".format(instance, ct, args))
        # Construct the custom function for the JVP
        transpose_instance = instance.transpose_instance()
        new_args = args[:-1] # We need to drop the last arg, it is None, and the abstract_evaluator crashes.
         # Transpose always has multiple results
        output_transpose = genericCustomPrimitive.bind(*ct, *new_args, instance=transpose_instance)
        _trace_unindent("{}.eval_transpose returned {}".format(instance, output_transpose))
        return output_transpose

    ad.primitive_transposes[genericCustomPrimitive] = eval_transpose

    def value_and_vjp(argnums, *arg_values, instance=None):
        """According to ad.defvjp_argnums this function must return
        a pair with the primal output and a function that takes the cotangent
        of the output to produce the cotangent of the inputs mentioned in argnums."""
        _trace_indent("Calling {}.value_and_vjp with {}".format(instance, arg_values))
        # TODO: this is called with (Traced<ShapedArray>,) when jitting the grad!
        # The result then is Traced<ShapedArray>
        # Run the custom function through JAX to get the primal output value
        primal_out = genericCustomPrimitive.bind(*arg_values, instance=instance)
        # Construct the custom function for the VJP
        vjp_instance = instance.vjp_instance(argnums)
        def pullback_py(ct):
            _trace_indent("Calling {}.pullback_py with ct={}. Closure: args={}, out={}".format(instance, ct, arg_values, primal_out))
            res = genericCustomPrimitive.bind(*arg_values, primal_out, ct, instance=vjp_instance)
            _trace_unindent("{}.pullback_py returned {}".format(instance, res))
            return res
        _trace_unindent()
        return (primal_out, pullback_py)

    if _DEF_VJP:
        ad.defvjp_argnums(genericCustomPrimitive, value_and_vjp)

    def eval_batch(vector_arg_values, batch_axes, instance=None):
        _trace_indent("Calling {}.eval_batch axes={} vector_args={}".format(instance, batch_axes, vector_arg_values))
        batch_instance = instance.batch_instance(batch_axes)
        vector_out_values = genericCustomPrimitive.bind(*vector_arg_values, instance=batch_instance)
        # TODO: Don't know how to handle batch operations where the result is not vectorized on first axis.
        # Should pass the desired vectorized dimension to the primitive.
        _trace_unindent()
        return vector_out_values, [0]

    batching.primitive_batchers[genericCustomPrimitive] = eval_batch

    def customFunction(*args, instance=None, multiple_results=False):
        """
        Params:
            args:
            instance:
            multiple_results: if the caller expects a single result
        """
        if multiple_results:
            return genericCustomPrimitive.bind(*args, instance=instance)
        else:
            res = genericCustomPrimitive.bind(*args, instance=instance)
            return res[0]

    return customFunction

customFunction = setupCustomFunction()

from jax import abstract_arrays


class CustomGradTest(jtu.JaxTestCase):
    """Tests to learn how to hook custom gradient to JAX."""


    def customFunctionTrivial(self, x):
        # Define the custom function.
        class Trivial(PrimitiveInterface):
            """Custom function that sums over all axes of two inputs and adds the result.

            We assume it will be invoked with  x : float32[] and returns float32[].
            """

            def eval(self, x):
                return x * x * x

            def abstract_eval(self, x_abs):
                assert isinstance(x_abs, abstract_arrays.ShapedArray)
                return abstract_arrays.ShapedArray((), x_abs.dtype)

            def xla_translation(self, c, x_op):
                # Fake the compilation
                return c.Constant(16.0, onp.float32)

            def jvp_instance(self):
                # Make a CustomPrimitive for computing the JVP
                class TrivialJVP(PrimitiveInterface):
                    def eval(self, x, primal_out, x_t):
                        return 3 * x * x * x_t

                    def abstract_eval(self, x, primal_out_abs, x_t_abs):
                        assert isinstance(x_t_abs, abstract_arrays.ShapedArray)
                        return abstract_arrays.ShapedArray((), x_t_abs.dtype)

                    def xla_translation(self, c, x_op, primal_out_op, x_t_op):
                        # Fake the compilation
                        return c.Constant(17.0, onp.float32)

                    def transpose_instance(self):
                        # Make a CustomPrimitive for computing the JVP
                        class TrivialTranspose(PrimitiveInterface):
                            def multiple_results(self):
                                return True

                            def eval(self, x_ct, x, primal_out):
                                # Must return None for the first twoarguments (x, primal_out)
                                return (None, None, 3 * x * x * x_ct, )

                            def abstract_eval(self, x_ct_abs, x_abs, primal_out_abs):
                                assert isinstance(x_ct_abs, abstract_arrays.ShapedArray)
                                return (None,
                                        None,
                                        abstract_arrays.ShapedArray((), x_ct_abs.dtype))

                            def xla_translation(self, c, x_t_op, x_op, primal_out_op):
                                # Fake the compilation
                                return c.Tuple(
                                    c.Constant(0.0, onp.float32),  # Not needed?
                                    c.Constant(0.0, onp.float32),  # Not needed?
                                    c.Constant(18.0, onp.float32))

                        return TrivialTranspose("p.jvp.transpose")

                return TrivialJVP("p.jvp")



            def vjp_instance(self, argnums):
                # Make a CustomPrimitive for computing the VJP
                class TrivialVJP(PrimitiveInterface):
                    def eval(self, x, primal_out, ct):
                        # Fake the result, so we recognize it in output
                        return (3 * x * x, )

                    def abstract_eval(self, x, primal_out, ct_abs):
                        return core.AbstractTuple([abstract_arrays.ShapedArray((), ct_abs.dtype)])

                    def xla_translation(self, c, x_op, primal_out_op, ct_op):
                        # This is called on GRAD o JIT
                        # Fake the compilation
                        return c.Tuple(c.Constant(20.0))

                    def vjp_instance(self, argnums):
                        # This is for double GRAD
                        raise NotImplemented


                return TrivialVJP("p.vjp")

            def batch_instance(self, in_axes):
                assert in_axes == (0,)  # The base function is scalar

                # Make a CustomPrimitive for the batched operation
                class TrivialBatched(PrimitiveInterface):
                    def eval(self, x_batch):
                        # Fake the result, so we recognize it in output
                        return x_batch * x_batch * x_batch

                    def abstract_eval(self, x_batch):
                        assert len(x_batch.shape) == 1
                        return abstract_arrays.ShapedArray(x_batch.shape, x_batch.dtype)

                    def xla_translation(self, c, x_batch_op):
                        # This is called on GRAD o JIT
                        # Fake the compilation
                        return c.Constant(onp.array([12.0, 13.0]))

                    def batch_instance(self, in_axes_2):
                        # For repeated vmap
                        assert in_axes_2 == (1,)
                        # Make a CustomPrimitive for the repeated batched operation
                        class TrivialBatched2(PrimitiveInterface):
                            def eval(self, x_batch):
                                return x_batch * x_batch * x_batch

                            def abstract_eval(self, x_batch):
                                assert len(x_batch.shape) == 2
                                return abstract_arrays.ShapedArray(x_batch.shape, x_batch.dtype)

                            def xla_translation(self, c, x_batch_op):
                                # This is called on GRAD o JIT
                                # Fake the compilation
                                return c.Constant(onp.ones((3, 2)) * 5.0)

                        return TrivialBatched2("p.batch.batch")

                    def vjp_instance(self, argnums):
                        # Make a CustomPrimitive for computing the VJP of BATCH
                        class TrivialBatchJVP(PrimitiveInterface):
                            def eval(self, x, primal_out, ct):
                                # Result must be a tuple even if we have a single argument
                                return (x * x * 3.0,)

                            def abstract_eval(self, x, primal_out, ct_abs):
                                return core.AbstractTuple([abstract_arrays.ShapedArray(x.shape, ct_abs.dtype)])

                            def xla_translation(self, c, x_op, primal_out_op, ct_op):
                                # This is called on GRAD o JIT
                                # Fake the compilation
                                assert False
                                return c.Tuple(c.Constant(25.0))

                        return TrivialBatchJVP("p.batch.vjp")

                    def jvp_instance(self):
                        # Make a CustomPrimitive for computing the JVP of BATCH
                        class TrivialBatchJVP(PrimitiveInterface):
                            def eval(self, x, primal_out, ct):
                                # Result must be a tuple even if we have a single argument
                                return x * x * 3.0

                            def abstract_eval(self, x, primal_out, ct_abs):
                                return abstract_arrays.ShapedArray(x.shape, ct_abs.dtype)

                            def xla_translation(self, c, x_op, primal_out_op, ct_op):
                                # This is called on GRAD o JIT
                                # Fake the compilation
                                assert False
                                return c.Tuple(c.Constant(25.0))

                            def transpose_instance(self):
                                # Make a CustomPrimitive for computing the JVP
                                class TrivialBatchTranspose(PrimitiveInterface):
                                    def eval(self, x_ct, x, primal_out):
                                        # Must return None for the first two arguments: x, primal_out
                                        return (None, None, 3 * x * x * x_ct,)

                                    def abstract_eval(self, x_ct_abs, x_abs, primal_out_abs):
                                        assert isinstance(x_ct_abs, abstract_arrays.ShapedArray)
                                        return (None,
                                                None,
                                                abstract_arrays.ShapedArray((), x_ct_abs.dtype))

                                    def xla_translation(self, c, x_t_op, x_op, primal_out_op):
                                        # Fake the compilation
                                        return c.Tuple(
                                            c.Constant(0.0, onp.float32),  # Not needed?
                                            c.Constant(0.0, onp.float32),  # Not needed?
                                            c.Constant(18.0, onp.float32))

                                return TrivialBatchTranspose("p.batch.jvp.transpose")

                        return TrivialBatchJVP("p.batch.jvp")

                return TrivialBatched("p.batch")


        return customFunction(x, instance=Trivial("p"))


    def testTrivial(self):
        f = lambda x: self.customFunctionTrivial(x)
        x = 2.0
        x_t = 0.2  # The tangent of x

        def one_transform(name="", func=None, args=None, expected=None):
            print("\n*** transformation: {}".format(name))
            print("\nmake_jaxpr computation:")
            print("\nJAXPR: ", api.make_jaxpr(func)(*args))
            print("\nEvaluation:")
            res = func(*args)
            print("  Result of {} = {}".format(name, res))
            self.assertTrue(onp.allclose(expected, res))

        one_transform(name="none (primal evaluation)",
                      func=f, args=[x], expected=8.0)

        one_transform(name="JIT",
                      func=api.jit(f),
                      args=[x],
                      expected=16.0)

        one_transform(name="JIT of JIT",
                      func=api.jit(api.jit(f)),
                      args=[x],
                      expected=16.0)

        one_transform(name="JVP",
                      func=lambda xs, xts: api.jvp(f, xs, xts),
                      args=[(x,), (x_t,)],
                      expected=(8., 2.4))

        one_transform(name="JIT of JVP",
                      func=api.jit(lambda xs, xts: api.jvp(f, xs, xts)),
                      args=[(x,), (x_t,)],
                      expected=(16., 17.))

        one_transform(name="GRAD",
                      func=api.grad(f),
                      args=[x],
                      expected=12.0)

        one_transform(name="JIT of GRAD",
                      func=api.jit(api.grad(f)),
                      args=[x],
                      expected=18.0)

        one_transform(name="VMAP",
                      func=api.vmap(f, in_axes=0, out_axes=0),
                      args=[onp.array([2.0, 3.0])],
                      expected=[8.0, 27.0])

        one_transform(name="JIT of VMAP",
                      func=api.jit(api.vmap(f, in_axes=0, out_axes=0)),
                      args=[onp.array([2.0, 3.0])],
                      expected=[12.0, 13.0])

        one_transform(name="VMAP of VMAP",
                      func=api.vmap(api.vmap(f, in_axes=0, out_axes=0),
                                            in_axes=1, out_axes=1),
                      args=[onp.array([[2.0, 3.0], [2.0, 4.0], [2.0, 5.0]])],
                      expected=[[8.0, 8.0, 8.0], [27.0, 64.0, 125.0]])

        one_transform(name="JIT of VMAP of VMAP",
                      func=api.jit(api.vmap(api.vmap(f, in_axes=0, out_axes=0),
                                            in_axes=1, out_axes=1)),
                      args=[onp.array([[2.0, 3.0], [2.0, 4.0], [2.0, 5.0]])],
                      expected=[[5., 5., 5.], [5., 5., 5.]])

        one_transform(name="GRAD of np.sum of VMAP",
                      func=api.grad(lambda xv: jnp.sum(api.vmap(f, in_axes=0, out_axes=0)(xv))),
                      args=[onp.array([2.0, 3.0])],
                      expected=[12., 27.])


    def customFunctionSum2(self, x, y):
        # Define the custom function.
        class Sum2(PrimitiveInterface):
            """Custom function that sums over all axes of two inputs and adds the result.

            We assume it will be invoked with  x : float32[2,2] and y : float32[3]
            """

            def eval(self, x, y):
                # This is called with ShapedArray when JIT of GRAD.
                # This works because numpy delegates to special methods on
                # the inputs when they are not arrays.
                assert x.shape == (2, 2)
                assert y.shape == (3, )
                return onp.sum(x) + onp.sum(y)

            def abstract_eval(self, x_abs, y_abs):
                assert isinstance(x_abs, abstract_arrays.ShapedArray)
                assert isinstance(y_abs, abstract_arrays.ShapedArray)
                return abstract_arrays.ShapedArray((), x_abs.dtype)

            def xla_translation(self, c, x_op, y_op):
                # Fake the compilation
                return c.Constant(16.0, onp.float32)

            def vjp_instance(self, argnums):
                # Make a CustomPrimitive for computing the VJP
                class Sum2JVP(PrimitiveInterface):
                    def eval(self, x, y, primal_out, ct):
                        # Fake the result, so we recognize it in output
                        if argnums == (0, 1):
                            return (onp.array([[6 * ct, 7 * ct], [8 * ct, 9 * ct]]).astype(onp.float32), # CT x
                                    onp.array([2 * ct, 3 * ct, 4 * ct]).astype(onp.float32))  # CT y
                        elif argnums == (1,):
                            return core.JaxTuple([
                                onp.zeros((2, 2), onp.float32),
                                onp.array([2 * ct, 3 * ct, 4 * ct]).astype(onp.float32)])
                        else:
                            assert False

                    def abstract_eval(self, x, y, primal_out, ct_abs):
                        # TODO: this is called with ConcreteArray for x, y, primal when grad, and ShapedArray with JIT
                        # It is also called with ConcreteArray for ct_abs
                        # assert isinstance(ct_abs, abstract_arrays.ShapedArray)
                        if argnums == (0, 1):
                            return core.AbstractTuple(
                                [abstract_arrays.ShapedArray((2, 2), ct_abs.dtype),  # CT x
                                 abstract_arrays.ShapedArray((3,), ct_abs.dtype)])  # CT y
                        elif argnums == (1,):
                            return core.AbstractTuple([abstract_arrays.ShapedArray((3,), ct_abs.dtype)])
                        else:
                            assert False

                    def xla_translation(self, c, x_op, y_op, primal_out_op, ct_op):
                        # This is called on GRAD o JIT
                        # Fake the compilation
                        if argnums == (0, 1):
                            return c.Tuple(
                                c.Constant(onp.array([[[60.0, 70.0], [80.0, 90.0]]]).astype(onp.float32)),  # For x
                                c.Constant(onp.array([20.0, 30.0, 40.0]).astype(onp.float32))
                            )
                        else:
                            assert False

                    def vjp_instance(self, argnums):
                        # This is for double GRAD
                        raise NotImplemented


                return Sum2JVP("p.vjp")

            def batch_instance(self, batch_axes):
                class Sum2Batch(PrimitiveInterface):
                    def eval(self, x, y):
                        return onp.array([100.0, 200.0])

                    def abstract_eval(self, x, y):
                        if batch_axes == (0, 0):
                            return abstract_arrays.ShapedArray((x.shape[0],), x.dtype)
                        else:
                            assert False

                    def xla_translation(self, c, x_op, y_op):
                        # This is called on GRAD o JIT
                        # Fake the compilation
                        if batch_axes == (0, 0):
                            return c.Constant(onp.array([101.0, 201.0]).astype(onp.float32))
                        else:
                            assert False

                return Sum2Batch("p.batch")

        return customFunction(x, y, instance=Sum2("p"))


    def testSum2(self):
        f = lambda x, y: self.customFunctionSum2(x, y)
        x = onp.array([[1.0, 2.0], [3.0, 4.0]]).astype(onp.float32)
        y = onp.array([5.0, 6.0, 7.0]).astype(onp.float32)
        #
        # print("\n*** Primal evaluation")
        # res_fun = f(x, y)
        # print("Result is {}".format(res_fun))
        # self.assertEquals(28.0, res_fun)
        #
        # print("\n*** JIT evaluation")
        # f_jit = api.jit(f)
        # res_jit = f_jit(x, y)
        # print("Result from JIT is {}".format(res_jit))
        # # Use the fake value from the xla_translation to test
        # self.assertEquals(16.0, res_jit)
        #
        # # JIT of JIT evaluation (uses "abstract_eval" and "xla_translation")
        # f_jit_jit = api.jit(api.jit(f))
        # res_jit_jit = f_jit_jit(x, y)
        # print("Result from JIT JIT is {}".format(res_jit_jit))
        # self.assertEquals(16.0, res_jit_jit)
        #
        # # Backwards AD
        # print("\n*** GRAD evaluation")
        # grad_argnums = (0, 1)
        # # grad_argnums = 1  # Only w.r.t. y
        # f_grad = api.grad(f, argnums=grad_argnums)
        #
        # res_grad = f_grad(x, y)
        # print("Result from f_grad is {}".format(res_grad))
        # if grad_argnums == (0, 1):
        #     self.assertEquals(2, len(res_grad))
        #     self.assertTrue(onp.allclose([[6.0, 7.0], [8.0, 9.0]], res_grad[0]))
        #     self.assertTrue(onp.allclose([2.0, 3.0, 4.0], res_grad[1]))
        # else:
        #     assert False

        # print("\n*** GRAD or GRAD evaluation")
        # f_grad_grad = api.grad(api.grad(f, argnums=grad_argnums), argnums=grad_argnums)
        # res_grad_grad = f_grad_grad(x, y)
        # print("Result from GRAD GRAD is {}".format(res_grad_grad))
        # self.assertTrue(onp.allclose([[2.0, 3.0], [4.0, 5.0]], res_grad_grad))
        #
        #
        # print("\n*** JIT of GRAD evaluation")
        # jit_static_argnums = ()
        # f_grad_jit = api.jit(f_grad)
        # res_grad_jit = f_grad_jit(x, y)
        # print("Result from f_grad_jit is {}".format(res_grad_jit))
        # # Assert we got the value from xla_translation for the vjp_instance
        # if grad_argnums == (0, 1):
        #     self.assertEquals(2, len(res_grad_jit))
        #     self.assertTrue(onp.allclose([[60.0, 70.0], [80.0, 90.0]], res_grad_jit[0]))
        #     self.assertTrue(onp.allclose([20.0, 30.0, 40.0], res_grad_jit[1]))
        # else:
        #     assert False

        # # Forward AD
        # Not supposed to work yet
        # f_fwd_grad = api.jvp(f, (x,y), (onp.array([[0.1, 0.2], [0.3, 0.4]]),))
        # res_fwd_grad = f_fwd_grad(x, y)
        # print("Result from f_fwd_grad is {}".format(res_fwd_grad))
        # self.assertTrue(onp.allclose([[1.0, 1.0], [1.0, 1.0]], res_grad))
        #
        # Batching
        print("\n*** VMAP")
        f_vmap = api.vmap(f, in_axes=0, out_axes=1)
        res_vmap = f_vmap(onp.array([x, x * 2]),
                          onp.array([y, y * 3]))
        print("Result from f_vmap is {}".format(res_vmap))
        self.assertTrue(onp.allclose([100.0, 200.0], res_vmap))

        print("\n*** VMAP of VMAP")
        f_vmap_vmap = api.vmap(api.vmap(f, in_axes=0, out_axes=0), in_axes=1, out_axes=1)
        res_vmap_vmap = f_vmap_vmap(onp.array([x, x * 2]),
                                    onp.array([y, y * 3]))
        print("Result from f_vmap is {}".format(res_vmap))
        self.assertTrue(onp.allclose([100.0, 200.0], res_vmap))

        # print("\n*** JIT of batch")
        # f_vmap_jit = api.jit(api.vmap(f, in_axes=0, out_axes=1))
        # res_vmap_jit = f_vmap_jit(onp.array([x, x * 2]),
        #                           onp.array([y, y * 3]))
        # print("Result from f_vmap_jit is {}".format(res_vmap_jit))
        # self.assertTrue(onp.allclose([101.0, 201.0], res_vmap_jit))

        # print("\n*** Batch of GRAD")
        # f_vmap_jit = api.vmap(api.grad(f, argnums=(0,1)), in_axes=0, out_axes=1))
        # res_vmap_jit = f_vmap_jit(onp.array([x, x * 2]),
        #                           onp.array([y, y * 3]))
        # print("Result from f_vmap_jit is {}".format(res_vmap_jit))
        # self.assertTrue(onp.allclose([101.0, 201.0], res_vmap_jit))

    def testDefVJP(self):
        """Test how defvjp is implemented."""
        @api.custom_transforms
        def f(x, y):
            return x * y + x * 2

        # The custom VJP function that we have access to is ((a, CT b) -> CT a)
        def my_custom_vjp(x, y, ct):
            return (ct * y + ct * 2, ct * x)

        # This is how JAX wants it: a -> (b, CT b -> CT a)
        def jax_required_custom_vjp(x, y):
            primal_res = f(x, y)
            def required_ct(ct):
                return my_custom_vjp(x, y, ct)
            return (primal_res, required_ct)
        api.defvjp_all(f, jax_required_custom_vjp)

        # Backward AD w.r.t. y
        res_back = api.grad(f, argnums=1)(3., 4.)
        print("Grad backward result is {}".format(res_back))
        self.assertEquals(3.0, res_back)  # AD wrt y is x = 3.0

        # JIT of Backward AD
        res_back_jit = api.jit(api.grad(f, argnums=1))(3., 4.)
        print("Grad backward JIT result is {}".format(res_back_jit))
        self.assertEquals(3.0, res_back_jit)

        # Forward AD
        # Getting: NotImplementedError: Evaluation rule for 'f_jvp' not implemented
        # res_fwd = api.jvp(f, (3., 4.), (0.1, 0.2))
        # print("Grad forward result is {}".format(res_fwd))
        # self.assertTrue(onp.allclose([6.0, 3.0], res_fwd))

    def testDefJVP2(self):
        """Test how defjvp is implemented."""
        @api.custom_transforms
        def f(x, y):
            return x * y + x * 2.0

        # The custom VJP function that we have access to is ((a, CT b) -> CT a)
        def my_custom_jvp(x, y, ct):
            return (ct * y + ct * 2, ct * x)

        # This is how JAX wants it: a -> (b, CT b -> CT a)
        def jax_required_custom_jvp(xy, xy_t):
            x, y = xy
            x_t, y_t = xy_t
            if x_t is ad_util.zero and y_t is ad_util.zero:
                t = x * 2.0
            elif x_t is ad_util.zero:
                t = x * y_t + x * 2.0
            elif y_t is ad_util.zero:
                t = y * x_t + x * 2.0
            else:
                t = x * y_t + x_t * y + x * 2.0
            return f(x, y), t

        api.defjvp_all(f, jax_required_custom_jvp)

        # # Forward AD
        # res_fwd = api.jvp(f, (2.,3.), (0.2, 0.3))
        # print("Grad forward result is {}".format(res_fwd))
        #
        # JIT of forward
        res_fwd_jit = api.jit(lambda xy, xy_t: api.jvp(f, xy, xy_t))((2.,3.), (0.2, 0.3))
        print("JIT of Grad forward result is {}".format(res_fwd_jit))

        # Backward AD w.r.t. y
        res_back = api.grad(f)(2., 3.)
        print("Grad backward result is {}".format(res_back))
        self.assertEquals(3.0, res_back)  # AD wrt y is x = 3.0

        # JIT of Backward AD
        res_back_jit = api.jit(api.grad(f))(2., 3.)
        print("Grad backward JIT result is {}".format(res_back_jit))
        self.assertEquals(3.0, res_back_jit)

    def testDefJVP1(self):
        """Test how defjvp is implemented."""

        @api.custom_transforms
        def f(x):
            return x * x * x


        # This is how JAX wants it: (a, Tan a) -> (b, Tan b)
        def jax_required_custom_jvp(xy, xy_t):
            x,  = xy
            x_t,  = xy_t
            if x_t is ad_util.zero:
                t = None
            else:
                t = 3.0 * x * x * x_t
            return f(x), t

        api.defjvp_all(f, jax_required_custom_jvp)

        # Forward AD
        res_fwd = api.jvp(f, (2.,), (0.2,))
        print("Grad forward result is {}".format(res_fwd))

        # JIT of forward
        res_fwd_jit = api.jit(lambda x, x_t: api.jvp(f, x, x_t))((2.,), (0.2,))
        print("JIT of Grad forward result is {}".format(res_fwd_jit))

        # Backward AD w.r.t. y
        res_back = api.grad(f)(2.)
        print("Grad backward result is {}".format(res_back))
        self.assertEquals(12.0, res_back)  # AD wrt y is x = 3.0


    def testDefBatch(self):
        """Test how batching is implemented."""

        #@api.custom_transforms
        def f(x, y):
            return jnp.vdot(x, y)

        # Customize batching. If the function is a -> b,
        # then
        def fun_batch(batched_args, batch_dims, **params):
            # TODO
            return (onp.array([1.0]), 0)
        # batching.primitive_batchers[f.prim] = fun_batch

        arg1 = onp.ones([3])
        arg2 = onp.ones([3]) * 2
        batched_f = api.vmap(f, in_axes=0, out_axes=0)
        #res_batch = batched_f(arg1, arg2)
        #print("Batch result is {}".format(res_batch))

        #print("Shape is: ", api.eval_shape(batched_f, arg1, arg2))
        #print(api.make_jaxpr(batched_f)(arg1, arg2))

        res_batch_jit = api.jit(batched_f)(onp.array([arg1, arg1 * 2]), onp.array([arg2, arg2 * 3]))
        #
        #self.assertTrue(onp.allclose(onp.array([[10.0, 40.0], [90.0, 160.0], [250.0, 360.0]]), res_batch))

    def testDefJVPBuiltIn (self):
        """Test how defjvp is implemented."""

        def f(x):
            return jnp.sin(x)

        # Forward AD
        res_fwd = api.jvp(f, (2.,), (0.2,))
        print("Grad forward result is {}".format(res_fwd))

        # JIT of forward
        res_fwd_jit = api.jit(lambda x, x_t: api.jvp(f, x, x_t))((2.,), (0.2,))
        print("JIT of Grad forward result is {}".format(res_fwd_jit))

        # Backward AD w.r.t. y
        res_back = api.grad(f)(2.)
        print("Grad backward result is {}".format(res_back))
        self.assertEquals(-0.41614684, res_back)  # AD wrt y is x = 3.0


    def testVarious(self):
        def f(x):
            assert x.shape == (2,)
            v = (x[0] == 5)
            if 3 == 2:
                return x[0] + x[1]
            else:
                return x[0] - x[1]
        x = onp.array([2.0, 3.0])
        f_grad = api.grad(f)
        f_jit = api.jit(f)

        print("Result of grad is {}\n\n".format(f_grad(x)))
        print("Result of jit is {}\n\n".format(f_jit(x)))
        # print("JAXPR is {}".format(api.make_jaxpr(f_grad)(x)))

import dis

class MADirect(object):
    def ma(self, x, y, z):
        return jnp.add(jnp.multiply(x, y), z)


def one_transform(name="", func=None, args=None, expected=None):
    print("\n*** Transformation: {}".format(name))
    #print("\nmake_jaxpr computation:")
    #print("\nJAXPR: ", api.make_jaxpr(func)(*args))
    _trace_indent("\nEvaluate transformation {}:".format(name))
    res = func(*args)
    _trace_unindent("result of transformation {} = {}".format(name, res))
    assert onp.allclose(expected, res)

def some_transforms(ma):
    @trace("sq_add")
    def sq_add(x, y):
        return ma.ma(x, x, y)

    one_transform("none",
                  func=sq_add,
                  args=[2., 10.],
                  expected=14.)

    one_transform(name="JIT",
                  func=api.jit(sq_add),
                  args=[2., 10.],
                  expected=14.)

    one_transform(name="JIT of JIT",
                  func=api.jit(api.jit(sq_add)),
                  args=[2., 10.],
                  expected=14.)

    one_transform(name="JVP",
                  func=lambda xs, xts: api.jvp(sq_add, xs, xts),
                  args=[(2., 10.), (1., 1.)],
                  expected=(14., 5.))

    one_transform(name="JIT of JVP",
                  func=api.jit(lambda xs, xts: api.jvp(sq_add, xs, xts)),
                  args=[(2., 10.), (1., 1.)],
                  expected=(14., 5.))

    one_transform(name="GRAD",
                  func=api.grad(sq_add),
                  args=[2., 10.],
                  expected=4.)

    one_transform(name="JIT of GRAD",
                  func=api.jit(api.grad(sq_add)),
                  args=[2., 10.],
                  expected=4.)

    one_transform(name="VMAP",
                  func=api.vmap(sq_add, in_axes=0, out_axes=0),
                  args=[onp.array([2., 3.]), onp.array([10., 20.])],
                  expected=[14., 29.])

    one_transform(name="JIT of VMAP",
                  func=api.jit(api.vmap(sq_add, in_axes=0, out_axes=0)),
                  args=[onp.array([2., 3.]), onp.array([10., 20.])],
                  expected=[14., 29.])

    one_transform(name="VMAP of VMAP",
                  func=api.vmap(api.vmap(sq_add, in_axes=0, out_axes=0),
                                in_axes=1, out_axes=1),
                  args=[onp.array([[2., 3.], [4., 5.]]),
                        onp.array([[10., 20.], [30., 40.]])],
                  expected=[[14., 29.], [46., 65.]])


class MAPrimitive(object):
    def __init__(self):
        self.ma_p = core.Primitive("ma")

        # The primal evaluation rules
        @trace("ma_impl")
        def ma_impl(x, y, z):
            """Primal implementation of the primitive.

            """
            return onp.add(onp.multiply(x, y), z)
        self.ma_p.def_impl(ma_impl)

        # The abstract evaluation rules
        @trace("ma_abstract_eval")
        def ma_abstract_eval(xs, ys, zs):
            assert xs.shape == ys.shape
            assert xs.shape == zs.shape
            return abstract_arrays.ShapedArray(xs.shape, xs.dtype)
        self.ma_p.def_abstract_eval(ma_abstract_eval)

        # The compilation rules
        @trace("ma_xla_translation")
        def ma_xla_translation(c, xc, yc, zc):
            return c.Add(c.Mul(xc, yc), zc)
        xla.backend_specific_translations['cpu'][self.ma_p] = ma_xla_translation

        # JVP
        @trace("ma_value_and_jvp")
        def ma_value_and_jvp(arg_values, arg_tangents):
            # Run the custom function through JAX to get the primal output value
            x, y, z = arg_values
            xt, yt, zt = arg_tangents
            _trace("Primal evaluation:")
            primal_out = self.ma(x, y, z)
            _trace("Tangent evaluation:")
            # We do need to deal specially with Zero.
            def make_zero(xt):
                return onp.zeros(onp.shape(xt), dtype=onp.float32) if xt is ad.zero else xt

            output_tangent = self.ma(make_zero(xt), y, self.ma(x, make_zero(yt), make_zero(zt)))
            return (primal_out, output_tangent)

        ad.primitive_jvps[self.ma_p] = ma_value_and_jvp

        @trace("ma_transpose")
        def ma_transpose(ct, x, y, z):
            """
            Args:
                ct: the cotangent of the output of the primitive.
                args: tuple of
            Returns:
                the cotangent of the inputs
            """
            if x is not ad.undefined_primal:
                # This use of ma is with a constant "x"
                assert y is ad.undefined_primal
                ct_y = ad.zero if ct is ad.zero else self.ma(x, ct, onp.zeros(onp.shape(x), dtype=onp.float32))
                res = None, ct_y, ct
            else:
                # This use of ma is with a constant "y"
                assert x is ad.undefined_primal
                ct_x = ad.zero if ct is ad.zero else self.ma(ct, y, onp.zeros(onp.shape(y), dtype=onp.float32))
                res = ct_x, None, ct
            return res

        ad.primitive_transposes[self.ma_p] = ma_transpose

        @trace("ma_batch")
        def ma_batch(vector_arg_values, batch_axes, instance=None):
            # Cheating, we use the fact that the primitive works on vectors as well
            assert batch_axes[0] == batch_axes[1]
            assert batch_axes[0] == batch_axes[2]
            _trace("Using ma to compute the batch:")
            res = self.ma(*vector_arg_values)
            return res, batch_axes[0]

        batching.primitive_batchers[self.ma_p] = ma_batch

    def ma(self, x, y, z):
        return self.ma_p.bind(x, y, z)


class ExperTest(jtu.JaxTestCase):
    """Tests to experiment."""

    def test1(self):
        def callee(x):
            return x + 1
        def fun(x):
            if x > 0:
                return x + x
            x = callee(x)
            sum = 0
            for i in range(len(x)):
                sum += x[i]
            return sum

        dis.show_code(fun)
        dis.dis(fun)

    def test_ma_direct(self):
        ma = MADirect()
        some_transforms(ma)

    def test_ma_primitive(self):
        ma = MAPrimitive()
        some_transforms(ma)


    def test_pmap(self):
        print(api.devices())
        f = api.pmap(lambda x: x**2, devices=api.devices())
        f = api.pmap(lambda x: x**2, devices=api.devices())
        print(f(onp.ones(4)))  # => [8 8 8 8]

if __name__ == '__main__':
  absltest.main()