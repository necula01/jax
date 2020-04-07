Mini-JAX: a pedagogical model of JAX tracing and transformations
===============================================================

This project started with the goal to provide a (loose) model of how JAX
works: tracing Python functions using a combination of concrete and abstract 
evaluation into a typed intermediate form on which one can apply composable 
transformations (GRAD, JVP), in conjunction with JIT compilation. I first
tried to describe all this using abstract notation, but it gets complicated
quickly, and I thought that an executable model will be easier to get right. 
This started as a learning project, and hopefully it can
be a teaching project also, for people who want to understand how JAX works 
internally. In order to keep mini-JAX simple, a number of features
and optimizations have been simplified or omitted:

* JAX features that are included in mini-JAX:

  * Tracing with a combination of static and dynamic arguments, including 
    cases when values are captured from outer Python scopes. Tracing builds
    an intermediate representation of the code. Tracing also carries concrete
    Python values that are used to resolve control flow for most transformations.
  * JIT transformation is lazy, in the sense that applying transformations on
    the result of JIT results in the JIT function body being transformed. The
    actual compilation happens as late as possible, once all transformations
    have been applied.
  * A conditionals higher-order primitive.
  * The following composable transformations: JVP (forward differentiation),
    GRAD (reverse differentiation), VMAP (vectorization), 
    FLOPS (a new toy transformation to estimate
    the FLOPS cost of the code, without
    actually running the computation, except as much as needed for when there
    are data-dependent FLOP counts.)   
  * Reverse differentiation (GRAD) is careful to ensure that the cost of the
    reverse differentiation is within a constant-factor of the cost of the forward
    computation.
  * All transformations work with arbitrary Python control-flow, except when 
    JIT is involved or when concolic testing is disabled. This means that 
    mini-JAX can perform JVP and GRAD through Python control-flow that depends
    on the data being differentiated. 
  * The result of tracing user functions and applying transformations is cached.      
  * Only the `float` type and `numpy.ndarray(float)` types are supported and only
    a few arithmetic operators (`+`, `-`, `*`, `**`). The arithmetic operators
    do not implement implicit broadcasting.
    
* Omitted JAX features and optimizations:

  * In mini-JAX all transformations on a function are performed after tracing 
    the function into the intermediate language. 
    This is called an "initial embeddeding" in JAX. 
    In real JAX, some transformations, e.g., JVP and VMAP, are performed online 
    as the JAXPR is generated (and often the JAXPR is not even materialized). 
    See below for a longer discussion of this difference between JAX and 
    mini-JAX.
  * Most higher-order primitives (`scan`, `while_loop`) are missing.  
  * There is no support for PyTrees, only floats, tensors of floats, 
    and tuples thereoff can be returned.
  * JIT in mini-JAX here means compiling to an executable Python string, no XLA.
  * There is no PMAP.
  * No support for custom transforms (overriding the transformation for various
    primitives).
  * no support for `static_argnums` (for JIT) and `argnums` (for GRAD), although
    this can be obtained with partial application functions.
  * there is no constant folding, beyond the part that is done by Python 
    outside the tracer's control.
  * perhaps other missing features?

## Highlights and lessons learned

The goal was to implement mini-JAX as simply as possible, but not simpler! It is
easy to end up with an implementation that is too simple. I wrote up here 
some of the things that are non-obvious. These lessons are meant to be relevant to both 
mini-JAX and JAX, but are perhaps easier to understand from the mini-JAX code. 

### Introduction to JAX tracing

Assume that we want to code up a forward-differentiation (a.k.a., JVP or 
Jacobian-vector-product) transformation such that given a 
Python function `traceable_func`, then `jvp(traceable_func, arg, arg_tan)` 
evaluates the perturbation of the result of `traceable_func` evaluated at 
input `arg` given a perturbation of the input `arg_tan`. For example:
```
   def func0(x):
     return x * x
```
Then `jvp(func0, a, a_tan)` evaluates to `2 * a * a_tan`.

Mini-JAX performs all function transformations on an internal representation
of the function, called `Expr` (symbolic expressions). In real-JAX there is a similar
data structure called JAXPR (JAX expressions); for specific
details in JAXPRs, see the "Understanding JAXPR" writeup. 
In real-JAX some transformations
are fused with the tracing pass and happen while tracing. 
We discuss this difference further below, for
now we assume that we always trace to symbolic expressions then apply transformations. 
The symbolic expressions have constructs such as `Var(v)` (for variables, 
where `v` is some internal unique identifier), 
`Literal(3.0)`, `Add(e1, e2)`, etc.  

To obtain the symbolic expression for the function to be transformed,
JAX does not try to parse the source code of the function.
Instead, it executes the function using special tracer objects as arguments. 
A tracer object has a type, e.g., `float`, and a symbolic expression that denotes the 
current value of the tracer. In many cases the tracer also carries the actual 
concrete value. 
The tracer object overloads
Python operators such that it gets notified when the Python interpreter
tries to use the tracer object in an operation. The tracer implementation of 
operators
builds symbolic expressions cbased on the operation and symbolic 
expressions for the arguments. The result is returned as a tracer. 

The pseudo-code for how the tracing is invoked (see `Function.trace_user_function`)
is roughly:
``` 
  arg_t: Tracer = new_var_tracer_from_val(arg)  # Create a tracer initialized to a Var expression 
``` 
Now we simply let the Python interpreter execute the user function: 
```
  primal_res_t: Tracer = traceable_user_func(arg_t)
```
The result `primal_res_t.expr` is a symbolic representation of the computation 
that was performed on tracers by the user function. 
For a user function to be adequate for such tracing it must: 

* use the arguments only with a designated set of operators (that
  have been overloaded.) In mini-JAX, these are `+`, `-`, `*`, 
  and `**` with integer 
  exponent. JAX has a much richer set of primitive operations, and most of 
  the `numpy`
  API has been implemented in terms of these primitive operations.   
  An example of inappropriate operations for tracing are library functions such as
  `copy` or `pickle`. 
* the result of the function should be returned through the function return value, 
  not through global state. The function can store values in mutable state 
  internally, but those values should not be used after the function returns, 
  and the result should be returned through the return value. 

Normally, JAX
traces through function calls, including nested ones. For example, when tracing 
the function `func1` below::
```
  def func_lexical_closure(x):
    z = x * 2
    def inner(y):
      return y + x * 4 + z
    return inner(x * 3)
```    
the resulting expression is essentially `x * 3 + x * 4 + x * 2` (tracers are passed
transparently through variable assignments, function calls, and lexical closures). 
In particular, in this form of tracing, all the function calls are inlined. 

### Handling control-flow with concolic tracing

For control-flow the situation is a bit more complicated. Loops and control flow 
that depend only on constants are inlined by the Python interpreter 
(in general, all operations that do not depend on the function arguments
are performed directly by the Python interpreter without any involvement, 
or even awareness, from the tracing machinery):
```
  def func_static_control_flow(x):
    for i in range(4):
      if i % 2 == 0:
        x = x * 2
      else:
        x = x * 3
    return x
```
The above function traces to `(((x * 2) * 3) * 2) * 3`. 

Consider the following control-flow that depends on function arguments:
```
  def func_dynamic_control_flow(x):
    z = x * 2
    if z >= 0:
      return z * 3
    else:
      return z * 4
```

In order to handle this, JAX (and mini-JAX) tracers carry not only the 
symbolic expression denoting the way the current value was computed, but also the 
actual concrete Python value itself. This is possible because tracing happens only once 
the actual arguments are available: `jvp(func_dynamic_control_flow, 3, 1)` 
(here `3` is the value of the argument `x` at which the JVP is being calculated,
and `1` is the value of the argument perturbation.)

The tracer constructed for `x` will have a `Var` symbolic expression and will also 
contain the concrete value `3`. The value of `z` will be a tracer with 
symbolic expression `x * 2` and concrete value `6`.
The concrete values of the tracers are used only for computing the concrete
values of derived tracers, and ultimately for resolving control flow. 
The body of the function above will be traced to 
the symbolic expression `(x * 2) * 3`. For a negative concrete value of `x`, we
would obtain the symbolic expression `(x * 2) * 4`. This form of combining
concrete evaluation with symbolic evaluation is called *concolic* tracing. 

Carrying concrete values has a cost: the tracer needs to perform 
each operation as it is traced. This may be more expensive than just building 
the symbolic expression, e.g., when the operands are large tensors. In mini-JAX
all transformations use non-concolic tracers by default, but most transformations
can be told to use concolic testing by setting the transformation parameter
`abstract = False`. 
This default needs to be changed whenever the traced function has control-flow
that needs to be resolved based on concrete values. Note that turning on 
concolic testing also turns off caching of the transformations (see below).

In real JAX it depends on the transformation whether the tracers carry concrete
values: JVP, GRAD, VMAP use concolic tracers, while JIT and PMAP use abstract 
tracers. 

### Composable transformations through traceable interpreters

Once we have the symbolic expression for a function to be transformed, one
can think of writing transformations as a expression transformer followed
by evaluating the resulting expressions. JAX (and mini=JAX) fuses the 
transformation with the evaluation, relying on the expression construction during
tracing of the evaluation for constructing the transformed expression. 

The simplest evaluator is the standard evaluator, that computes the value
of the symbolic expression given values for arguments of the expression: 
(code in `Expr.eval_std_operator`):
```
  def eval_std(e: Expr, args_v: List[Value]) -> Value:
    if e is Literal(c):
      return c
    elif e is Add(e1, e2):
      return args_v[0] + args_v[1]
    ...
```

Continuing the `jvp(traceable_func, arg, arg_tan)` example, the actual 
transformation is performed lazily by evaluating the symbolic
expression `primal_res_t.expr` with a non-standard evaluator. 
In our case we are going
to write an `Jvp.eval_operator` function that takes an expression and computes its 
actual value and tangent given actual values and tangent values for the
arguments.
```
    def Jvp.eval_operator(e: Expr, args_and_tan: List[Tuple[Value, Value]]):
        args_v, args_tan = unzip(args_and_tan)
        if e is Literal(c):
          return (c, 0.)
        elif e is Add(e1, e2):
          return (eval_std(e, args_v), 
                  args_tan[0] + args_tan[1])
        elif e is Mul(e1, e2):
          # Here we need the standard values for sub-expressions
          return (eval_std(e, args_v), 
                  args_tan[0] * args_v[1] + args_v[0] * args_tan[1])
```

Returning to our `func0` example, after the tracing of the body of `func0`
we obtain the symbolic expression `func0_e = Mul(Var(v), Var(v))`. 
To evaluate the tangent at point `args_v = 3` with a perturbation value `arg_tan = 4`,
i.e., `jvp(func0)(3, 1)`, we set the variable `Var(v)` to `(3, 4)` and we
invoke `Jvp.eval_operator` bottom up. The most important detail is that 
the `+` and `*` operations in the implementation of the custom evaluator 
will operate 
on concrete Python values and will be performed directly by the interpreter
to yield the value `24`. 

What happens if we want to nest transformations (during the tracing of
a function being transformed, we encounter another transformation)? 
This kind of behavior arises naturally if one wants to compute the 
second-derivative of a function:
```
  def func1(y):
    def func0(x):
      return x * x
    return jvp(func0, y, 1)
  second_derivative_at_3 = jvp(func1, 3, 1)
```
Here, `func1` is being traced as part of `jvp(func1)`, and while tracing it we
encounter `jvp(func0)`. At that 
point a nested tracing process starts, a new tracing variable is created for
`x` and `func0` is traced. The rough sequence of steps that happen are: 
```    
  arg_y_t: Tracer = new_var_tracer_from_val(3)
  res1_t : Tracer = func1(arg_y_t)
     # during func1(arg_y_t) the following happens:
     arg_x_t: Tracer = new_var_tracer_from_val(arg_y_t) # The concrete value 3 is carried over
     res0_t: Tracer = func0(arg_x_t)     # = 'arg_x_t * arg_x_t'
     # Evaluate the traced expression using Jvp.eval_operator, with the 
     # value of the arg_x_t mapped to a tuple of value and its perturbation.
     res0_jvp_t: Tracer = Jvp.eval_expression(res0_t.expr, {arg_x_t: (arg_y_t, 1)})
     # Here res0_jvp_t will be the Expr '1 * arg_y_t + arg_y_t * 1'
     res1_t = res0_jvp_t
  # Now apply the jvp(func1)
  res1_jvp_t = Jvp.eval_expression(res1_jvp_t, {arg_y_t: (3, 1)})
  # Here res1_jvp_t is the concrete value 2 (1 * 1 + 1 * 1)  
``` 

Note that in the first invocation of `Jvp.eval_expression(res0.expr)` some input
values are tracers (`arg_y_t`), so the result is a tracer. 
In the second invocation `Jvp.eval_expression(res1_t.expr)` all values
are concrete values, so the `+` and `*` in the definition of `Jvp.eval_operator`
are executed directly by the Python interpreter and the result is the 
concrete value `2`.

The key is that the non-standard evaluators implementing the transformations
must be traceable themselves: they must be functional and use their arguments
only with supported primitives. If you follow this rule, then you can write
new transformations that compose nicely with the other ones.  

### Many Python numerical contain statically-well-typed computations waiting to come out

The classic adage "well-typed programs don't go wrong" comes in very handy here.
We would like to have a static type system for our symbolic expressions such 
that (1) the output of tracing is a well-typed expression, (2) a well-typed 
expression can be evaluated without error, can be JIT-compiled and executed
without error, and can be transformed without errors into other 
well-typed expressions.
Here, by "error" we mean some internal errors in the JAX transformation and
compilation machinery,
excluding data-dependent run-time errors such as division-by-zero. 

The reason why this type safety property is very useful is that we want to 
give all the JAX errors during the tracing (during the invocation of the 
user function on tracers)
because that is when the Python interpreter executes user code and can 
localize the error with precise stack traces. If instead, we allow 
any of our standard or custom evaluators and JIT-compilers to fail later, the 
stack trace will contain only JAX internal code. 

JAX's internal representation JAXPR is typed, and so is mini-JAX's internal
`Expr` language, with types representing the shape and base type of a tensor. 
In fact, the real JAX has a richer hierarchy of types that are called
`abstract values`. 

One of the secrets of the success of JAX's tracing-based approach to transformations
is that many `numpy` numerical and ML programs
can be easily written with Python control-flow resticted only on array shape values. 
Thus one can trace the code with tracer values that are aware of shapes by otherwise
abstract the concrete values to extract
a statically-typed symbolic expression specialization of the original program.

The JAX type system is not quite type safe in the sense described above. For
example, the GRAD transformation is not defined for the whole JAXPR language,
hence some transformation errors may arise after tracing. 

### The need for functions in the intermediate language

With the tracing described so far all Python control flow, including functions
and conditionals get inlined and the expression language denoting the
computation does not need to have control-flow, functions, or scopes. This 
simple semantics would make it very easy to write transformations. 

In reality, JAXPRs (and mini-JAX `Expr`) have scopes and function calls, for
several reasons. 

First, the JIT transformation requires a portion of the computation to be 
compiled, perhaps for a specific device or hardware. For example, the `jit`
transformation in the example below requests the compilation of the body 
of `inner`.  
```
  def func_compile_inner(x)
    z = x * 2
    def inner(y):
      return y + x * 4 + z
    return jit(inner)(x * 3)
```   
Furthermore, the compilation cannot all happen during tracing because if there
are enclosing transformations, e.g., `grad(func_compile_inner)`, then the computation required
by `grad` has to be performed in the same JIT compilation scope as `jit(inner)`. 
This means that upon encountering a `jit` transformation the expression 
corresponding to the function is captured as a function in the intermediate
language. (It is not desirable to require `jit` to always be the top-most 
transformation, because we may want to put the `jit` in a library function, 
while allowing the library user to request a gradient computation, or any
other transformation.)

The JAXPR language has functions, but is not really higher-order, 
because functions are not first-class objects. They can only be used as 
parameters to a few special higher-order operators (in mini-JAX: `Operator.JIT_CALL`,
and `Operator.COND_GE`; in JAX, the similar control-flow and compilation-related
operators). The functions in mini-JAX are represented as instances of 
`Function` objects (containing a set of symbolic expressions for the input 
arguments and a set of symbolic expressions for the function results).

The functions in JAXPR and mini-JAX `Expr` are:
* closed (all the data they need is passed explicitly through parameters); this 
  is achieved through a closure conversion feature of tracing (see below),
* typed,
* anonymous, and called in a single place. Through caching (see below) we 
get some re-use of `Function` objects. 

In addition to JIT, in JAX and mini-JAX the conditionals also take advantage 
of functions in the intermediate language. 
Since we rely on tracing using the Python interpreter to capture the 
symbolic expressions, we will not see regular Python control flow. Instead, 
we ask the programmer to use a higher-order conditional construct to capture
both branches of a conditional as functions:
```
jax.ops.cond(pred, true_arg, true_func, false_arg, false_dunc) 
```
The semantics is: `if pred then true_func(true_arg) else false_func(false_arg)`.
This operator is strict in the `pred`, `true_arg`, and `false_arg` in the sense
that they are evaluated no matter which branch is taken. 

The above concrete syntax is mostly due to the limitations of Python tracing. 
In JAX and mini-JAX we choose an internal representation that carries two 
functionals for the branches. As a further incentive
to use a higher-order operator, the HLO language to which JAX compiles
has a similar higher-order construct. There are other ways of course to encode 
conditionals in the intermediate language, e.g., by introducing a lazy 
conditional operator.

The higher-order operators are the most complicated in the intermediate language. 
For example, the `grad` and `jvp` rules for conditionals are 30 and respectively
20 lines of code, compared to two lines for each of the arithmetic operations.    

### Tracing must consider closure conversion for nested Python functions

Once we introduce functions in the expression language, we have to 
decide how to handle capturing computations from the Python nested lexical
scopes. Consider again the following example of a `jit`:
```
  def func1(x)
    z = x * 2
    def inner(y):
      return y + x * 4 + z
    return jit(inner)(x * 3)
```

Clearly we want the `x * 2` computation outside of the `jit`, and presumably, 
the programmer intends to have the `x * 4` computation inside the jit, even 
though the data dependencies for `x * 4` are ready before we enter the `jit`
scope. 
This is not a clear design requirement though. For example, should the gradient
computations arising from code inside a `jit` be also kept strictly under `jit`, 
even if they are mere accumulation of constant factors? 

JAX and mini-JAX do different things here. JAX does pretty aggressive constant
folding, meaning that it does computations as soon as their dependencies are ready,
which may mean that the computation happens outside the body of a jitted function.
This can be viewed as an optimization, but users are sometimes surprised
when they see multiple JIT-compilations for sub-expressions, when they 
expected only a single JIT-compilation. There is ongoing design exploration
in JAX for how to handle this (perhaps forcing some computations
to stay in the JIT scope in which they were written.)

In mini-JAX, for simplicity, all computations are kept in the tracing scope
in which they appear. Note that we are not talking about Python lexical 
scoped here, but the dynamic scopes that correspond to JAX transformations. 
In mini-JAX all computations have a dependency on the 
dynamically-nearest enclosing transformation scope (all transformations,
not just JIT).

Essentially the code above traces to 
`xla_call (lambda y, x, z: y + x * 4 + z) (x * 3) x (x * 2)`.
Note that the body of `inner` is kept as an anonymous function. Also, 
the body is "closed", and is passed explicitly the *environment values* for `x`
and `z`. The computation for `x * 4` is kept in the body. 

This is not trivial to achieve. From a pure data dependence perspective, 
when the Python interpreter encounters `x * 4` it looks the same as if it 
were written outside `inner`. We solve this by introducing global state
in the tracing: each time a new transformation scope is entered, a
global counter is incremented and when the symbolic expression for `x * 4`
is build, it is tagged with the scope nesting depth. When the final result 
of tracing `inner` is obtained, all the computations in it that belong to 
enclosing scopes are lifted out of the function. For example, the symbolic 
expression for `x * 2` constructed outside `inner` is encountered in the 
computation of the return value of `inner`. At that point it is replaced
with a fresh variable, and lifted as a new parameter to `inner`. 
This code is in `mini_jax.py:Tracer.closure_convert`. 

The functional branches of conditionals are handled similarly. 

TODO: one can argue that for conditionals and for JIT is is important to 
keep computations dependent on the dynamic scope where they were created, 
but perhaps for other computations we can follow strict data dependence
and allow the optimizer to lift them as needed. Perhaps one can explore 
such a mixed strategy in mini-JAX (and JAX), but for simplicity this is not
yet done. 

### Transformations and choice of primitives

The choice is primitives (operators) depends on the choice of transformations
to be supported: the existing operators must be sufficient to express the
result of transforming all well-typed applications of the operators. 

Consider the JVP operation, defined as follows, for an expression `e`
depeinding on two variables `x1` and `x2`: 
```
   jvp(e)((x1, x2), (x1t, x2t)) = (de/dx1)(x1, x2) * x1t + (de/dx2)(x1, x2) * x2t  
```   
(We write `jvp(e)` for the transformed expression; 
(de/dx) is the expression that denotes the derivative of `e` w.r.t. `x`.)

It turns out that addition is closed under `jvp`, but multiplication is not, 
in that it requires addition:
```
  jvp(e1 + e2)(x, xt) = jvp(e1)(x, xt), jvp(e2)(x, xt)
  jvp(e1 * e2)(x, xt) = jvp(e1)(x, xt) * e2 + e1 * jvp(e2)(x, xt)
```

However, the union of addition and multiplication is closed under `jvp`.

VJP is defined (simplistically) as follows:
```
  vjp(e)((x1, x2), yadj) = ((de/dx1)(x1, x2) * yadj, (de/dx2)(x1, x2) * yadj)
```

(`yadj` is the adjoint for the result of `e`)

It turns out that addition and mutliplication are closed also under `vjp`. Adding
a power operation (`e ** n`) to the set keeps it closed. Same for subtraction. 

The story becomes more interesting with vectorization, or `vmap` (see details
in `mini_jax_vmap.py` and `tests/mini_jax_vmap.py`).
 
Vectorization, or `vmap` is a transformation for a function that
takes arguments of rank k to a function that takes arguments or rank `k+1`.
Vectorization is defined technically as follows
(we use upper-case letter for index tuples, `,` as a tuple
and element concatenation, and `a I` for the indexing operation, i.e.,
):

>  If `e` is an expression that depends on free variable `x` of shape `S`,
>  and `b` is a positive integer, 
>  then `vmap(e, b)` is an expression that depends on a free variable
>  `xv` of shape `b,S` such that, for any array `a` of shape `S`:
```
      vmap(e, b) xv (i,I) = e(xv i) I  , for i in range(b)
```

(we used `xv i` as the partial application of `xv` to the index `i`, i.e., 
`(xv i) I = xv (i, I)`.

Most operators are already rank-polymorphic, so they do not need to be
changed. Essentially:
```

  vmap (op x y, b) = op (vmap(x, b)) (vmap(y, b))
```
In mini-JAX two complications arise:

First, we support vectorizing a function w.r.t. only a subset of its
arguments. This is needed we apply `vmap` to an internal function that
captures some values from the environment:
```
   y = 5
   def fun(x):
     # Assume `fun` was written to operate on scalars `x` and `y`
     return x + y
   vmap(fun, 4)(np.arange(4.))  # vmap(fun) operates on a vector argument
```
Here, the `x + y` operator must be changed to first broadcast the scalar
`y` to the same vector size as `x`, then applying the rank-polymorphic
`+` operator on vectors.

To support broadcasting one may attempt to add a new primitive `bcast0(a, b0)`
that broadcasts the array `a` around a new leading dimension of size `b0`.

>  If `a` has shape `S` then `res = bcast0(a, b)` has shape `b,S` and
>  for any I of size `|S|` and `i in range(b0)` we have `res (i,I) = a I`.

This is not enough though because `bcast0` is not rank-polymorphic. When
applying `vmap(bcast0(a, b0), b1)` we would need `bcast1(vmap(a, b1), b0)`,
i.e., broadcasting along axis 1. This suggests, that we need to generalize
broadcasting to `bcastdim(a, d, b)` (broadcast to size `b` in dimension `d`
of the resulting array):

For an array `a` and any indices sets `I`, `J`, (`|I| = d` and `|J| = |S| - d`),
```
bcastdim(a, d, b) (I,i,J) = a (I,J)  , for i in range(b).
```

This generalized broacast operator is closed under `vmap`:
```
  vmap(bcastdim(a, d, b0), b1) = bcastdim(vmap(a, b1), d+1, b0)
```

When computing the reverse AD for `bcastdim` we need to sum-up the
contributions to the adjoint of all elements in the dimension `d`.
Thus, we need to introduce a `sumdim` operators:

>  If `a` is an array of shape `(S1,b,S2)` with `|S1|=d`,
>  `sumdim(a, d)` has shape `(S1,S1)`, and
>  `sumdim(a, d) (I,J)` = sum of `a[I,i,J] for i in range(b)`

We can now define:
```
  vjp(bcastdim(a, d, b0)) = sumdim(vjp(a), d)
```
Fortunately, `sumdim`'s VJP is defined in terms of `bcastdim`:
```
  vjp(sumdim(a, d)) = bcastdim(vjp(a), d, b)
```
where `b` is the `d`th element of `shape(a)`. Also, `sumdim` is
closed under VMAP:
```
  vmap(sumdim(a, d), b) = sumdim(vmap(a, b), d+1)
```

Another complication is that `cond_ge` is defined only
for scalar predicates. When we vectorize, it is possible
to need to vectorize the predicate. Thus, we generalize
`cond_ge` into `where_ge`:

>  If `a`, `t`, `f` are arrays of shape `S`, then
```
where_ge(a, t, f)[I] = if a[I] >= 0 then t[I] else F[I]
```

Unlike, `cond_ge` which only executes one branch of the conditional,
`where_ge` executes always both branches and then picks the right
element from each branch. `where_ge` is rank-polymorphic so it
is closed under VMAP. It is also closed under VJP.

The final complication we discuss here is that if a function has
multiple outputs, and some of the outputs depend only on
unvectorized inputs, then we want to leave the outputs unvectorized.
This is true for the internal functions (e.g., in CALL_JIT, or COND_GE).
At the top level (see `vmap` below), we vectorize all outputs. In real-JAX,
there is a parameter to request some of the outputs to be left
unvectorized (only if the do not depend on vectorized inputs).

These operators are defined in `mini_jax_operators.py`.

### Ahead-of-time transformations through caching

Caching of transformations in JAX is more than a simple optimization. Since 
tracing and transformations happen just-in-time (once the actual arguments
to the transformed functions are available and the transformed function is invoked),
one way to achieve ahead-of-time transformations is to cache the result of 
the transformation upon the first use, and then reuse the transformation from 
cache on subsequent compatible uses.

In mini-JAX there are 3 places where caching happens:
* When tracing a user function the resulting `Function` object is cached, but only if
  all the following are true:
  
  * the tracing is abstract (not concolic). In presence of concolic tracing
  (controlled by the `abstract` parameter to the user transformations) the 
  tracing is able to follow Python control-flow, and subsequent invocations
  with different values of arguments may take a different path through the 
  function. At the moment we do not 
  keep track which control-flow was followed, so for safety we do not cache 
  concolic tracing results. Note that the `jit` transformation never uses
  concolic testing, precisely so that the result can be cached or else
  the tracing would incur the cost of executing all the computations in the 
  function and it would be pointless to compile it afterwards.
  * the traced function does not capture tracers from outer tracing. This 
  would happen if the function being traced is nested in another 
  Python function and refers to state of the enclosing function, and 
  the enclosing function is itself being traced for some other transformation. 
  Anyway caches are not very effective on nested functions, as explained below. 
  * the keyword parameter `cache` is set to True. This parameter is useful for
  disabling caching when the function being traced may depend on global state, 
  or for deterministic testing. 
  
  The cache for tracing is associated with the user function object instance 
  being traced. 
  This means that nested functions will not share the cache between 
  different invocations of the enclosing function. Similarly, caches will 
  not be shared between lambdas. 
  
* The result of transforming a `Function` is always cached. This is safe
  because transforming functions is deterministic. This cache is 
  attached to the `Function` object itself, and the key is just the 
  transformation name (in mini-JAX transformations are not parameterized).
  
* The result of JIT-compiling a `Function` is always cached, as if it
  were a transformation.
  
It is pretty easy to miss the cache. For example, in the code 
`mj.jvp(mj.jit(func))` the result of tracing `func` will be cached attached
to the `func` object (if `func` does not capture tracers from outer
tracing). A subsequent `mj.jit(func)` will yield the same `Function`. However,
the result of `mj.jit(func)` is an anonymous Python function, and the tracing
cache that is part of `mj.jvp` is attached to this volatile object. So, a subsequent
invocation of `mj.jvp(mj.jit(func))` with trace twice `mj.jit(func)`. To 
maximize caching hits, it is best to save and reuse the function returned
by `mj.jit(func)`. 

### Careful sharing in the intermediate language is important

The choice of the intermediate language is important to ensure that the sharing
that arises naturally from the computation is preserved. For example, this 
Python code
`x = x + x; x = x + x; x = x + x` has three operations but with a symbolic
expression representation we may end up with an exponential-sized printed form: 
`((x + x) + (x + x)) + ((x + x) + (x + x))`. Note that the actual symbolic 
data structure will likely contain the proper sharing, since only three `+` nodes
are constructed.   

In mini-JAX I wanted to experiments with simple symbolic expressions:
* it is actually a simpler language: just operators applied to sub-expression arguments.
* it is a functional representation of a whole computation DAG, the expression
 meaning is determined by the meaning of function's variables, we can cache
 results by expression object identity.
* dead-code naturally falls out

With such an expression representation it is crucial to be careful about 
sharing. We achieve this in mini-JAX by making **heavy use of memoization: a
distinct expression instance is processed at most once**. We added a visitor
helper function for expressions to take care of memoization. The cost of 
memoization I believe is approximately the same as using JAXPRs intermediate variables, 
although one can argue that in JAXPRs the cost of exposing sharing is paid only 
when the JAXPR is constructed.   

JAXPRs instead are written with explicit naming of all intermediate computations.
This adds more indirection, makes it somewhat harder to see the expression
in the debugger, and requires a more complex data structure, with things like
topological sorting after transformations. 

In retrospect, I am not sure that the simplicity of representation as symbolic
expressions is worth the extra cost of being careful about preserving sharing.
   
### Real-JAX gives better error messages

Perhaps the most important JAX feature that is not reflected in mini-JAX
is JAX's ability to perform some transformations inline, during tracing. This 
has the major advantages that errors are reported during tracing when the 
user-program's stack trace is available. There is also a likely minor cost 
advantage of 
not having to materialize and traverse multiple times the JAXPR. 

JAX tries hard to do all transformations inline. The only cases when it cannot
do so are if a `jit`, or `pmap` transformation are present, or if we have 
higher-order control-flow (`cond` and `while`).

I have made a quick attempt to refactor the mini-JAX transformations such 
that they can be applied after tracing, or fused with tracing. This seems
possible but the resulting code is hard to read, in part because during tracing
there are several other transformations being applied. One has to be careful
to keep the various transformations separate. A further complication is that
for the `jit` transformation one has to postpone the jitting until all the 
outer transformations have been applied. 

Since in mini-JAX simplicity is a more important goal than efficiency, I decided
to keep all transformations offline. 

### Efficient reverse differentiation is tricky

Implementing reverse differentiation correctly is not hard, if all you care 
is functional correctness. The "complexity correctness" is quite a bit harder.
At some point, I had an implementation that was traversing the expressions
bottom-up
and constructing a gradient (set of partial derivatives) data structure with
addition and scaling. Using memoization for expression, we ensure that 
for each expression instance we construct one instance of a gradient structure. 
As long as all the scaling is by constants, the various
factors can be associated together and all is fine. But as soon as we start
having (1) non-constant scaling factors arising from non-linear expressions, 
and (2) result of an expression used by multiple other expressions, then 

An example code that gives rise to expensive gradient computation was 
(suggested by Dougal):
```
   z = a * (z + z)
   z = b * (z + z)
   ...
```   
The proper way to compute gradients is to traverse an expression only after
the parent expressions have been traversed, and for each we have accumulated
the adjoint. To do this using a symbolic expression language we have
to construct a set of pointers to parents 
(see `mini_jax_grad.Grad.eval_vjp_func_tracer). Thereafter, with careful
memoization things work. 

### Real-JAX implement reverse differentiation from transposition of forward differentiation

A clever insight implemented in JAX is that after forward differentiation the
computation is reduced to a linear combination of the perturbations. Reverse
differentiation can then be implemented by transposing linear combinations
backwards. This reduces the complexity of the code in most cases, but I felt it 
actually complicates the code for non-arithmetic primitives such as conditionals, 
and scans (for which the transpose rule is complex and looses the linear-combination
intuition). In mini-JAX reverse differentiation is implemented directly and is
completely separate from the forward differentiation transformation.

### Hacking mini-JAX was great fun, and humbling!

It is a rare project where 1000 lines of code can pack so much functionality,
so much trickiness, and so much reward. This is in part due to the fact that
tracing leverages much of the power of a Python interpreter, cleverly hijacked
through operator overloading. Some of the trickiness came from having to "undo"
some of the information lost through tracing; access to the source would have
made some of the tasks simpler, but dealing with Python ASTs is significantly
more involved than the intermediate form we use in JAX.

The effort to try to find the simplest correct implementation was non-trivial,
and is but one fraction of the cleverness that is packed into the real JAX!

Code structure
--------------

Mini-JAX is structured as follows:

* General tracing machinery (`mini_jax.py`):

   * `Expr` is a class representing symbolic expressions 
    (similar to, but simpler than, JAXPR in the real-JAX), constructed with 
   `Operator`s applied to `Expr` arguments. All operators are considered to be
   strict in the arguments.
   * In addition to arguments, an operator application has parameters. 
   In essence, each `Operator` represents a parameterized family. For 
   example, the `Operator.LITERAL` has 0 arguments and 1 parameter with the 
   value of the literal. The `Operator.COND_GE` and `Operator.JIT_CALL` carry
   the branch and function as parameters.
   * `ExprType` represents types associated with each `Expr`. In 
    mini-JAX only the `float` type is implemented.  
   * `Function` is a class to represent *closed* functions over `Expr`. In the 
   addition to one or more function results `Expr`, a `Function` also has a 
   set of argument `Expr` of variable kind. 
   * `Tracer` is the class of tracer values. A tracer value has not 
   only the symbolic `Expr` but also information about the scope nesting depth at 
   which it was constructed and the environment (tracer values from shallower
   scope depth that are used), and optionally a concrete value.
   * `Cache` is a simple implementation of caching. The cache is attached to the 
   user-functions or the `Function` objects, so that it gets collected when 
   the related object goes dead. 
   
* `mini_jax_jit.py`: JIT machinery. In the context of 
  mini-JAX, jitting means that we construct a Python executable string, 
  and then we `exec` it. 
    
* `mini_jax_jvp.py`: JVP transformation (forward differentiation).
  The public function exposed is `jvp(func)(a1, a2, a1_tan, a2_tan)`. 
  
* `mini_jax_grad.py`: GRAD transformation (backward differentiation). 
  The public function exposed is `grad(func)(a1, a2)`.

* `mini_jax_flops.py`: A toy FLOPS counting transformation.

* `tests`: unit tests, many showing the symbolic expressions for 
  transformations. 


Anatomy of a transformation
---------------------------

You can look at examples of transformations in the implementation of JVP, FLOPS, 
and GRAD. The rough pseudo-code for how a transformation, say `pickle`, 
is written is shown below:

First we have the public API of the new transformer to be applied to a user function.
We use "before" and "after" to refer to the function and arguments before and after
the transformation.
```
def pickle(traceable_user_func):
  def do_pickle(args_after: List[Value]):
     """The transformed function, taking the arguments after transformation."""
     # Prepare argument types for tracing the traceable_user_func (before transformation)
     # These may be a subset of the args_after (e.g., JVP, where args_after include tangents) 
     args_before : List[Value] = ...  

     # Trace the user function to get the `Function`, and the extra arguments captured
     # from lexical scopes of outer transformations.
     func_before_f: Function, func_env: List[Tracer] = \
         Function.trace_user_function(traceable_user_func, args_before)

     # Now run the pickle evaluator on the closed `Function`
     res: Value = pickle_eval_function(func_before_f, args_after + func_env)
     return res

  return do_pickle
```

Also specific to our transformation is the custom `pickle` evaluator for a 
already traced and *closed* `Function`:
```
def pickle_eval_function(func: Function, args_after: List[Value]) : Value:
  """The transformer custom evaluator for a closed Function"""
  # The values for func's arguments. E.g., for JVP each argument is a 
  # tuple of the actual value and its tangent (all from args_after).
  func_pickle_arguments = ... args_after ... 
  return func.make_evaluator(func_pickle_arguments,
                             eval_operator=pickle_eval_operator)
```

And the custom evaluator for a use of an `Operator`. The `args_v` are the
custom values for the transformed arguments: 
```
def eval_pickle_operator(op: Operator, params, args_v: List[Value]):
  if op == ADD:
    # the result of custom evaluation of ADD. Can use overloaded operators on args_v.
    return ...  
  if op == JIT_CALL:
    # Must push the transformation into the body of the function to be jitted
    func_pickled_f = params.func.transform_function(pickle_eval_function)
    # Construct a JIT_CALL expression, or evaluate the JIT, depending
    # on whether args_v contain tracers. 
    return eval_std_operator(JIT_CALL, {func: func_pickled_f}, args_v)
```

Finally, we have the following important core code that shared by all 
transformations (`mini_jax.py`). 
First is `Function.trace_user_function` to trace a user function into 
an internal `Function` (along with the additional arguments captured from 
outer scopes).
This function is called before every transformation:
```
def Function.trace_user_function(traceable_user_func, args_v: List[Value],
                                 abstract = True, cache = True) -> Function:
    if cache and found_in_cache: return cached_function

    scope_depth = push_scope()  # A new global dynamic scope to detect computations from outer scopes
    args_t : List[Tracer] = map(Tracer.new_var_tracer_from_val(scope_depth), args_v)
    # Trace the user function
    res_t : Tracer = traceable_user_func(args_t)
    pop_scope()
    # Look for captured expressions from outer scope. Those will be new arguments
    res_f: Function, new_args: List[Tracer] = closure_convert(res_t, scope_depth)

    return res_f, new_args  
```

Next is `transform_function`, which is used to trace our custom evaluators, e.g.,
`pickle_eval_function` over the body of an existing `Function`. It is important
to note that `transform_function` is only used for `Functions` that are part of 
deferred computations (in COND_GE and JIT), 
invoked from `pickle_eval_operator`. These functions are closed, so there is
no need to worry about capturing tracers from outer transformations.
 
```
def Function.transform_function(func: Function, evaluator, 
                               extra_args_typ: List[ExprType]) -> Function:
    args_t : List[Tracer] = map(Tracer.new_var_tracer_from_type, 
                                func.vars.typ + extra_args_typ)
    res_t : Tracer = evaluator(func, args_t)
    # No need to closure convert, input `func` is closed.
    return Function(args_t.vars, res_t)
```

TODO
-----

* Make a cleanup pass over the tests
* Improve the property-based testing using hypothesis.
* Explore fusing of forward transformations
* Explore constant folding for JVP and GRAD
* Explore writing JVP with lazy std_eval, like FLOPS
