Mini-JAX: a pedagogical model of JAX tracing and transformations
===============================================================

This project started with the goal to provide a (loose) model of how JAX
works: tracing Python functions using abstract evaluation into a typed
intermediate form on which one can build composable transformations (GRAD, JVP),
in conjunction with JIT. It started as a learning project, and hopefully it can
be a teaching project also, for people who want to understand how JAX works 
internally. In order to keep mini-JAX simple, a number of features
and optimizations have been simplified or omitted:

* JAX features that are included:

  * Tracing with a combination of static and dynamic arguments, including 
    cases when values are captured from outer Python scopes. 
  * JIT transformation is lazy, in the sense that applying transformations on
    the result of JIT results in the JIT function body being transformed. The
    actual compilation happens as late as possible.
  * Conditionals as higher-order primitives.  
  * All transformations are composable.  
  * Reverse differentiation (GRAD) is careful to ensure that the cost of the
    reverse differentiation is within a constant-factor of the cost of the forward
    computation.
  * Forward differentiation (JVP).
  * A new toy transformation to estimate the FLOPS cost of the code, without
    actually running the computation, except as much as needed for when there
    is data-dependent FLOPS.
  
* Omitted JAX features and optimizations:

  * In mini-JAX all transformations are performed after tracing the function 
    to the intermediate language. This is called an "initial embeddeding" in JAX. 
    In JAX, some transformations, e.g., JVP and VMAP, are performed online 
    as the JAXPR is generated (and often the JAXPR is not even materialized). 
    This is an important feature that for now was left out of mini-JAX for 
    simplicity. 
  * only the `float` type is supported and only a few arithmetic operators (`+`, `-`, 
    `*`, `**`). There are no tensors nor `numpy`.
  * no `scan`, `while_loop`.  
  * no support for PyTrees, only floats and tuples of floats can be returned.
  * JIT here means compiling to an executable Python string, no XLA.
  * There is no VMAP, PMAP, Jacobian.
  * No support for custom transforms (overriding the transformation for various
    primitives).
  * no support for `static_argnums` (for JIT) and `argnums` (for GRAD), although
    this can be obtained with partial application functions.
  * there is no caching of transformations or compilations.
  * there is no constant folding, beyond the part that is done by Python, 
    outside the tracer's control.
  * perhaps other missing features?

## Highlights and lessons learned

The goal was to implement mini-JAX as simply as possible, but not simpler! It is
easy to end up with an implementation that is too simple. Here are some of the 
things that are non-obvious. These lessons are meant to be relevant to both 
mini-JAX and JAX, but are perhaps easier to understand from the mini-JAX code. 

### Introduction to JAX tracing

Assume that we want to code up a transformation such that given a 
Python function `traceable_func`, then `jvp(traceable_func, arg, arg_tan)` 
evaluates the perturbation of the result of `traceable_func` evaluated at 
input `arg` given a perturbation of the input `arg_tan`. For example:
```
   def func0(x):
     return x * x
```
Then `jvp(func0)(a, a_tan)` evaluates to `2 * a * a_tan`.

JAX performs function transformations on an internal representation of
the function, called JAXPR (JAX expressions). (Mini-JAX performs all 
transformations that way, but in real JAX some transformations, e.g., JVP
are performed online, during tracing, and no JAXPR is materialized.)
For simplicity, in this write-up (and in mini-JAX) we will not use specific details of
how JAXPR are implemented and are going to simply use symbolic expressions
`Expr` with constructs such as `Var(v)` (for variables, where `v` is some
internal unique identifier), `Literal(3.0)`, `Add(e1, e2)`, etc. For specific
details in JAXPRs, see "Understanding JAXPR" writeup. 

To obtain the symbolic expression for the function to be transformed,
JAX does not try to parse the source code of the function.
Instead, it executes the function using special tracer objects as arguments. 
A tracer object has a type, and a symbolic expression that denotes the 
current value of the tracer. The tracer object also overloads
Python operators such that it gets notified when the Python interpreter
tries to use the tracer object in an operation. The tracer implementation of 
operators builds symbolic expressions corresponding to the computation and
return tracers. 

A pseudo-code for how the tracing is invoked is roughly:
```
  arg_typ: ExprType = type_of_val(arg)    # Abstract 'arg' to type, e.g., 'float', or 'float[3,4]'   
  in_t: Tracer = new_var_tracer(arg_typ)  # Create a tracer initialized to a Var expression 
``` 
Now we simply let the Python interpreter execute the user function: 
```
  primal_res_t: Tracer = traceable_user_func(in_t)
```
The result `primal_res_t.expr` is a symbolic representation of the computation that was performed
on tracers by the user function. For a user function to be adequate for such tracing it must: 

* use the arguments only with a designated set of operators (that
  have been overloaded.) In mini-JAX, this are `+`, `-`, `*`, `**` with integer 
  exponent. JAX has a much richer set of primitive operations, and the `numpy`
  API has been implemented in terms of these primitive operations.   
  An example of inappropriate operations are library functions such as
  `copy` or `pickle`, or Python conditional constructs. 
* the result of the function should be returned through the function return value, 
  not through global state. The function can store values in mutable state 
  internally, but those values should not be used after the function returns, 
  and the result should be returned through the return value. 

Normally, JAX
traces through Python control flow and function calls. For example, when tracing 
the function `func1` below::
```
  def func1(x):
    z = x * 2
    def inner(y):
      return y + x * 4 + z
    return inner(x * 3)
```    
the resulting JAXPR is essentially `x * 3 + x * 4 + x * 2` (tracers are passed
transparently through variable assignments, function calls, and lexical closures). 
In particular, in this form of tracing, all the function calls are inlined. Loops and
control flow are also inlined:
```
  def func1(x):
    for i in range(4):
      if i % 2 == 0:
        x = x * 2
      else:
        x = x * 3
    return x
```
The above function traces to `(((x * 2) * 3) * 2) * 3`. 

### Composable transformations through traceable interpreters

Once we have the symbolic expression for a function to be transformed, one
can think of writing transformations as a expression transformer. This 
requires each transformed to be building expressions, and then we need to 
evaluate the resulting expressions. Instead, JAX fuses the transformation 
with the evaluation, and performs all transformations using custom evaluators. 

The simplest evaluator is the standard evaluator, that computes the value
of the symbolic expression given values for the variables:
```
  def std_eval(e: Expr, std_env: Dict[int, Value]) -> Value:
    if e is Var(v):
      return std_env[v]
    elif e is Literal(c):
      return c
    elif e is Add(e1, e2):
      return std_eval(e1, std_env) + std_eval(e2, std_env)
    ...
```

Continuing the `jvp(traceable_func, arg, arg_tan)` example, the actual 
transformation is performed lazily by evaluating the symbolic
expression `primal_res_t.expr` with a non-standard evaluator. In our case we are going
to write an `eval_jvp` function that takes an expression and computes its 
tangent given tangent values for the variables. First, we prepare a variable 
environment with the input tangents:
```
  tan_env = {in_t.expr: arg_tan}
```

Often during non-standard evaluation we also need the standard value of
a sub-expression. In `eval_jvp` this will be needed when evaluating the tangent
of a multiplication; we need to use the tangents of the operands and also 
their standard values. For this purpose we also set up a standard environment
which we'll use with the standard evaluator:
```
  std_env = {in_t.expr: arg}
```

Then we write the `eval_jvp` function:
```
    def eval_jvp(e: Expr, tan_env, std_env):
        if e is Var:
          return tan_env[e]
        elif e is Const:
          return 0.
        elif e is Add(e1, e2):
          return (eval_jvp(e1, tan_env, std_env) + 
                  eval_jvp(e2, tan_env, std_env))
        elif e is Mul(e1, e2):
          # Here we need the standard values for sub-expressions
          return (eval_jvp(e1, tan_env, std_env) * eval_std(e2, std_env) +
                  eval_std(e1, std_env) * eval_jvp(e2, tan_env, std_env))
```

Returning to our `func0` example, after the tracing of the body of `func0`
we obtain the symbolic expression `func0_e = Mul(Var(v), Var(v))`. 
To evaluate the tangent at point `arg = 3` we use a value `arg_tan = 1`,
i.e., `jvp(func0)(3, 1)`. This leads to the call
`eval_jvp(func0_e, {Var(v): 1}, {Var(v): 3})` and we should get the result `6`.
Importantly, the `+` and `*` operations in `eval_jvp` will operate 
on naked Python values and will be performed directly by the interpreter. 

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
Here, `func1` is being traced as part of `jvp(func1)`, and then we
encounter `jvp(func0)`. At that 
point a nested tracing process starts, a new tracing variable is created for
`x` and `func0` is traced. The rough sequence of steps that happen are: 
```
  arg1_typ: ExprType = type_of_val(3)       
  in1_t: Tracer = new_var_tracer(arg1_typ)  # = float
  res1_t : Tracer = func1(in1_t)
     # during func1(in1_t) the following happens:
     arg0_type: ExprType = type_of_val(in1_t)  # = float
     in0_t: Tracer = new_var_tracer(arg0_type)
     res0_t: Tracer = func0(in0_t)  # = 'in0_t * in0_t'
     tan_env0 = {in0_t.expr: 1}
     std_env0 = {in0_t.expr: in1_t}  # We are eval_jvp with a tracer object
     res0_tan = eval_jvp(res0.expr, tan_env0, std_env0)  # = '1 * in1_t + in1_t * 1'
     res1_t = res0_tan  
  tan_env1 = {in1_t.expr: 1}
  std_env1 = {in1_t.expr: 3)
  res1_tan = eval_jvp(res1_t.expr, tan_env1, std_env1)  # = 1 * 1 + 1 * 1 = 2
``` 

Note that in the first invocation of `eval_jvp(res0.expr)` some values are tracers, so 
the result is a tracer. In the second invocation `eval_jvp(res1_t.expr)` all values
are constants, so the `+` and `*` were executed directly by the Python interpreter
and the result is the Python constant `2`.

The key is that the non-standard evaluators implementing the transformations
must be traceable themselves: they must be functional and use their arguments
only with supported primitives. If you follow this rule, then you can write
new transformations that compose nicely with the other ones.  

### Many Python numerical contain statically-well-typed computations waiting to come out

The classic adage "well-typed programs don't go wrong" comes in very handy here.
We would like to have a static type system for our symbolic expressions such 
that (1) the output of tracing is a well-typed expression, (2) a well-typed 
expression can be evaluated without error, can be JIT-compiled and executed
without error, and can be transformed without errors into well-typed expressions.
Here, by "error" we mean some internal errors in the JAX transformation and
compilation machinery,
excluding data-dependent run-time errors such as division-by-zero. 

The reason why this type safety property is very useful is that we want to 
give all the JAX errors during tracing (during `traceable_user_func(in_t)`)
because that is when the Python interpreter executes user code and can 
localize the error with precise stack traces. If instead, we allow 
any of our standard or custom evaluators and JIT-compilers to fail, the 
stack trace will contain only JAX internal code. 

JAX's internal representation JAXPR is typed, and so is mini-JAX's internal
`Expr` language, with types representing the shape and base type of a tensor. 
In fact, the real JAX has a richer hierarchy of types that it called 
`abstract values`. 

One of the secrets of the success of JAX's tracing-based approach to transformation
is that many `numpy` numerical and ML programs
can be easily written with Python control flow based only on array shape values. Thus
one can trace the code with tracer values that capture the shapes to extract
a statically-typed symbolic expression specialization of the original program.

### The need for functions in the intermediate language

With the tracing described so far all Python control flow, including functions
and conditionals get inlined and the expression denoting a function to be
transformed does not need to have control-flow, functions, or scopes. This 
simple semantics would make it very easy to write transformations. 

In reality, JAXPRs (and mini-JAX `Expr`) have scopes and function calls, for
several reasons. 

First, the JIT transformation requires a portion of the computation to be 
compiled, perhaps for a specific device or hardware. For example, the `jit`
transformation in the example below requests the compilation of the body 
of `inner`.  
```
  def func1(x)
    z = x * 2
    def inner(y):
      return y + x * 4 + z
    return jit(inner)(x * 3)
```   
Furthermore, the compilation cannot all happen during tracing because if there
are enclosing transformations, e.g., `grad(func1)`, then the computation required
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
operators). The functions in JAXPR are:
* closed (all the data they need is passed explicitly through parameters),
* typed,
* anonymous, and called in a single place. These conditions can be relaxed, e.g., by
caching and reusing traced functions, but that would make some transformations
truly interprocedural. 

In addition to JIT, in JAX the conditionals also take advantage of functions. 
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
conditionals, e.g., by introducing a lazy operator.

The higher-order operators are the most complicated in the intermediate language. 
For example, the `grad` and `jvp` rules for conditionals are 30 and respectively
20 lines of code, compared to two lines for the arithmetic operations.    

### Various strategies for closure conversion during tracing

Once we introduce functions in the expression language, we have to 
decide how to handle capturing computations from the Python nested static
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
This is not a clear design requirement though, perhaps the `inner` function
is in a library and was written without thinking about `jit`, or other
transformations coming from the dynamic scope. 

JAX and mini-JAX do different things here. JAX does pretty aggressive constant
folding, meaning that it does computations as eagerly as possible. This 
can be viewed as an optimization, but users are also sometimes surprised
when they see multiple JIT-compilations for sub-expressions, when they 
expected only a single JIT-compilation. There is ongoing debate among
the JAX designers how to handle this (perhaps forcing some computations
to stay in the JIT scope in which they were written.)

In mini-JAX, for simplicity, all computations have a dependency on the 
dynamically-nearest enclosing transformation scope (all transformation,
not just JIT). Essentially the code above traces to 
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
expression for `x * 2` constructed outside `inner` is encounted in the 
computation of the return value of `inner`. At that point it is replaced
with a fresh variable, and lifted as a new parameter to `inner`. 
This code is in `mini_jax.py:Tracer.build`. 

The functional branches of conditionals are handled similarly. 

TODO: one can argue that for conditionals and for JIT is is important to 
keep computations dependent on the dynamic scope where they were created, 
but perhaps for other computations we can follow strict data dependence
and allow the optimizer to lift them as needed. Perhaps one can explore 
such a mixed strategy in mini-JAX (and JAX), but for simplicity this is not
yet done. 

### Careful sharing in the intermediate language is important

The choice of the intermediate language is important to ensure that the sharing
that arises naturally from the computation is preserved. For example, this code
`x = x + x; x = x + x; x = x + x` has three operations but with a symbolic
expression representation we may end up with an exponential printed form: 
`((x + x) + (x + x)) + ((x + x) + (x + x))`. Note that the actual symbolic 
data structure will likely contain the proper sharing, since only three `+` nodes
are constructed.   

Nevertheless, I wanted to experiments with simple symbolic expressions:
* it is actually simpler: just operators applied to sub-expression arguments.
* it is a functional representation of a whole computation DAG, the expression
 meaning is determined by the meaning of function's variables, we can cache
 results by expression object identity.
* dead-code naturally falls out

With such an expression representation it is crucial to be careful about 
sharing. We achieve this in mini-JAX by making **heavy use of memoization: a
distinct expression instance is processed at most once**. We added a visitor
helper function for expressions that takes care of memoization. The cost of 
memoization I believe is approximately the same as using JAXPRs intermediate variables, 
although one can argue that in JAXPRs the cost of exposing sharing is paid only 
when the JAXPR is constructed.   

JAXPRs instead are written with explicit naming of all intermediate computations.
This adds more indirection, makes it somewhat harder to see the expression
in the debugger, and requires a more complex data structure, with things like
topological sorting after transformations. 

In retrospect, I am not sure that the simplicity of representation as symbolic
expressions is worth the extra cost of being careful about preserving sharing.
   
### Real-JAX gives better errors can differentiate through control-flow

Perhaps the most important JAX features that is not reflected in mini-JAX
is JAX's ability to perform some transformations inline, during tracing. This 
has the major advantages that (1) one can apply differentiation
transformations on a program that contains data-dependent control flow, 
and (2) errors are reported during tracing when the 
user-program's stack trace is available. JAX does
this by carrying the actual concrete value as it traces the program, which 
enables it to resolve control flow. There is also a minor cost advantage of 
not having to materialize and traverse multiple times the JAXPR. 

JAX tries hard to do all transformations inline. The only cases when it cannot
do so are if a `jit`, or `pmap` transformation are present, or if we have 
higher-order control-flow (`cond` and `while`).

TODO: fusing transformations in mini-JAX should be doable, hopefully without 
too much complication.  

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

### Hacking mini-JAX was great fun, and humbling!

It is a rare project where 1000 lines of code can pack so much functionality,
so much reward, and so much trickiness. This is in part due to the fact that
tracing leverages much of the power of a Python interpreter, cleverly hijacked
through operator overloading. Some of the trickiness came from having to "undo"
some of the information lost through tracing; access to the source would have
made some of the tasks simpler, but dealing with Python ASTs is significantly
more involved than the intermediate form we use here.

The effort to try to find the simplest correct implementation was non-trivial,
and is but one fraction of the cleverness that is packed into the real JAX!

Code structure
--------------

Mini-JAX is structured as follows:

* General tracing machinery (`mini_jax.py`):

   * `Expr` is a class representing symbolic expressions (JAXPR), constructed with 
   `Operator`s applied to `Expr` arguments. All operators are considered to be
   strict in the arguments.
   * In addition to arguments, an operator application has parameters. 
   In essence, each `Operator` represents a parameterized family. For 
   example, the `Operator.LITERAL` has 0 arguments and 1 parameter with the 
   value of the literal. The `Operator.COND_GE` and `Operator.JIT_CALL` carry
   the branch and function as parameters.
   * `ExprType` represents types associated with each `Expr`. In 
    mini-JAX only the `float` type is implemented.  
   * `Function` is a class to represent *closed* functions over `Expr`. 
   * `Tracer` is the class of tracer values. A tracer value has not 
   only the symbolic `Expr` but also information about the scope nesting depth at 
   which it was constructed and the environment (tracer values from shallower
   scope depth that are used).
   
* `mini_jax_jit.py`: JIT transformation machinery. In the context of 
  mini-JAX, jitting means that we construct a Python executable string, 
  and then we `exec` it. 
    
* `mini_jax_jvp.py`: JVP transformation (forward differentiation).
  The public function exposed is `jvp(func)(a1, a2, a1_tan, a2_tan)`. 
  
* `mini_jax_grad.py`: GRAD transformation (backward differentiation). 
  The public function exposed is `grad(func)(a1, a2)`.

* `mini_jax_flops.py`: A toy FLOPS counting transformation.

* `tests`: unit tests, many showing the symbolic expression for many 
  transformations. 

TODO
-----

* Make a cleanup pass over the tests
* Improve the property-based testing using hypothesis.
* Explore fusing of forward transformations
* Explore constant folding for JVP and GRAD
* Explore writing JVP with lazy std_eval, like FLOPS
