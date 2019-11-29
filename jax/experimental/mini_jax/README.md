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
  * All transformations are composable.  
  * Reverse differentiation (GRAD) is careful to ensure that the cost of the
    reverse differentiation is within a constant-factor of the cost of the forward
    computation.
  * Forward differentiation (JVP).
  * A new toy transformation to estimate the FLOPS cost of the code, without
    actually running the computation, except as much as needed for when there
    is data-dependent FLOPS.
  * Conditionals as higher-order primitives.

* Omitted JAX features and optimizations:

  * All transformations are performed after tracing the function to the intermediate
    language. This is called an "initial embeddeding" in JAX. 
    In JAX, some transformation are performed online as the JAXPR 
    is generated (and often the JAXPR is not even materialized). This has some
    important benefits, which we discuss further down.  
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
  * perhaps others?

 
## Highlights and lessons learned

The goal was to implement mini-JAX as simply as possible, but not simpler! It is
easy to end up with an implementation that is too simple. Here are some of the 
things that are non-obvious.

### Introduction to JAX tracing

Assume that we want to code up a transformation such that given a traceable 
Python function `traceable_func`, then `jvp(traceable_func, arg, arg_tan)` 
evaluates the perturbation of the result of `traceable_func` evaluated at 
input `arg` given a perturbation of the input `arg_tan`. For example:
```
   def func0(x):
     return x * x
```
Then `jvp(func0)(a, a_tan)` evaluates to `2 * a * a_tan`.

JAX performs function transformations on an internal representation of
the function, called JAXPR (JAX expressions). (Some transformations, e.g., JVP
are performed online, during tracing, and no JAXPR is materialized.)
In this write-up (and in mini-JAX) we will not use specific details of
how JAXPR are implemented and are going to simply use symbolic expressions
`Expr` with constructs such as `Var(v)` (for variables, where `v` is some
internal unique identifier), `Literal(3.0)`, `Add(e1, e2)`, etc. For specific
details in JAXPRs, see "Understanding JAXPR". 

To obtain the symbolic expression, JAX does not try to parse the source code 
of the function.
Instead, it executes the function using special tracer objects as arguments. 
A tracer object has a type, and a symbolic expression that denotes the 
current value of the tracer. The tracer object also overloads
Python operators such that it gets notified when the Python interpreter
tries to use the tracer object in an operation. The tracer implementation of 
operators builds symbolic expressions corresponding to the computation and
return tracers. 

A pseudo-code for how the tracing is invoked is roughly:
```
  arg_typ: ExprType = type_of_val(arg)    # Abstract 'arg' to type, e.g., float, or float[3,4]   
  in_t: Tracer = new_var_tracer(arg_typ)  # Create a tracer for a Var expression 
``` 
Now we simply let the Python interpreter execute the user function: 
```
  primal_res_t: Tracer = traceable_user_func(in_t)
```
Now `primal_res_t.expr` is a symbolic representation of the computation that was performed
on tracers by the user function. For a user function to be adequate for such tracing it must: 

* use the arguments only in computations that use a designated set of operations (that
  have been overloaded.) In mini-JAX, this are `+`, `-`, `*`, `**` with integer 
  exponent. JAX has a much richer set of primitive operations, and the `numpy`
  API has been implemented in terms of these primitive operations.   
  An example of inappropriate operations are library functions such as
  `copy` or `pickle`, or even Python conditional constructs. 
* the result of the function should be returned through the function result, 
  not through global state. The function can store values in mutable state 
  internally, but those values should not be used after the function returns, 
  and the result should be returned directly. 

There are a few interesting details in the way JAX tracing works. Normally, JAX
traces through Python control flow and function calls. For example, when tracing 
the function `func1` below::
```
  def func1(x):
    z = x * 2
    def inner(y):
      return y + x * 4 + z
    return inner(x * 3)
```    
the resulting JAXPR is essentially `x * 3 + x * 4 + x * 2` (tracing through
variable assignments, function calls, and lexical closures). In particular, 
in this form of tracing, all the function calls are inlined. Loops and
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

One of the simplest things that one can do with a symbolic expression is to
evaluate it using certain values for the variables. We call this standard
evaluation to separate it from other evaluations we'll have. 
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
to write a `eval_jvp` function that takes an expression and computes its 
tangent given tangent values for the variables. First, we prepare a variable 
environment with the input tangents:
```
  tan_env = {in_t.expr: arg_tan}
```

Often during non-standard evaluation we also need the standard value of
a sub-expression. For `eval_jvp` this will be needed when evaluating the tangent
of a multiplication; we need to use the tangents of the operands and also 
their standard values. For this purpose we also set up a standard environment:
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
Importantly, the `+` and `*` operations in from `eval_jvp` will operate 
on naked Python values and will be performed directly by the interpreter. 

What happens if we want to nest transformations: during the tracing of
a function being transformed, we encounter another transformation. 
This kind of behavior arises naturally if one wants to compute the 
second-derivative of a function:
```
  def func1(y):
    def func0(x):
      return x * x
    return jvp(func0, y, 1)
  second_derivative_at_3 = jvp(func1, 3, 1)
```
Here, `func1` is being traced, and then we encounter `jvp(func0)`. At that 
point a nested tracing process starts, a new tracing variable is created for
`x` and `func0` is traced. The rough sequence of steps that happen are: 
```
  arg1_typ: ExprType = type_of_val(3)       
  in1_t: Tracer = new_var_tracer(arg1_typ)  # = float
  res1 : Tracer = func1(in1_t)
     arg0_type: ExprType = type_of_val(in1_t)  # = float
     in0_t: Tracer = new_var_tracer(arg0_type)
     res0_t: Tracer = func0(in0_t)  # = 'in0_t * in0_t'
     tan_env0 = {in0_t.expr: 1}
     std_env0 = {in0_t.expr: in1_t}  # We are eval_jvp with a tracer object
     res0_tan = eval_jvp(res0.expr, tan_env0, std_env0)  # = '1 * in1_t + in1_t * 1'
     res1 = res0_tan  
  tan_env1 = {in1_t.expr: 1}
  std_env1 = {in1_t.expr: 3)
  res1_tan = eval_jvp(res1.expr, tan_env1, std_env1)  # = 1 * 1 + 1 * 1 = 2
``` 

Note that in the first invocation of `eval_jvp(res0.expr)` some values were treacers, so 
the result is a tracer. In the second invocation `eval_jvp(res1.expr)` all values
were constants, so the `+` and `*` were executed directly by the Python interpreter
and the result is the Python constant `2`.

The key was that the non-standard evaluators implementing the transformations
must be traceable themselves: they must be functional and use their arguments
only with supported primitives. If you follow this rule, then you can write
new transformations that compose nicely with the other ones.  

### Type-safe intermediate language is very important

TODO


### Careful sharing in the intermediate language is important

The choice of the intermediate language is important to ensure that the sharing
that arises naturally from the computation is preserved. For example, this code
`x = x + x; x = x + x; x = x + x` has three operations but with a symbolic
expression representation we may end up with an exponential printed form: 
`((x + x) + (x + x)) + ((x + x) + (x + x))`. Note that the actual symbolic 
data structure will likely contain the proper sharing, since only 3 `+` nodes
are constructed.   

Nevertheless, I wanted to keep the simple symbolic expressions:
* it is actually simpler: operators applied to sub-expression arguments.
* it is a functional representation of a whole computation DAG, the expression
 meaning is determined by the meaning of function's variables, we can cache
 results by expression object identity.

With such an expression representation it is crucial to be careful about 
sharing. We achieve this in mini-JAX by making *heavy use of memoization: a
distinct expression instance is processed at most once*. We added a visitor
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
   
### Various strategies for closure conversion


*This section needs complete rewrite*

A special transformation in JAX is `jit` which prepares the code of the traced
function for compilation for a device. The actual compilation happens only
when the function is invoked. If there is another transformation, e.g., `grad`,
applied before invocation, the body of the jitted function is transformed 
appropriately. This means that `grad(jit(func))` is approximately the same
as `jit(grad(func))`. (This is not quite true, `grad` after `jit` typically 
can do fewer optimizations because it is operating with code that contains 
function calls, while `grad` before `jit` can do better since it operates
on plain expressions.) 
 
To achieve this lazy-JIT effect, JAX has a special JAXPR primitive `xla_call` 
to carry the body of the function. These are the only functions in JAXPR, regular
Python functions have been inlined. When we have nested `jit`, we obtain
nested `xla_call` in JAXPR.  

Taking the example above, and applying the `jit` transformation to the `inner` 
function:
```
  def func1(x)
    z = x * 2
    def inner(y):
      return y + x * 4 + z
    return jit(inner)(x * 3)
```    
results in the following JAXPR:
`xla_call (lambda y, x, z: y + x * 4 + z) (x * 3) x (x * 2)`.
Note that the body of `inner` is kept as an anonymous function. Also, 
the body is "closed", and is passed explicitly the *environment values* for `x`
and `z`. 

Also note that the computations `x * 2` and `x * 3` are kept outside the jitted
function, as they were written. This is important, because we want the user 
to have control over how and on what device the computations are performed. 
Achieving this effect is tricky. In a pure data flow tracing implementation,
meaning that a tracing value knows only its symbolic expression form, there is 
no way to distinguish between the `x * 4` that was traced inside `inner` and the
`x * 2` from outside. To achieve this effect, we must keep as global state the 
*scope nesting depth* where a tracing value is being build. During tracing
when we encounter arguments from a shallower scope depth, we replace them with 
fresh variables that are collected as part of a tracing environment accompanying
the tracing values. See below the code in `Tracer`.

### Real-JAX can differentiate through control-flow

TODO

Fusing transformations?


### The need for functions in the intermediate language

JIT has to defer the compilation and execution of a block of code. 
This forces us to introduce functions and function calls in the internal
expression language. At the moment we have only anonymous, non-recursive functions
that are called only once. Nevertheless, this is a source of complexity in 
the internal expression language and the code.  

Another reason to have functions is the presence of a higher-order 
conditional construct. 

TODO: explain conditionals

### Efficient reverse differentiation is tricky

Implementing reverse differentiation correctly is not hard, if all you care 
is functional correctness. The "complexity correctness" is quite a bit harder.
At some point, I had an implementation that was traversing the expressions
and constructing a gradient (set of partial derivatives) data structure with
addition, scaling, and variable uses. Being careful about sharing in expressions
meant that at most one gradient node was being built for one distinct 
expression object. In a second pass, the gradient data structure was traversed 
to accumulate the partial derivatives. Here too was important to memoize 
the traversal of the gradients. Even after these two careful traversal, I 
could still end with exponential blowup. An example code (due to Dougal) was:
```
   z = a * (z + z)
   z = a * (z + z)
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
   `Operator`s applied to `Expr` arguments. 
   * In addition to arguments, an operator application has parameters. 
   In essence, each `Operator` represents a parameterized family. For 
   example, the `Operator.LITERAL` has 0 arguments and 1 parameter with the 
   value of the literal. 
   * `ExprType` represents types associated with each `Expr`. In 
    mini-JAX only the `float` type is implemented. It is important that `Expr` be
    typed, and that well-typed `Expr` cannot result in late compilation and
    transformation errors. We want to catch all ill-formed `Expr` early, during
    tracing, while the errors have access to the proper Python stack trace. 
   * `Function` is a class to represent closed functions over `Expr`. 
   * `Tracer` is the class of tracing values. A tracing value has not 
   only the `Expr` but also information about the scope nesting depth at 
   which it was constructed and the environment (tracing values from shallower
   scope depth that are used).
   
* `mini_jax_jit.py`: JIT transformation machinery. In the context of 
  mini-JAX, jitting means that we construct a Python executable string, 
  and then we `exec` it. 
    
* `mini_jax_jvp.py`: JVP transformation (forward differentiation).
  The public function exposed is `jvp(func)(a1, a2, a1_tan, a2_tan)`. 
  
* `mini_jax_grad.py`: GRAD transformation (backward differentiation). 
  The public function exposed is `grad(func)(a1, a2)`.

* `mini_jax_flops.py`: FLOPS counting transformation.

TODO
-----

* Explore how can we do better for grad(jit)
* Make a pass over the tests
* Try to use hypothesis for property-based testing. See bug in 01b8036 (wrong
  order of arguments in conditional functionals that return multiple values.)
