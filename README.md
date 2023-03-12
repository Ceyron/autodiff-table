# Explicit Autodiff Table  📚

An overview of major automatic differentiation primitive rules for explicit
scalar and tensor operations, e.g., addition and multiplication.

👉 Here is the web version: [https://ceyron.github.io/autodiff-table/](https://ceyron.github.io/autodiff-table/).

## 💡 Background

Given a unary function with one input and one output

$$ f(x) =: z $$

these autodiff rules define ways of obtaining the Jacobian-vector product (Jvp)

$$ \dot{x} \mapsto \frac{\partial f}{\partial x} \cdot \dot{x} = \dot{z} $$

and the vector-Jacobian product (vJp)

$$ \bar{z} \mapsto \bar{z}^T \frac{\partial f}{\partial x} = \bar{x}^T $$

**without ever explicitly constructing the Jacobian matrix $\frac{\partial f}{\partial x}$**.

🧠 Modern automatic differentiation engines like [JAX](https://jax.readthedocs.io/en/latest/), [TensorFlow](https://www.tensorflow.org/guide/autodiff), [PyTorch](https://pytorch.org/tutorials/beginner/basics/autogradqs_tutorial.html), [Autograd](https://github.com/HIPS/autograd), [Zygote](https://fluxml.ai/Zygote.jl/stable/) and many more then use these rules in different approaches (static graphs, dynamic graphs, source-code transformation, piggybacking etc.) to compute derivatives of arbitrary programs (often associated to numerical computing like machine learning --- especially deep learning --- and scientific computing).

If one has a general function with many inputs and outputs, the Jvp for one
specific output tangent is sum of propagation from all the inputs. Vice versa,
the vJp for one specific input cotangent is the sum of the backpropagation from
all output cotangents.

##  📖 Resources

Check out my [video playlist](https://www.youtube.com/watch?v=PwSaD50jTv8&list=PLISXH-iEM4Jn3SEi07q8MJmDD6BaMWlJE) I created with in-depth derivations for most of the rules in this table. You find the handwritten notes over on the [GitHub Repo of the channel](https://github.com/Ceyron/machine-learning-and-simulation/tree/main/english/adjoints_sensitivities_automatic_differentiation/rules).

* General Books on Automatic Differentiation:
    * [Evaluating Derivatives](https://epubs.siam.org/doi/book/10.1137/1.9780898717761): Principles and Techniques of Algorithmic Differentiation, by Andreas Griewank and Andrea Walther.
    * [The Art of Differentiating Computer Programs](https://epubs.siam.org/doi/10.1137/1.9781611972078) by Uwe Naumann.
* Survey Paper:
    * [Automatic Differentiation in Machine Learning: a Survey](https://arxiv.org/abs/1502.05767) by Atilim Gunes Baydin et al.
* Concrete Derivations:
    * [An extended collection of matrix derivative results for forward and
      reverse mode algorithmic
      differentiation](https://people.maths.ox.ac.uk/gilesm/files/NA-08-01.pdf) by Mike Giles

