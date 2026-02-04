---
layout: single
permalink: /notation/
title: "Notation Reference"
mathjax: true
---

A living reference for the notation used across this blog. Aimed at students with strong programming skills who are still picking up mathematical notation.

# Summation (Sigma) Notation

$$
\Sigma_{i=1}^n i = 1 + 2 + 3 + \dots + n
$$

$\Sigma_{i=1}^n$ means "sum up the expression to the right for $i$ from 1 to $n$". I use it more loosely too, like $\Sigma_{\text{row}=1}^n M_{\text{row}}$ to mean "sum across rows of matrix $M$". The dimension being summed over is named in the subscript.

# Matrix Dimensions and Shapes

- $A \in \mathbb{R}^{m \times n}$: $A$ is a matrix with $m$ rows and $n$ columns. The first number is rows, the second is columns.
- Matrix multiplication $A X$ is valid when $A \in \mathbb{R}^{m \times n}$ and $X \in \mathbb{R}^{n \times p}$, producing a result in $\mathbb{R}^{m \times p}$. The inner dimensions ($n$) must match.
- "Shape" refers to the dimensions of a tensor. $X$ has shape $(n_{\text{hc}}, d)$ means $X$ is a matrix with $n_{\text{hc}}$ rows and $d$ columns.

**Intuition:**
"Matrix multiplication is dot products between every row of $A$ and every column of $X$. The number of columns in $A$ must equal the number of rows in $X$ because each dot product pairs entries one-to-one."

# Element-wise Operations

- $A \odot B$: element-wise multiplication (Hadamard product). Each entry $(A \odot B)_{ij} = A_{ij} \cdot B_{ij}$. Both matrices must have the same shape.
- $A \oslash B$: element-wise division. Each entry $(A \oslash B)_{ij} = A_{ij} / B_{ij}$.
- $A_{\text{element}} \ge 0$: every element of $A$ is non-negative.
- $\sigma(\cdot)$: typically an element-wise activation function applied to each entry independently.

### Broadcasting

If $A \in \mathbb{R}^{m \times n}$ and $b \in \mathbb{R}^m$, then $A \oslash b$ broadcasts $b$ across columns:

$$
A \oslash b = \begin{bmatrix}
a_{11}/b_1 & a_{12}/b_1 & \dots \\
a_{21}/b_2 & a_{22}/b_2 & \dots \\
\vdots & \vdots & \ddots
\end{bmatrix}
$$

This means "repeat $b$ along the missing dimension to match $A$, then divide element-wise". Broadcasting is a convention borrowed from NumPy/PyTorch to keep notation compact.

# Set Notation

$M := \{M \in \mathbb{R}^{n \times n} \mid M_{ij} \ge 0\}$ defines the set of all $n \times n$ matrices with non-negative entries. The colon-equal $:=$ means "is defined as".

# Piecewise (Case) Notation

A piecewise function assigns different expressions depending on a condition:

$$
f(x) = \begin{cases}
\text{expression}_1 & \text{if condition}_1 \\
\text{expression}_2 & \text{if condition}_2
\end{cases}
$$

The causal attention mask used in transformers:

$$
M_{ij} = \begin{cases}
0 & \; j \le i \\
-\infty & \; j > i
\end{cases}
$$

meaning "put 0 if $j \le i$ (allow attending), $-\infty$ if $j > i$ (block attending)". After softmax, $-\infty$ becomes exactly 0.

The KeepTopK function in Mixture-of-Experts routers:

$$
\text{KeepTopK}(v, k)_i = \begin{cases}
v_i & \text{if } v_i \text{ is in the top } k \text{ elements of } v \\
-\infty & \text{otherwise}
\end{cases}
$$

meaning "keep the value if it is among the $k$ largest, otherwise discard it (set to $-\infty$)". After softmax, only the kept experts get non-zero probability.

# Activation Functions

An activation function is a non-linear transformation applied element-wise. The non-linearity is what lets neural networks learn curved decision boundaries instead of just straight lines.

- **ReLU:** $\text{ReLU}(x) = \max(0, x)$. Returns $x$ if positive, 0 otherwise. Cheap, sparse, but kills gradient for dead neurons. Used in the original Transformer FFN.

- **GeLU:** $\text{GeLU}(x) = x \cdot \Phi(x)$ where $\Phi(x)$ is the CDF of the standard normal. Smooth everywhere, non-zero gradient for negative inputs. Used in BERT and GPT-2.

- **SiLU (Swish):** $\text{SiLU}(x) = x \cdot \sigma(x)$. Smooth, non-monotonic. Used inside the SwiGLU FFN variant that powers Llama, Mistral, and DeepSeek.

- **Sigmoid:** $\sigma(x) = 1/(1 + e^{-x}) \in (0, 1)$. Used in MoE gating networks to produce routing probabilities.

- **Tanh:** $(e^x - e^{-x})/(e^x + e^{-x}) \in [-1, 1]$. Used in the GeLU approximation.

- **Softplus:** $\text{Softplus}(x) = \log(1 + e^x)$. Smooth, positive. Used in the MoE router to ensure noise magnitudes stay positive.

# Softmax

$\text{Softmax}(x)$ converts a vector of real numbers into a probability distribution:

$$
\text{Softmax}(x)_i = \frac{e^{x_i}}{\sum_{j=1}^n e^{x_j}}
$$

The exponential makes everything positive, the denominator makes it sum to 1. $\text{Softmax}_{\text{row}}(M)$ applies softmax to each row independently, giving row-stochastic attention weights from raw scores.

# Gradients

For a scalar function $f: \mathbb{R}^n \to \mathbb{R}$, the gradient $\nabla f(x)$ is a vector of partial derivatives:

$$
\nabla f(x) = \begin{bmatrix}
\frac{\partial f}{\partial x_1} &
\frac{\partial f}{\partial x_2} &
\dots &
\frac{\partial f}{\partial x_n}
\end{bmatrix}^\mathsf{T} \in \mathbb{R}^n
$$

Each entry $\partial f / \partial x_i$ measures how much $f$ changes when $x_i$ changes by a tiny amount. The gradient points in the direction of steepest ascent: moving $x$ in the direction of $\nabla f$ increases $f$ the fastest, and moving opposite to it (gradient descent) decreases $f$ the fastest.

**Intuition:**
"The gradient is a 'local compass'. At any point $x$, it tells you which direction to step in to make $f$ go up the most. The size of each component tells you how sensitive $f$ is to that input."

# Jacobians

For a vector-valued function $f: \mathbb{R}^n \to \mathbb{R}^m$, the Jacobian $J$ collects all partial derivatives into a matrix:

$$
J = \begin{bmatrix}
\frac{\partial f_1}{\partial x_1} & \frac{\partial f_1}{\partial x_2} & \dots & \frac{\partial f_1}{\partial x_n} \\
\frac{\partial f_2}{\partial x_1} & \frac{\partial f_2}{\partial x_2} & \dots & \frac{\partial f_2}{\partial x_n} \\
\vdots & \vdots & \ddots & \vdots \\
\frac{\partial f_m}{\partial x_1} & \frac{\partial f_m}{\partial x_2} & \dots & \frac{\partial f_m}{\partial x_n}
\end{bmatrix} \in \mathbb{R}^{m \times n}
$$

Entry $J_{ij} = \partial f_i / \partial x_j$: how output $i$ responds to input $j$. When $m = 1$, the Jacobian is a row vector $\nabla f(x)^\mathsf{T}$ (the transpose of the gradient).

**Why it matters:** The Jacobian is the backbone of backpropagation. Each layer's Jacobian tells you how a small change in its input affects its output. Chaining these via the chain rule gives you the gradient of the loss with respect to every parameter.

**Intuition:**
"If the gradient is a compass for a scalar output, the Jacobian is a bundle of $m$ compasses, one for each output dimension. Each row $J_{i:}$ is the gradient of $f_i$, telling you how that specific output responds to each input."

# Directional Derivatives

A directional derivative $D_v f(x)$ measures the rate of change of $f$ along a direction $v$ (a unit vector).

For a scalar function $f: \mathbb{R}^n \to \mathbb{R}$:

$$
D_v f(x) = \nabla f(x) \cdot v = \sum_{i=1}^n \frac{\partial f}{\partial x_i} v_i
$$

This is the slope of $f$ at $x$ if you walk in direction $v$. The gradient direction $\nabla f / \|\nabla f\|$ gives the steepest ascent; the negative gradient gives steepest descent.

For a vector-valued function $f: \mathbb{R}^n \to \mathbb{R}^m$:

$$
D_v f(x) = J(x) \, v \in \mathbb{R}^m
$$

where $J(x)$ is the Jacobian. This is a vector: how each output changes along direction $v$.

# Statistical Notation

- $\mathbb{E}[X]$: expected value (mean) of random variable $X$. For a sample $\{x_1, \dots, x_n\}$, $\mathbb{E}[X] = \frac{1}{n}\sum_{i=1}^n x_i$.
- $\text{Std}[X]$: standard deviation. Measures how spread out the values are.
- $\text{CV}(X) = \text{Std}[X] / \mathbb{E}[X]$: coefficient of variation. A unitless measure of relative dispersion, used in the MoE importance loss to normalize load across experts.
- $\mathcal{N}(0, 1)$: standard normal distribution (mean 0, variance 1). Used in the MoE gating network for adding random noise to explore routing assignments.

# Concatenation of Matrices

$[A; B]$ means concatenating $A$ and $B$ along a chosen dimension. If $a = [1, 2]$ and $b = [3, 4]$, then $[a; b] = [1, 2, 3, 4]$. For matrices, the axis depends on the shapes. Fused weight matrices in efficient implementations are often built by concatenating separate weight matrices along the appropriate dimension so a single matrix multiply replaces several.

---

# GeLU Derivation

<details>
<summary>Click to expand</summary>

$\Phi(x)$ is the cumulative distribution function (CDF) of the standard normal distribution $f(t) = \frac{1}{\sqrt{2\pi}} e^{-t^2/2}$:

$$
\Phi(x) = \int_{-\infty}^x f(t) \, dt = \int_{-\infty}^x \frac{1}{\sqrt{2\pi}} e^{-t^2/2} dt
$$

This has no closed form, so we approximate it. Use the error function:

$$
\text{erf}(x) = \int_0^x \frac{2}{\sqrt{\pi}} e^{-t^2} dt
$$

Split the integral for $\Phi(x)$:

$$
\Phi(x) = \int_{-\infty}^0 f(t) dt + \int_0^x f(t) dt
$$

The first part is $0.5$ (half the symmetric bell curve). For the second, substitute $u = t / \sqrt{2}$:

$$
\int_0^x f(t) dt = \int_0^{x/\sqrt{2}} \frac{1}{\sqrt{\pi}} e^{-u^2} du = \frac{1}{2} \text{erf}\!\left(\frac{x}{\sqrt{2}}\right)
$$

So:

$$
\Phi(x) = \frac{1}{2}\left(1 + \text{erf}\!\left(\frac{x}{\sqrt{2}}\right)\right)
$$

To make this practical, approximate $\text{erf}$ with $\tanh$, which is fast on hardware. The Taylor series around 0 are:

$$
\text{erf}\!\left(\frac{x}{\sqrt{2}}\right) = \sqrt{\frac{2}{\pi}}\left(x - \frac{x^3}{6} + \frac{x^5}{40} - \cdots\right)
$$

$$
\tanh(y) = y - \frac{y^3}{3} + \frac{2y^5}{15} - \cdots
$$

Let $y = \sqrt{2/\pi}(x + a x^3)$ and match coefficients. Matching the $x$ term is automatic. Matching $x^3$:

$$
\sqrt{\frac{2}{\pi}} a - \frac{1}{3}\left(\frac{2}{\pi}\right)^{3/2} = -\frac{1}{6}\sqrt{\frac{2}{\pi}}
$$

Solving gives $a = (4 - \pi) / (6\pi) \approx 0.04554$. The paper uses $0.044715$, which is the minimax-optimal value (found by minimizing maximum error over $[-2, 2]$ instead of matching the Taylor $x^3$ term exactly).

The final approximation:

$$
\text{GeLU}(x) \approx x \cdot \frac{1}{2}\left(1 + \tanh\left(\sqrt{\frac{2}{\pi}}\left(x + 0.044715 x^3\right)\right)\right)
$$

with maximum error $\approx 0.001$ over $x \in [-4, 4]$, more than accurate enough for training.

</details>
