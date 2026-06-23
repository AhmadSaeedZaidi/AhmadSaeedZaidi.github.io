---
layout: single
title: "Manifold-Constrained Hyper Connections: Theory"
description: "Extending residual connections with structured routing across multiple embedding lanes using doubly-stochastic matrices"
date: 2026-06-23
categories: cuda
---

## Residual Skip connections

The standard residual connection

$$
x_{l+1} = F(x_l) + x_l
$$

exists, as an innovation in backpropagation. As my previous blog post covered, backpropogation requires smooth gradients, close to 1 (or in the case of matrices, close to the identity matrix `I`). Now, observe the derivative of the above equation:

$$
\frac{\partial x_{l+1}}{\partial x_l} = F'(x_l) + \mathbb{I}
$$

you might notice, even if $F'(x_l) \approx 0$, the identity path $I$ preserves gradient flow. This lets us train networks dozens of layers deep while diminishing the vanishing gradient problem.

## The Problem with Standard Residuals

Modern deep networks have multiple channels (or lanes) of $X$, each with a full embedding. This allows the model to learn different features in different lanes, and fuse them at the end.

However, this creates challenges in designing skip connections. As discussed above, a skip connection acts as a path for gradient to flow. The model might learn to only use one channel, and ignore the rest.

mHC addresses this by (1) projecting multiple lanes into a smaller compute lane, (2) applying a heavy block (MoE / Attention) there, (3) redistributing the result back to the full lane dimension, and (4) imposing strict constraints on the skip connection to ensure stability across 100+ layers. This creates a structured "hyper-connection" that mixes and routes multiple embedding lanes.

## The Hyper-Connection Equation

The HC layer is written compactly as:

$$
X_{l+1} = B_l X_l + C_l \, F_l(A_l X_l)
$$

### Components

- $X_l, X_{l+1}$: input and output of layer $l$ respectively, where $X_l \in \mathbb{R}^{n_{hc} \times d}$.
- $n_{hc}$: the number of hyper-connection lanes.
- $d$: the embedding width.
- $A_l$: projection that squeezes the $n_{hc}$ lanes into a single lane. Shape $A_l \in \mathbb{R}^{1 \times n_{hc}}$, so $A_l X_l \in \mathbb{R}^{1 \times d}$.
- $F_l(\cdot)$: the expensive compute block (e.g. MoE or multi-head attention) that operates on the compressed $\mathbb{R}^{1 \times d}$ lane and returns $\mathbb{R}^{1 \times d}$.
- $C_l$: expansion (or coloring) matrix that maps the $(1, d)$ output of $F_l$ back to $(n_{hc}, d)$. Shape $C_l \in \mathbb{R}^{n_{hc} \times 1}$, so $C_l F_l(A_l X_l) \in \mathbb{R}^{n_{hc} \times d}$.
- $B_l$: mixes the original $n_{hc}$ input lanes. Shape $B_l \in \mathbb{R}^{n_{hc} \times n_{hc}}$; input and output retain shape $X_l, B_l X_l \in \mathbb{R}^{n_{hc} \times d}$ before adding to the expanded path, creating the bypass connection.

### Why

- We make a massive dimension $d = 7168$, each value in this vector is a feature of the token.
- We use multi-head attention: each head in the $7168$ embedding tries to learn a single group of features (grammar, semantics, context), and they fuse at the end using some kind of projection.
- We use multiple channels (e.g. 4), each with a full $7168$ embedding. Each channel is a global stream that learns a different feature. Before this, the single vector was overburdened, and prone to losing knowledge as it passed through attention layers.

### Notes

- The ordering of linear ops shown above is convenient for exposition; implementations often fuse projections or use learned channel-wise gates.
- Using a single compressed lane for the heavy block reduces compute and memory compared with applying the block to all lanes.
- Replace or tune the shapes ($n_{hc}$ and $d$) to match your model's lane count and embedding dimension.

### Quick Reference

- Equation: $X_{l+1} = B_l X_l + C_l F_l(A_l X_l)$
- Typical shapes: $X_l: (n_{hc}, d)$, $A_l: (1, n_{hc})$, $F_l: (1, d) \to (1, d)$, $C_l: (n_{hc}, 1)$, $B_l: (n_{hc}, n_{hc})$

## Constraints on the Parameter Matrices

**Constraints from the paper:**

$$
B_l \in M := \{ M \in \mathbb{R}^{n \times n} \mid \sum_{row} M_{row} = 1,\ \sum_{col} M_{col} = 1,\ M_{ij} \ge 0 \}
$$

Each $B_l$ belongs to a family of matrices $M$ — $n \times n$ matrices where each element $\ge 0$ and the sum of each row and each column is 1.

$$
A_l, C_l \in \mathbb{R}^{1 \times n}, \mathbb{R}^{n \times 1} \mid A_{ij} \ge 0,\ C_{ij} \ge 0
$$

**Hardware activations (implementation notes):**

$$
A_l = \sigma(\tilde A_l) + \epsilon, \qquad C_l = 2 \cdot \sigma(\tilde C_l)
$$

Thus:

$$
A_l \le 1, \qquad C_l \le 2
$$

where $\tilde A_l$ and $\tilde C_l$ are the raw, unnormalized parameters, $\sigma(\cdot)$ is element-wise sigmoid, and $\epsilon > 0$ is a small floor to avoid exact zeros in hardware.

### Why

**Non-Explosiveness** $\|B_l\|_2 \le 1$: spectral norm of the matrix. This ensures multiplication by $B$ can never be increasing. Safe for forward pass and gradient backprop, avoids exploding values after 100+ layers of transformation.

**Closure Under Multiplication** $\forall M_1, M_2 \in M \Rightarrow M_1 M_2 \in M$: this means that as multiplication by various matrices $M$ stack, the resulting matrix is from the set $M$, and has the same restrictions and properties. This ensures $B$ is well behaved over hundreds of layers, where these transformations by $B$ will stack many times.

**Non-Negative Bounding** $A_{ij}, C_{ij}, B_{ij} \ge 0$: since these matrices also stack additively, this constraint ensures they don't cause destructive interference over hundreds of layers. Without this, features could overwrite or negate each other.

**Bounded Magnitudes** $\sigma(\cdot) \le 1$: the sigmoid function enforces strict upper boundaries. $A_l$ is bounded near $1$, and $C_l$ is bounded at $2$. This gives the network the ability to safely scale the lanes without risking infinite amplification.

**Identity Initialization** $2 \cdot \sigma(\cdot)$: initially during training, weight matrices are initialized to small values close to zero. $\sigma(0) \approx 0.5$. Thus multiplication by $2$ initializes $C_l$ close to identity.

**Non-Zero Floor** $+\epsilon$: $A_l$ projects $n_{hc}$ lanes into 1 lane. If an element in $A_l$ is ever exactly $0$ — possible through floating-point rounding, especially at low-bit quantizations — the lane is disconnected from the next layers during forward pass, and previous layers during backprop. Thus we add a small offset to ensure this doesn't occur.

### The Sinkhorn-Knopp Algorithm

To enforce the doubly-stochastic constraint on $B_l$, we use the iterative Sinkhorn-Knopp algorithm. Very simple.

1. Start with $M^{(0)} = \text{Softmax}(\tilde B_l)$ — ensures non-negativity, one of our constraints.
2. Iterate for $t = 1, \dots, t_{\max}$:

$$
M_{ij}^{(t)} = \frac{M_{ij}^{(t-1)}}{\sum_k M_{ik}^{(t-1)}} \quad \text{(row normalization)}
$$

$$
M_{ij}^{(t)} = \frac{M_{ij}^{(t)}}{\sum_k M_{kj}^{(t)}} \quad \text{(column normalization)}
$$

3. Set $B_l = M^{(t_{\max})}$.

After row normalization, the matrix is normalized across rows but not columns. After column normalization, the matrix is normalized across columns but not rows. After each iteration, the matrix converges toward full constraint satisfaction.

Usually, the algorithm runs until $|\sum_{row} M_{row} - 1| \approx |\sum_{col} M_{col} - 1| \le \Delta$ where $\Delta$ is some small threshold (e.g. $10^{-6}$). But on GPU, loops with variable exit conditions are horrible for performance. So we set a fixed number of iterations — $t_{\max} = 20$ in practice — avoiding the variable-length loop that would degrade warp execution efficiency.

## Dynamic Parameterization

The parameters $A_l$, $B_l$, $C_l$ are not simple learned constants. They are **dynamically generated** from the input $X_l$ itself, using static and dynamic components.

The input is flattened and normalized:

$$
\hat X_l = \text{RMSNorm}(\text{vec}(X_l)) \in \mathbb{R}^{1 \times n_{hc} \cdot d}
$$

where $\text{vec}(X_l) \in \mathbb{R}^{1 \times n_{hc} \cdot d}$ is a flattening operation on $X_l \in \mathbb{R}^{n_{hc} \times d}$, and RMSNorm is:

$$
\text{RMSNorm}(x) = \frac{x}{\text{RMS}(x)} = \frac{x}{\sqrt{\frac{1}{n}\sum_{i=1}^n x_i^2 + \epsilon}}
$$

Then each raw parameter is a combination of static and dynamic components:

$$
\tilde A_l = \alpha_l^{\text{pre}} \cdot \hat X_l W_l^{\text{pre}} + S_l^{\text{pre}}
$$

$$
\tilde B_l = \alpha_l^{\text{comb}} \cdot \text{Mat}(\hat X_l W_l^{\text{comb}}) + S_l^{\text{comb}}
$$

$$
\tilde C_l = \alpha_l^{\text{post}} \cdot \hat X_l W_l^{\text{post}} + S_l^{\text{post}}
$$

where:

$$
W_l^{\text{pre}}, W_l^{\text{post}} \in \mathbb{R}^{n_{hc} \cdot d \times n_{hc}}, \qquad
W_l^{\text{comb}} \in \mathbb{R}^{n_{hc} \cdot d \times n_{hc}^2}
$$

these $W$ matrices are learnable weights for generating dynamic components.

### Components

- $\tilde A_l, \tilde B_l, \tilde C_l$: the raw, unnormalized parameters for the $A$, $B$, and $C$ matrices.
- $\alpha_l^{\text{pre}}, \alpha_l^{\text{comb}}, \alpha_l^{\text{post}}$: learnable scalar parameters that control the influence of the dynamic components for $A$, $B$, and $C$ respectively.
- $S_l^{\text{pre}}, S_l^{\text{comb}}, S_l^{\text{post}}$: learnable static parameters for $A$, $B$, and $C$ respectively.
- $W_l^{\text{pre}}, W_l^{\text{comb}}, W_l^{\text{post}}$: learnable weight matrices that project the normalized input $\hat X_l$ into the dynamic components for $A$, $B$, and $C$ respectively.
- $\text{Mat}(\cdot)$: a reshaping operation that converts $\hat X_l W_l^{\text{comb}} \in \mathbb{R}^{1 \times n_{hc}^2}$ into $\text{Mat}(\hat X_l W_l^{\text{comb}}) \in \mathbb{R}^{n_{hc} \times n_{hc}}$.

### Why

- **Dynamic Components $(\hat X_l W_l)$:** these allow the model to make token-specific routings for each hyper-connection. For example, for a token that is a verb, the model might learn to route more information through lane 1, and for a token that is a noun, it might route more through lane 2.
- **Static Components $S_l$:** these allow the model to learn general patterns of routing that are useful across all tokens.
- **Scaling Factors $\alpha_l$:** these allow the model to control the overall influence of the dynamic components.
- **Training Stability** $\alpha_l \approx 0$: the model can use static components as a baseline, and organically rely on dynamic components as training progresses.
- **RMSNorm:** normalizing the input to the dynamic parameter generation helps stabilize training and ensures the dynamic parameters are generated from a consistent scale of input features — geometrically only the direction is considered, the magnitude is normalized.
- **Flattening $\text{vec}(X_l)$:** this allows the dynamic parameter generation to consider interactions across all lanes and embedding dimensions, enabling more complex and informed routing decisions.
- **Reshape $\text{Mat}(\hat X_l W_l^{\text{comb}})$:** this is needed to match the shape of the $B$ matrix ($n_{hc} \times n_{hc}$), allowing it to properly mix the hyper-connection lanes based on the dynamic input.

## Hardware Considerations

The three separate matrix multiplications for generating $\tilde A_l$, $\tilde B_l$, and $\tilde C_l$ can be fused into a single large matrix multiplication:

$$
W_l^{\text{fused}} \in \mathbb{R}^{(n_{hc} \cdot d) \times (n_{hc} + n_{hc}^2 + n_{hc})}
$$

with $W_l^{\text{fused}} = [W_l^{\text{pre}}; W_l^{\text{comb}}; W_l^{\text{post}}]$ concatenating the individual weight matrices. Instead of three separate GEMM operations, a single fused GEMM produces all the raw parameters at once. This is more efficient on GPU hardware, as it reduces the number of separate operations and allows better utilization of the GPU's parallel processing capabilities.

For $n_{hc} = 4$, the output dimension is $4 + 16 + 4 = 24$ values per token. This small output dimension creates a challenge for the GEMM — it's very skinny. The GPU can have difficulty saturating its compute units when the output has only 24 columns per token, which is why techniques like Split-K (dividing the input dimension into chunks) are used to create more parallelism.

## Summary

mHC replaces the simple additive residual with a structured multi-lane routing mechanism. The key innovations are:

- **Compress-Compute-Expand** architecture: reduces compute by applying the expensive block on a single compressed lane.
- **Mathematical constraints** on $A_l, B_l, C_l$: ensure stability across hundreds of layers through non-explosiveness, closure under multiplication, and non-negative bounding.
- **Dynamic parameterization**: generates token-specific routing weights by combining input-dependent and static components.
- **Sinkhorn-Knopp projection**: enforces doubly-stochastic constraints on the routing matrix with a fixed-iteration GPU-friendly algorithm.


<small> **Small Authors Note:**

This is the first of two posts on mHC. The next post will cover the practical implementation details of the `mhc_pre_big_fuse` kernel (from the repository flashinfer, a frontier gpu inference library), which computes the dynamic parameters and applies the Sinkhorn-Knopp algorithm, on GPU. The kernel is a critical part of the mHC block, and optimizing it was a major focus of my work on this project. I will discuss the challenges of implementing the Sinkhorn-Knopp algorithm efficiently on GPU, and how I overcame some short comings of the existing kernel to achieve high performance. Also, my health has improved, I am now on a consistent routine of studying, practicing and documenting. Along with exercise. I hope the summer will bring enough positive change, that I can get employed soon. Stay tuned!
</small>