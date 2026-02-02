---
layout: single
title: "Matrix Decompositions: LU and QR"
description: "Factorizing matrices into simpler pieces (triangular and orthogonal) and why QR is the stable choice"
date: 2026-02-02
categories: [math]
author_profile: false
show_excerpts: false
sidebar:
  nav: "math"
---

Factorizing a matrix $A$ into simpler pieces is the backbone of numerical linear algebra. Instead of solving $Ax = b$ directly (costs $O(n^3)$ and you start over if $b$ changes), you factor $A$ once into cheap-to-invert pieces and solve in $O(n^2)$ per right-hand side. Two factorizations dominate: **LU** (triangular × triangular) for square systems, and **QR** (orthogonal × triangular) for least squares and ill-conditioned problems. This post walks through both, from Gaussian elimination up to blocked Householder reflections.

# LU Decomposition

## The Algorithm

$$
A = LU \qquad L \in \mathbb{R}^{n \times n},\; U \in \mathbb{R}^{n \times n}
$$

$L$ is unit lower triangular ($L_{ii} = 1$, zeros above the diagonal), and $U$ is upper triangular (zeros below the diagonal).

Intuition: "We write $A$ as a product of a *lower* and an *upper* triangular matrix. Once you have this, solving $Ax = b$ is two cheap triangular solves: forward then backward."

## Derivation (Gaussian Elimination)

Elimination turns $A$ into $U$ by subtracting multiples of each row from the rows below. Each subtraction is an elementary matrix $E_k$. For a $3 \times 3$:

$$
E_3 E_2 E_1 A = U
$$

where each $E_k$ is unit lower triangular. Move them to the other side:

$$
A = (E_3 E_2 E_1)^{-1} U = LU
$$

The product $L = (E_3 E_2 E_1)^{-1}$ is unit lower triangular, and it turns out its entries are exactly the multipliers we used during elimination, placed below the diagonal.

Concretely, for $A \in \mathbb{R}^{3 \times 3}$. During step 1 we compute multipliers $m_{21} = a_{21} / a_{11}$, $m_{31} = a_{31} / a_{11}$ and subtract $m_{21} \cdot \text{row}_1$ from row 2 and $m_{31} \cdot \text{row}_1$ from row 3. During step 2 we compute $m_{32} = a_{32} / a_{22}$ (using the updated $a_{22}$) and subtract $m_{32} \cdot \text{row}_2$ from row 3. The resulting $L$ and $U$ are:

$$
L = \begin{bmatrix}
1 & 0 & 0 \\
m_{21} & 1 & 0 \\
m_{31} & m_{32} & 1
\end{bmatrix}, \qquad
U = \begin{bmatrix}
a_{11} & a_{12} & a_{13} \\
0 & a_{22} & a_{23} \\
0 & 0 & a_{33}
\end{bmatrix}
$$

The multipliers go directly into $L$ at the positions where they zeroed entries in $A$. This is why $LU$ is essentially just Gaussian elimination with a different accounting.

## Components

- $A \in \mathbb{R}^{n \times n}$: the original square matrix.
- $L \in \mathbb{R}^{n \times n}$: unit lower triangular. $L_{ii} = 1$, $L_{ij} = 0$ for $j > i$. The sub-diagonal entries are the elimination multipliers.
- $U \in \mathbb{R}^{n \times n}$: upper triangular. $U_{ij} = 0$ for $i > j$. This is the result of elimination.
- $P \in \mathbb{R}^{n \times n}$: permutation matrix (for partial pivoting). $P$ reorders rows so that the pivot is never zero.

## Solving $Ax = b$ with $LU$

Once $A = LU$ (or $PA = LU$):

1. **Forward substitution**: solve $Ly = Pb$ for $y$. Since $L$ is lower triangular, you start from the first row and work down ($O(n^2)$).
2. **Back substitution**: solve $Ux = y$ for $x$. Since $U$ is upper triangular, you start from the last row and work up ($O(n^2)$).

The factorization costs $\frac{2}{3}n^3$ flops. Each solve costs $O(n^2)$. If you have many right-hand sides $b$, you pay the cubic cost once and the quadratic cost per solve.

## Pivoting ($PA = LU$)

If $a_{11} = 0$ at any step, elimination divides by zero. More subtly, even if $a_{kk}$ is nonzero but tiny, dividing by it produces huge multipliers, amplifying rounding errors.

**Partial pivoting** fixes this: before eliminating column $k$, swap row $k$ with the row below it that has the largest entry in column $k$. Track the swaps in a permutation matrix $P$:

$$
PA = LU
$$

## Why

- **Direct, not iterative:** LU gives the exact solution (up to floating-point precision) in a fixed number of operations. No convergence criteria, no iteration count to tune.
- **Cheap per RHS:** after the $O(n^3)$ factorization, each additional solve is $O(n^2)$. This is the main advantage of factorization over computing $A^{-1}$ or using iterative methods for many right-hand sides.
- **Well-understood stability:** with partial pivoting, the growth factor is bounded in practice. Pathological examples exist (the Wilkinson matrix), but they rarely appear in real applications.

## Stability

Without pivoting, error can grow exponentially. The famous example:

$$
A = \begin{bmatrix}
1 & 0 & 0 & \cdots & 1 \\
-1 & 1 & 0 & \cdots & 1 \\
-1 & -1 & 1 & \cdots & 1 \\
\vdots & \vdots & \vdots & \ddots & \vdots \\
-1 & -1 & -1 & \cdots & 1
\end{bmatrix}
$$

The multipliers grow as $2^{k-1}$, and $U$ develops entries as large as $2^{n-1}$. For $n = 60$, this overflows double precision.

With partial pivoting ($PA = LU$), the multipliers are bounded by $1$, and the growth factor is typically $O(n)$. Backward error is proportional to $\text{cond}(A) \cdot \varepsilon$, where $\varepsilon$ is machine epsilon. For well-conditioned $A$, LU with pivoting is perfectly safe.

# QR Decomposition

## The Algorithm

$$
A = QR \qquad A \in \mathbb{R}^{m \times n},\; Q \in \mathbb{R}^{m \times m},\; R \in \mathbb{R}^{m \times n}
$$

$Q$ is orthogonal ($Q^\mathsf{T}Q = I$), and $R$ is upper triangular.

Thin (economy) QR: when $m > n$, we only need the first $n$ columns of $Q$:

$$
A = QR \qquad Q \in \mathbb{R}^{m \times n},\; R \in \mathbb{R}^{n \times n}
$$

Intuition: "QR writes $A$ as a rigid rotation/reflection ($Q$) times a triangular factor ($R$). Because $Q$ preserves lengths ($\|Qx\| = \|x\|$), multiplying by $Q$ never amplifies errors; this is what makes QR numerically stable."

## Components

- $A \in \mathbb{R}^{m \times n}$: the matrix to factorize. Can be square ($m = n$) or rectangular ($m > n$).
- $Q \in \mathbb{R}^{m \times m}$: orthogonal matrix. $Q^\mathsf{T}Q = QQ^\mathsf{T} = I$. Columns are orthonormal: each has unit length and is perpendicular to every other column.
- $R \in \mathbb{R}^{m \times n}$: upper triangular. $R_{ij} = 0$ for $i > j$. In thin QR, the bottom $m-n$ rows of $R$ are zero.
- Thin $Q \in \mathbb{R}^{m \times n}$: only the first $n$ columns, still orthonormal ($Q^\mathsf{T}Q = I_n$ but $QQ^\mathsf{T} \neq I_m$).

## Approach 1: Gram-Schmidt

### Classical Gram-Schmidt (CGS)

This is the intuitive approach: orthogonalize one column at a time.

**Formal:**

$$
\begin{aligned}
q_1 &= a_1 / \|a_1\| \\
\tilde{q}_k &= a_k - \sum_{i=1}^{k-1} (q_i^\mathsf{T} a_k)\, q_i, \quad k = 2, \dots, n \\
q_k &= \tilde{q}_k / \|\tilde{q}_k\|
\end{aligned}
$$

**Intuition:**
"At step $k$, take column $a_k$ and subtract off its projection onto every previous $q_i$ in one shot. What's left is orthogonal to all of them. Then normalize."

The $R$ entries fall out naturally: $R_{ik} = q_i^\mathsf{T} a_k$ for $i < k$, and $R_{kk} = \|\tilde{q}_k\|$.

**Why CGS is unstable:** The subtraction $\tilde{q}_k = a_k - \Sigma(q_i^\mathsf{T} a_k) q_i$ suffers from catastrophic cancellation when $\tilde{q}_k$ is small: the projection terms nearly sum to $a_k$ itself. Since $a_k$ is the *original* column and the $q_i$ already have O(ε) errors from earlier steps, the cancellation magnifies those errors. The computed $q_k$ drifts from orthogonality; $Q^\mathsf{T}Q$ can have off-diagonals as large as $O(1)$.

### Modified Gram-Schmidt (MGS)

**Formal:**

$$
\begin{aligned}
v^{(0)}_k &= a_k, \quad k = 1, \dots, n \\
\text{for } i &= 1, \dots, n: \\
q_i &= v^{(i-1)}_i / \|v^{(i-1)}_i\| \\
\text{for } k &= i+1, \dots, n: \\
R_{ik} &= q_i^\mathsf{T} v^{(i-1)}_k \\
v^{(i)}_k &= v^{(i-1)}_k - R_{ik}\, q_i
\end{aligned}
$$

**Intuition:**
"MGS orthogonalizes against the *running residual* $v_k$ at each inner step. Instead of computing all dot products from the original $a_k$ (which lets errors accumulate), each $q_i$ only sees what's left after previous projections. The difference: CGS computes $q_i^\mathsf{T} a_k$ once and subtracts everything. MGS updates $v_k$ after each $q_i$, so later $q_j$ see a cleaner signal."

MGS is backward stable for least squares: the computed $Q$ satisfies $Q^\mathsf{T}Q \approx I + O(\varepsilon)$, and $(Q, R)$ is the exact QR of $A + \Delta A$ with $\|\Delta A\| \leq O(\varepsilon)\|A\|$.

**Cost:** $2mn^2$ flops for both CGS and MGS. But MGS is BLAS-2 (matrix-vector), so it's memory-bound on modern hardware.

## Approach 2: Householder Reflections

This is the method used in production: LAPACK, MATLAB, SciPy, cuSOLVER all use it.

### The Algorithm

A Householder reflector is a matrix that reflects a vector across a hyperplane:

$$
H = I - 2\frac{v v^\mathsf{T}}{v^\mathsf{T}v} \in \mathbb{R}^{m \times m}, \qquad v \in \mathbb{R}^m
$$

$H$ is orthogonal and symmetric ($H^\mathsf{T} = H$, $H^\mathsf{T}H = I$).

**Intuition:**
"$H$ reflects any vector across the hyperplane orthogonal to $v$. If we want to zero out everything below the diagonal in column $k$, we find $v$ such that $H \cdot x = \|x\| e_1$: a vector with zeros below the first entry. Geometrically: take the column $x$, find the mirror image across the plane that bisects $x$ and the first basis vector $e_1$, and reflect. The zeros fall out automatically."

### The Householder Vector

Given $x \in \mathbb{R}^{m-k+1}$ (the portion of column $k$ from row $k$ down), we want $v$ such that $Hx = \|x\| e_1$.

The formula:

$$
v = x + \operatorname{sign}(x_1) \|x\| e_1
$$

where $e_1 = [1, 0, \dots, 0]^\mathsf{T}$.

**Intuition:**
"We take $x$ and add $\|x\| e_1$ (pointing in the direction of the first axis, scaled by $x$'s length). The sum of these two vectors points along the normal of the reflection plane. The sign of $x_1$ ensures we pick the mirror that keeps $v$ away from zero: if $x_1$ is positive, we reflect using $\|x\| e_1$ (same direction); if negative, we use $-\|x\| e_1$ (opposite direction). This avoids $v \approx 0$, which would cause catastrophic cancellation."

After computing $v$, normalize it so $v^\mathsf{T}v = 2$ (this makes the formula $H = I - v v^\mathsf{T}$, saving a multiplication). The reflector can then be applied to any matrix $A$ as:

$$
H A = A - v (v^\mathsf{T} A)
$$

which is a rank-1 update: cheap and stable.

### QR via Householder

For $A \in \mathbb{R}^{m \times n}$:

```
for k = 1..min(m, n):
    x = A[k:m, k]
    v = x + sign(x₁) * ‖x‖ * e₁
    v = v / ‖v‖
    A[k:m, k:n] = A[k:m, k:n] - v * (vᵀ * A[k:m, k:n])
```

After the loop, $A$ has been overwritten by $R$ (upper triangular part). The $Q$ matrix is the product of all reflectors: $Q = H_1 H_2 \cdots H_n$, but it's never formed explicitly; it's stored as the list of $v$ vectors and applied on demand.

**Cost:** $2mn^2 - \frac{2}{3}n^3$ flops. About $2\times$ the cost of LU for square matrices.

## Why QR is Numerically Stable

This is the central selling point of QR, and it comes down to one inequality:

$$
\|Q\|_2 = 1, \qquad \|Q^\mathsf{T}\|_2 = 1
$$

$Q$ is orthogonal. Multiplying by $Q$ preserves the 2-norm: it's a rigid rotation or reflection. Errors in $A$ or $b$ are never amplified when multiplied by $Q^\mathsf{T}$.

Compare solving $Ax = b$ using QR vs. the normal equations $A^\mathsf{T}Ax = A^\mathsf{T}b$:

| Method | Effective condition number | Digits lost for $\text{cond}(A) = 10^8$ |
|--------|--------------------------|----------------------------------------|
| Normal equations | $\text{cond}(A)^2 = 10^{16}$ | **0** (all 16 digits gone) |
| QR ($Rx = Q^\mathsf{T}b$) | $\text{cond}(A) = 10^8$ | ~8 digits |

**Intuition:**
"Normal equations form $A^\mathsf{T}A$, which squares the condition number. If $\text{cond}(A) = 10^8$ (not uncommon for ill-conditioned problems), then $A^\mathsf{T}A$ has $\text{cond} = 10^{16}$. IEEE double precision gives exactly 16 digits. You lose *all* of them. QR never forms $A^\mathsf{T}A$. It operates on the original $A$ and uses orthogonal transformations that preserve norms. You lose only ~8 digits. This is not subtle: it's the difference between a working solver and complete garbage."

**Backward stability:** QR is backward stable. The computed solution $\hat{x}$ is the exact solution of a nearby problem:

$$
(A + \Delta A) \hat{x} = b + \Delta b, \quad \|\Delta A\| \leq c \cdot \varepsilon \|A\|,\; \|\Delta b\| \leq c \cdot \varepsilon \|b\|
$$

where $c$ is a small constant and $\varepsilon$ is machine epsilon. Normal equations do **not** achieve this: the formation of $A^\mathsf{T}A$ introduces O($\varepsilon$) errors that are then amplified by $\text{cond}(A)^2$.

## Approach 3: Givens Rotations (brief)

Givens rotations zero out a single entry at a time using a $2 \times 2$ rotation embedded in the identity. They're more expensive per entry than Householder ($6$ flops per zero vs. $4$), but they allow selective zeroing (useful for sparse matrices or when you need to add a row to an already-factorized QR). Not covered in depth here.

# Blocked Householder Reflections

## The Problem with Standard Householder

Each Householder reflector $H_k$ modifies the entire trailing submatrix $A[k:m, k:n]$ as a rank-1 update:

$$
A[k:m, k:n] := A[k:m, k:n] - v_k (v_k^\mathsf{T} A[k:m, k:n])
$$

This is a **matrix-vector operation** (BLAS-2): one dot product $v_k^\mathsf{T}A$ followed by an outer product update. On modern hardware, BLAS-2 is memory bandwidth bound: the CPU spends most of its time waiting for data from main memory. For large matrices, the unblocked Householder QR runs at $5$–$10\times$ below peak flop/s.

## The WY Representation

Bischof and Van Loan (1987) showed that $k$ consecutive reflectors can be compacted into a compact form:

$$
H_1 H_2 \cdots H_k = I + W Y^\mathsf{T}, \qquad W \in \mathbb{R}^{m \times k},\; Y \in \mathbb{R}^{m \times k}
$$

**Intuition:**
"Instead of applying $k$ reflectors one at a time ($k$ passes over the matrix), we compact them into a single rank-$k$ update: $A := A + W(Y^\mathsf{T} A)$. The term $Y^\mathsf{T}A$ is $(k \times n)$; it fits in cache. Computing $W(Y^\mathsf{T}A)$ is a general matrix multiply (BLAS-3). Cache efficiency goes through the roof."

### Constructing $W$ and $Y$

Given reflectors $v_1, \dots, v_k$ normalized so $v_i^\mathsf{T}v_i = 2$:

$$
\begin{aligned}
W_1 &= -v_1, \quad Y_1 = v_1 \\
\text{for } j &= 2, \dots, k: \\
W_j &= \begin{bmatrix} W_{j-1} & -v_j - W_{j-1} (Y_{j-1}^\mathsf{T} v_j) \end{bmatrix} \\
Y_j &= \begin{bmatrix} Y_{j-1} & v_j \end{bmatrix}
\end{aligned}
$$

After the loop: $W = W_k$, $Y = Y_k$, and $I + WY^\mathsf{T} = H_1 \cdots H_k$.

### Mini example ($k = 2$)

Take two reflectors $H_1 = I - v_1 v_1^\mathsf{T}$ and $H_2 = I - v_2 v_2^\mathsf{T}$ with $v_1^\mathsf{T}v_1 = v_2^\mathsf{T}v_2 = 2$.

Compute:

$$
\begin{aligned}
W_1 &= -v_1, \qquad Y_1 = v_1 \\
t &= Y_1^\mathsf{T} v_2 = v_1^\mathsf{T} v_2 \quad (\text{a scalar}) \\
W_2 &= \begin{bmatrix} -v_1 & -v_2 - (-v_1)(t) \end{bmatrix} = \begin{bmatrix} -v_1 & -v_2 + t v_1 \end{bmatrix} \\
Y_2 &= \begin{bmatrix} v_1 & v_2 \end{bmatrix}
\end{aligned}
$$

Verify: $I + W_2 Y_2^\mathsf{T} = I + (-v_1)v_1^\mathsf{T} + (-v_2 + t v_1)v_2^\mathsf{T} = I - v_1 v_1^\mathsf{T} - v_2 v_2^\mathsf{T} + t v_1 v_2^\mathsf{T}$.

Meanwhile, $H_1 H_2 = (I - v_1 v_1^\mathsf{T})(I - v_2 v_2^\mathsf{T}) = I - v_1 v_1^\mathsf{T} - v_2 v_2^\mathsf{T} + (v_1^\mathsf{T} v_2) v_1 v_2^\mathsf{T}$.

Since $t = v_1^\mathsf{T} v_2$, these are identical. The compact representation is exact.

### Applying the Block

Once $W$ and $Y$ are computed, the trailing matrix update is:

$$
A := (I + W Y^\mathsf{T}) A = A + W (Y^\mathsf{T} A)
$$

1. Compute $C = Y^\mathsf{T} A$: $(k \times m) \cdot (m \times n) = k \times n$, two small matrices with high arithmetic intensity.
2. Compute $A := A + W C$: $A$ plus a rank-$k$ update (another gemm).

Both steps are **BLAS-3** (matrix-matrix). The factorization still computes reflectors one column at a time (that part isn't blocked), but the *application* to the trailing matrix is deferred and batched into a single gemm call.

## Why

- **BLAS-3 performance:** The blocked version runs at near-peak flop/s on modern CPUs and GPUs. Unblocked Householder is memory-bound and typically $5$–$10\times$ slower.
- **Cache friendly:** $Y^\mathsf{T}A$ operates on an $(m \times n)$ matrix but only produces a $(k \times n)$ result. The trailing matrix is streamed through cache once per block instead of once per reflector.
- **Same math:** $I + WY^\mathsf{T}$ is bitwise equivalent (modulo floating-point associativity) to applying the reflectors sequentially. $Q$ and $R$ are identical.
- **Block size $k$:** typically $32$–$64$, chosen so $W$ and $Y$ fit in L2 cache. LAPACK's `DGEQRF` uses `ILAENV` to pick $k$ automatically based on the matrix size and cache geometry.

# Worked Example

Let's work through a concrete $3 \times 2$ matrix to see both LU and QR in action.

$$
A = \begin{bmatrix}
3 & 1 \\
1 & 2 \\
2 & 3
\end{bmatrix} \in \mathbb{R}^{3 \times 2}
$$

## LU Decomposition ($PA = LU$)

Since $A$ is $3 \times 2$, we need a square system. Let's work with the square $2 \times 2$ submatrix formed by the first two rows:

$$
A_{2\times 2} = \begin{bmatrix} 3 & 1 \\ 1 & 2 \end{bmatrix}
$$

**Step 1 (column 1):** Pivot is $a_{11} = 3$. No swap needed. Multipliers: $m_{21} = 1/3$.

$$
A^{(1)} = \begin{bmatrix} 3 & 1 \\ 0 & 2 - \frac{1}{3} \cdot 1 \end{bmatrix} = \begin{bmatrix} 3 & 1 \\ 0 & \frac{5}{3} \end{bmatrix} = U
$$

**Step 2:** The multipliers go into $L$:

$$
L = \begin{bmatrix} 1 & 0 \\ \frac{1}{3} & 1 \end{bmatrix}, \qquad U = \begin{bmatrix} 3 & 1 \\ 0 & \frac{5}{3} \end{bmatrix}
$$

**Verify:** $LU = \begin{bmatrix} 1 \cdot 3 + 0 \cdot 0 & 1 \cdot 1 + 0 \cdot \frac{5}{3} \\ \frac{1}{3} \cdot 3 + 1 \cdot 0 & \frac{1}{3} \cdot 1 + 1 \cdot \frac{5}{3} \end{bmatrix} = \begin{bmatrix} 3 & 1 \\ 1 & 2 \end{bmatrix} = A_{2 \times 2}$.

No pivoting needed since the pivot was already the largest in its column.

## QR Decomposition via Gram-Schmidt

**Step 1:** $q_1 = a_1 / \|a_1\|$:

$$
a_1 = \begin{bmatrix} 3 \\ 1 \\ 2 \end{bmatrix}, \quad \|a_1\| = \sqrt{3^2 + 1^2 + 2^2} = \sqrt{14}
$$

$$
q_1 = \frac{1}{\sqrt{14}} \begin{bmatrix} 3 \\ 1 \\ 2 \end{bmatrix} \approx \begin{bmatrix} 0.8018 \\ 0.2673 \\ 0.5345 \end{bmatrix}
$$

$R_{11} = \|a_1\| = \sqrt{14} \approx 3.7417$.

**Step 2:** Project $a_2$ onto $q_1$ and subtract:

$$
R_{12} = q_1^\mathsf{T} a_2 = \frac{1}{\sqrt{14}} (3 \cdot 1 + 1 \cdot 2 + 2 \cdot 3) = \frac{11}{\sqrt{14}} \approx 2.9399
$$

$$
\tilde{q}_2 = a_2 - R_{12} q_1 = \begin{bmatrix} 1 \\ 2 \\ 3 \end{bmatrix} - \frac{11}{\sqrt{14}} \cdot \frac{1}{\sqrt{14}} \begin{bmatrix} 3 \\ 1 \\ 2 \end{bmatrix} = \begin{bmatrix} 1 \\ 2 \\ 3 \end{bmatrix} - \frac{11}{14} \begin{bmatrix} 3 \\ 1 \\ 2 \end{bmatrix}
$$

$$
\tilde{q}_2 = \begin{bmatrix} 1 - \frac{33}{14} \\ 2 - \frac{11}{14} \\ 3 - \frac{22}{14} \end{bmatrix} = \begin{bmatrix} -\frac{19}{14} \\ \frac{17}{14} \\ \frac{20}{14} \end{bmatrix} = \frac{1}{14} \begin{bmatrix} -19 \\ 17 \\ 20 \end{bmatrix}
$$

$R_{22} = \|\tilde{q}_2\| = \frac{1}{14} \sqrt{(-19)^2 + 17^2 + 20^2} = \frac{\sqrt{1170}}{14} \approx 2.4431$.

$$
q_2 = \frac{\tilde{q}_2}{\|\tilde{q}_2\|} = \frac{1}{\sqrt{1170}} \begin{bmatrix} -19 \\ 17 \\ 20 \end{bmatrix} \approx \begin{bmatrix} -0.5555 \\ 0.4970 \\ 0.5847 \end{bmatrix}
$$

**Result:**

$$
Q = \begin{bmatrix}
0.8018 & -0.5555 \\
0.2673 & 0.4970 \\
0.5345 & 0.5847
\end{bmatrix}, \qquad
R = \begin{bmatrix}
3.7417 & 2.9399 \\
0 & 2.4431
\end{bmatrix}
$$

**Verify $Q^\mathsf{T}Q$:**

$$
Q^\mathsf{T}Q = \begin{bmatrix}
0.8018 & 0.2673 & 0.5345 \\
-0.5555 & 0.4970 & 0.5847
\end{bmatrix}
\begin{bmatrix}
0.8018 & -0.5555 \\
0.2673 & 0.4970 \\
0.5345 & 0.5847
\end{bmatrix}
= \begin{bmatrix}
1.0000 & 0.0000 \\
0.0000 & 1.0000
\end{bmatrix}
$$

Orthonormal to machine precision.

**Verify $QR = A$:**

$$
QR = \begin{bmatrix}
0.8018 & -0.5555 \\
0.2673 & 0.4970 \\
0.5345 & 0.5847
\end{bmatrix}
\begin{bmatrix}
3.7417 & 2.9399 \\
0 & 2.4431
\end{bmatrix}
= \begin{bmatrix}
3 & 1 \\
1 & 2 \\
2 & 3
\end{bmatrix}
$$

Matches $A$ exactly (up to floating-point rounding).

## QR Decomposition via Householder

**Step 1:** Take the first column $x = a_1 = [3, 1, 2]^\mathsf{T}$.

$x_1 = 3 > 0$, so $\operatorname{sign}(x_1) = +1$.

$$
v = x + \|x\| e_1 = \begin{bmatrix} 3 \\ 1 \\ 2 \end{bmatrix} + \sqrt{14} \begin{bmatrix} 1 \\ 0 \\ 0 \end{bmatrix} = \begin{bmatrix} 3 + 3.7417 \\ 1 \\ 2 \end{bmatrix} = \begin{bmatrix} 6.7417 \\ 1 \\ 2 \end{bmatrix}
$$

Normalize so $v^\mathsf{T}v = 2$: $\|v\| = \sqrt{6.7417^2 + 1^2 + 2^2} \approx 7.1362$, $v \leftarrow \frac{v}{\|v\|/\sqrt{2}} = \frac{\sqrt{2} v}{\|v\|}$.

This gives $v_1 \approx [0.9445, 0.1401, 0.2802]^\mathsf{T}$ with $v^\mathsf{T}v = 2$.

Apply $H_1 = I - v_1 v_1^\mathsf{T}$ to $A$:

$$
H_1 A = A - v_1 (v_1^\mathsf{T} A)
$$

$v_1^\mathsf{T} A = [3.7417, 3.1128]$. After the update:

$$
H_1 A = \begin{bmatrix}
-3.7417 & -2.9399 \\
0 & \text{*} \\
0 & \text{*}
\end{bmatrix}
$$

The first column is now $[-3.7417, 0, 0]^\mathsf{T}$: zero below the diagonal, as promised. The rest of $R$ continues to fill in.

**Step 2:** Take the subcolumn $x = \tilde{a}_2[2:3]$ (rows 2–3 of the updated second column) and repeat. The final $R$ matches the Gram-Schmidt result (up to sign flips in rows of $R$, since Householder can introduce row sign changes that Gram-Schmidt doesn't).

# Quick Reference

| Aspect | LU ($PA = LU$) | QR (Householder) |
|--------|---------------|-------------------|
| Cost ($n \times n$) | $\frac{2}{3}n^3$ | $\frac{4}{3}n^3$ |
| Cost ($m \times n$, $m > n$) | N/A (square only) | $2mn^2 - \frac{2}{3}n^3$ |
| Least squares | ✗ (needs QR or SVD) | ✓ backward stable |
| cond sensitivity | $\text{cond}(A)$ | $\text{cond}(A)$ |
| Square $Ax = b$ | ✓ cheaper | ✓ ($2\times$ cost) |
| BLAS level (unblocked) | 2 (panel) | 2 |
| BLAS level (blocked) | 3 | 3 (WY representation) |
| Stability w/o pivoting | ❌ exponential growth | ✅ always stable |

# Notes

- LU is the default for square $Ax = b$ when the matrix is well-conditioned (roughly $2\times$ cheaper than QR). For ill-conditioned systems ($\text{cond}(A) > 10^6$), QR is safer even for square matrices.
- For least squares $\min \|Ax - b\|_2$, QR is the standard choice. The normal equations $A^\mathsf{T}Ax = A^\mathsf{T}b$ should be avoided unless $\text{cond}(A)$ is tiny, because squaring the condition number loses digits rapidly.
- LAPACK's `DGEQRF` uses blocked Householder with the WY representation. `DGELS` solves least squares via QR. MATLAB's `\` operator picks QR automatically for rectangular $A$.
- Modified Gram-Schmidt is backward stable but BLAS-2 (pedagogical but not used in production libraries). However, MGS is useful when you need to maintain orthogonality incrementally (e.g., in Krylov subspace methods like GMRES).
- The WY blocking trick generalizes beyond QR. The same idea (compact a panel's transformations into a rank-$k$ update) is used in blocked LU and blocked Cholesky.
- Block size $k$ is typically $32$–$64$ in LAPACK. Larger $k$ gives better BLAS-3 performance but increases the cost of forming $W$ and $Y$. The optimal $k$ depends on the cache hierarchy and is determined by `ILAENV`.
- For very large matrices on distributed memory, communication-avoiding QR (CAQR, Demmel et al., 2008) reorganizes the panel factorization itself (not just the trailing matrix update) to reduce network traffic by a factor of $O(\sqrt{n})$.
