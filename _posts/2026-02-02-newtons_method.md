---
layout: single
title: "Newton's Method for Root Finding"
description: "A simple yet powerful algorithm for finding roots of functions, with quadratic convergence near simple roots."
date: 2026-02-02
categories: [math]
author_profile: false
show_excerpts: false
sidebar:
  nav: "math"
---

Newton's method (Newton–Raphson) is an iterative algorithm for finding roots of a real-valued function $f : \mathbb{R} \to \mathbb{R}$. Given an initial guess $x_0$, it produces a sequence $x_0, x_1, x_2, \dots$ that converges to a root $x^*$ where $f(x^*) = 0$, provided the function is smooth and the guess is close enough. The update comes from a first-order Taylor approximation. I will try to post about Taylor series expansion in the future, but for now, use this resource [taylor series expansion](https://mathworld.wolfram.com/TaylorSeries.html). Basically, we approximate $f$ near $x_n$ by its tangent line at $x_n$, and find where that line crosses the $x$-axis. This is the next guess $x_{n+1}$. This will be explained further in the derivation section below.

# Iterative/Recursive Formula

$$
x_{n+1} = x_n - \frac{f(x_n)}{f'(x_n)}
$$

## Components

- $x_n$: current approximation at step $n$. Initialized to $x_0$, the user's best guess.
- $f(x_n)$: value of the target function at $x_n$. The method converges when $|f(x_n)|$ is small.
- $f'(x_n)$: derivative of $f$ at $x_n$. This is the slope of the tangent line. If $f'(x_n) = 0$, the method fails at this step (division by zero).
- $x_{n+1}$: next approximation. The $x$-intercept of the line through $(x_n, f(x_n))$ with slope $f'(x_n)$.

## Derivation

The tangent line to $f$ at $x_n$ is

$$
L(x) = f(x_n) + f'(x_n)(x - x_n)
$$
where $y = f(x_n)$ is the y-intercept and $f'(x_n)$ is the slope. We used the known point $(x_n, f(x_n))$ to write the line in point-slope form.

Set $L(x) = 0$ (find where the tangent crosses the $x$-axis) 

this is because, the tangent line is a linear approximation to $f$ near $x_n$, so we hope its root is a better guess than $x_n$ itself. Solving for $x$ gives

now solve for $x$:

$$
0 = f(x_n) + f'(x_n)(x - x_n)
\;\Longrightarrow\;
x = x_n - \frac{f(x_n)}{f'(x_n)}
$$

That $x$ becomes $x_{n+1}$. The same result falls out of a first-order Taylor expansion: $0 = f(x^*) \approx f(x_n) + f'(x_n)(x^* - x_n)$ gives $x^* \approx x_n - f(x_n)/f'(x_n)$.



# Convergence

Near a simple root ($f(x^*) = 0$, $f'(x^*) \neq 0$), Newton's method converges **quadratically**:

$$
|x_{n+1} - x^*| \le C |x_n - x^*|^2
$$

This means the number of correct digits roughly doubles at each step, which is much faster than bisection or secant methods.

## Why

- **Quadratic convergence** is the main draw. With a good initial guess, you reach machine precision in 4–6 iterations. Bisection, by contrast, gains one bit per iteration (linear convergence, $C = 0.5$). For engineering, you pay for the speed with the need for $f'(x_n)$.
- **Self-correcting:** a bad step is not catastrophic: the next iteration still uses the true $f$ and $f'$ at the new point, so the method can recover if the new point is closer.
- **Generalizes naturally** to systems of equations (multivariate Newton's method with the Jacobian) and to optimization (Newton's method for finding critical points, using the Hessian). This is why Newton-style updates appear in second-order optimizers.

# Improved Variant: Halley's Method

If the second derivative $f''(x_n)$ is available, we can do better than Newton. Instead of a linear tangent approximation, fit a **local quadratic model** using the first two terms of the Taylor expansion:

$$
0 = f(x^*) \approx f(x_n) + f'(x_n)(x^* - x_n) + \frac{1}{2}f''(x_n)(x^* - x_n)^2
$$

Solving this quadratic for $(x^* - x_n)$ gives Halley's update:

$$
x_{n+1} = x_n - \frac{2f(x_n)f'(x_n)}{2[f'(x_n)]^2 - f(x_n)f''(x_n)}
$$

## Why

- **Cubic convergence:** near a simple root, the error shrinks as $|x_{n+1} - x^*| \le C|x_n - x^*|^3$. Correct digits roughly **triple** each iteration instead of doubling. For the $\sqrt{2}$ example above, Halley's method reaches $10^{-12}$ in 3 iterations instead of 4.
- **More robust to curvature:** the denominator has a built-in correction for sharp bends in $f$. Halley's method converges on some problems where Newton oscillates or overshoots.
- **Cost tradeoff:** you need $f''(x_n)$ per step. For analytic functions this is one extra derivative, but for black-box functions the finite-difference cost doubles. In practice, Halley is used when $f$ is cheap and high precision is needed quickly (e.g. computing elementary functions (exp, log, trig) in math libraries).

The same idea extends to even higher derivatives (Householder methods), but Halley is the most practical second-order variant.

# Failure Modes

Newton's method is not robust out of the box. It fails when:

1. **Zero derivative** $f'(x_n) = 0$: the tangent is horizontal, never crosses the axis. The update divides by zero. This is addressed by halley's method, which uses the second derivative to avoid this issue. In practice, a small damping factor or line search can also help prevent large jumps when $f'(x_n)$ is small but nonzero.
2. **Poor initial guess:** if $x_0$ is far from the root, the tangent may point in the wrong direction and diverge.
3. **Oscillation:** for some functions (e.g. $f(x) = x^3 - 2x + 2$ with $x_0 = 0$), the iterates cycle without converging.
4. **Multiple roots:** at a repeated root where $f'(x^*) = 0$, convergence drops to linear and the constant $C$ grows large.

## Why

- **No global guarantee:** unlike bisection (which always converges given a sign change), Newton's method makes no such promise. It is a local method, provably convergent only within a "basin of attraction" around the root.
- **Derivative cost:** you must compute $f'(x_n)$ at every step. For analytic functions this is cheap; for black-box simulations it may require finite differences, adding cost and truncation error.

# Worked Example

Find $\sqrt{2}$ (i.e. solve $f(x) = x^2 - 2 = 0$).

Here $f'(x) = 2x$, and the Newton update is

$$
x_{n+1} = x_n - \frac{x_n^2 - 2}{2x_n}
 = \frac{x_n + 2/x_n}{2}
$$

Start with $x_0 = 2$:

| $n$ | $x_n$ | $f(x_n)$ | error $|x_n - \sqrt{2}|$ |
|-----|-------|-----------|--------------------------|
| 0 | 2.00000000 | 2.0 | $4.14 \times 10^{-1}$ |
| 1 | 1.50000000 | 0.25 | $8.58 \times 10^{-2}$ |
| 2 | 1.41666667 | $6.94 \times 10^{-3}$ | $2.45 \times 10^{-3}$ |
| 3 | 1.41421569 | $6.01 \times 10^{-6}$ | $2.12 \times 10^{-6}$ |
| 4 | 1.41421356 | $4.51 \times 10^{-12}$ | $1.59 \times 10^{-12}$ |

After 4 iterations the error is at $10^{-12}$. Doubling the correct digits each step is the hallmark of quadratic convergence. For comparison, bisection on the interval $[1, 2]$ would need about 40 iterations to reach the same precision (one bit per iteration).

# Quick Reference

| Aspect | Newton's Method | Bisection |
|--------|----------------|-----------|
| Convergence rate | Quadratic ($C|x_n - x^*|^2$) | Linear ($0.5|x_n - x^*|$) |
| Needs derivative | Yes ($f'(x_n)$) | No |
| Robustness | Local (diverges if far) | Global (given sign change) |
| Iterations to $10^{-12}$ | ~5 | ~40 |
| Per-iteration cost | $f + f'$ eval | 1 function eval |

# Notes

- The update $x_{n+1} = x_n - f(x_n)/f'(x_n)$ is the same as applying Newton's method in optimization to find $f'(x) = 0$, where the update becomes $x_{n+1} = x_n - f'(x_n)/f''(x_n)$. The root-finding and optimization versions are the same idea at different orders of differentiation.
- In practice, Newton's method is often hybridized: bisection or bracketing ensures robustness for the first few steps, then Newton takes over for fast convergence near the root. This is what `sqrt` implementations in hardware and GLIBC do.
- For high-dimensional systems, the Jacobian replaces $f'(x_n)$ and a linear solve replaces division. The cost scales as $O(d^3)$ for dense systems, which is why quasi-Newton methods (BFGS, L-BFGS) are preferred in ML.
- The denominator $f'(x_n)$ can be zero in exact arithmetic but often just very small in floating point; the iteration can still produce a huge jump, overshooting the root. A damping factor or line search is the standard fix.
