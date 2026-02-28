---
layout: single
title: "Feed-Forward Networks"
description: "Building non-linear mappings by stacking matrix multiplications and activations, and why this pattern is the workhorse of deep learning"
date: 2026-02-28
categories: [ml]
author_profile: false
show_excerpts: false
sidebar:
  nav: "ml"
---

A feed-forward network (FFN) is a sequence of linear transformations separated by non-linear activation functions. You take an input $x$, multiply by a weight matrix, add a bias, push each entry through a non-linearity, then repeat. The result is a function that can approximate almost anything.

$$
\begin{aligned}
h_0 &= x \\
h_\ell &= \sigma_\ell(W_\ell h_{\ell-1} + b_\ell), \qquad \ell = 1, \dots, L-1 \\
f(x) &= W_L h_{L-1} + b_L
\end{aligned}
$$

$$
W_\ell \in \mathbb{R}^{d_\ell \times d_{\ell-1}}, \quad b_\ell \in \mathbb{R}^{d_\ell}, \quad \ell = 1, \dots, L
$$

$d_0$ is the input dimension, $d_1, \dots, d_{L-1}$ are the hidden dimensions (widths), and $d_L$ is the output dimension. Each $W_\ell$ mixes the entries of the previous layer, $b_\ell$ shifts the result, and $\sigma_\ell$ breaks the linearity. The output layer usually skips the activation so the output can be any real value.

## Why You Need the Activation

If you drop all $\sigma_\ell$, the whole chain collapses:

$$
W_L (W_{L-1} (\cdots (W_1 x + b_1) \cdots) + b_{L-1}) + b_L = \widetilde{W} x + \widetilde{b}
$$

No matter how deep, a stack of linear layers is just one linear transformation. Without non-linearity the network can only learn straight-line decision boundaries.

The textbook example is XOR. The four points $(0, 0)$, $(0, 1)$, $(1, 0)$, $(1, 1)$ have labels $0, 1, 1, 0$ (true when exactly one input is 1). No single line separates the 0s from the 1s. But a two-layer FFN can: the hidden layer projects the input into a space where the classes are linearly separable, and the output layer draws the separating line.

**Intuition:**
"Each hidden neuron detects a pattern: it fires (activates) when the input matches a certain weighted combination. The output layer then combines these detected patterns into a decision. More hidden neurons means more patterns can be memorized, and more layers means patterns can build on simpler patterns."

## Two-Layer Case and Universal Approximation

The simplest useful FFN has one hidden layer:

$$
f(x) = W_2 \sigma(W_1 x + b_1) + b_2
$$

$$
W_1 \in \mathbb{R}^{d \times h}, \; W_2 \in \mathbb{R}^{h \times o}, \; b_1 \in \mathbb{R}^h, \; b_2 \in \mathbb{R}^o
$$

$h$ is the hidden width (number of feature detectors). Each column of $W_1$ is a direction in input space, and $\sigma$ decides whether the input projects onto that direction.

The Universal Approximation Theorem (Cybenko 1989, Hornik 1991) says: with enough hidden neurons, a single-hidden-layer FFN can approximate any continuous function on a bounded domain, to any accuracy you want. This is why FFNs are everywhere: they are general-purpose function approximators.

**Intuition:**
"Each hidden neuron contributes a 'bump' to the function. The output layer is a weighted sum of these bumps. With enough bumps you can build any shape. It is the same idea as a Fourier series or a Taylor expansion, except the network learns the basis functions from data instead of using fixed sines or polynomials."

## Activation Functions

Every activation is a trade-off between gradient flow, computational cost, and output range:

| Function | Formula | Gradient | Notes |
|----------|---------|----------|-------|
| ReLU | $\max(0, x)$ | $1$ if $x > 0$, $0$ otherwise | Cheap, sparse activations. Dead neurons if $x < 0$ always. |
| GeLU | $x \Phi(x)$ | smooth, approx $\sigma(1.702x)$ | Smooth ReLU. Used in BERT, GPT-2. |
| SiLU (Swish) | $x \sigma(x)$ | $\sigma(x) + x \sigma(x)(1 - \sigma(x))$ | Self-gated. Used in Llama, Mistral (via SwiGLU). |
| Tanh | $\frac{e^x - e^{-x}}{e^x + e^{-x}}$ | $1 - \tanh^2(x)$ | Zero-centered. Saturates at extremes. |
| Sigmoid | $\frac{1}{1 + e^{-x}}$ | $\sigma(x)(1 - \sigma(x))$ | Outputs in $(0, 1)$. Good for gating, bad for deep nets. |

**ReLU** is the default for most networks: one comparison, gradient is either 0 or 1, and it produces sparse activations (many neurons output exactly 0). The downside is the "dead ReLU" problem: once a neuron's weights push its input below zero for every training example, its gradient is always 0 and it never recovers.

**GeLU** and **SiLU** are smooth alternatives. They avoid the hard kink at 0 and gradient flows through them more consistently. They cost more to compute but produce better results at the same model size. All modern LLMs use SiLU inside a gated FFN (SwiGLU).

**Sigmoid** and **tanh** were the standard before ReLU. They saturate at large positive and negative values, which kills gradients in deep networks. They are still used when a bounded output is needed (e.g., LSTM gates need values in $(0, 1)$).

## Where FFNs Show Up

FFNs are the decision-making backbone of most architectures. The feature extractor (convolution, attention, recurrence) produces a representation; the FFN maps that representation to the final output.

- **CNN classifier head:** after convolutional layers extract spatial features, a few fully-connected layers (an FFN) map the feature vector to class scores. This is the standard pattern in ResNet, VGG, and EfficientNet.
- **Transformer FFN sub-layer:** every transformer block has a two-layer (or gated three-layer) FFN after attention. Attention handles inter-token mixing; the FFN processes each token independently through an expansion-activation-contraction pattern. Every major LLM (GPT, Llama, DeepSeek) uses this design.
- **LSTM/GRU gates:** the forget, input, and output gates are linear projections followed by sigmoid (to get a value in $(0, 1)$). Each gate is a small FFN with a fixed sigmoid activation.
- **RNN output projection:** the hidden-to-output mapping at each timestep is a linear layer (a depth-1 FFN without activation).

## Components

- $x \in \mathbb{R}^{d_0}$: input vector (e.g., flattened image pixels, token embedding).
- $W_\ell \in \mathbb{R}^{d_\ell \times d_{\ell-1}}$: weight matrix for layer $\ell$. Entry $(i, j)$ controls how much unit $j$ from layer $\ell-1$ contributes to unit $i$ in layer $\ell$.
- $b_\ell \in \mathbb{R}^{d_\ell}$: bias vector for layer $\ell$. Shifts the activation threshold.
- $\sigma_\ell$: activation function, applied element-wise. The only source of non-linearity.
- $h_\ell$: hidden state after layer $\ell$ (post-activation).
- $L$: depth (number of weight matrices). Deeper networks can represent hierarchical features: earlier layers detect simple edges or patterns, later layers combine them into concepts.
- $d_1, \dots, d_{L-1}$: hidden widths. Wider layers can detect more features per layer. Most models use a hidden width larger than the input (e.g., $4\times$ in Transformers).

## Quick Reference

| Aspect | One Layer | Two Layer | Deep ($L > 2$) |
|--------|-----------|-----------|-----------------|
| Expressivity | Linear only | Universal approximator | Universal, more parameter-efficient |
| Cost (forward) | $O(d_o d_i)$ | $O(d h + h o)$ | $O(\sum d_\ell d_{\ell-1})$ |
| Gradient flow | Trivial | OK | Needs residual connections and normalization |
| When to use | Output projection only | Small classifiers, simple tasks | Modern DNNs for complex tasks |

## Notes

- The number of parameters in an FFN is $\sum_{\ell=1}^L (d_\ell d_{\ell-1} + d_\ell)$ (weights + biases). Most of the cost is in the weight matrices.
- Training uses backpropagation: the chain rule computes $\partial f / \partial W_\ell$ and $\partial f / \partial b_\ell$ from the output layer backward. The gradient flows through the linear transform and the activation derivative at each step.
- In practice, FFNs are almost never trained alone. They are stacked with convolutions (CNN), attention (Transformer), or recurrence (RNN). The FFN provides the non-linear mapping; the surrounding layers handle structure (spatial, sequential, pairwise).
- The two-layer FFN with ReLU has a geometric interpretation: the first layer partitions the input space into convex polytopes, and the second layer assigns a value to each region. Deep FFNs create a hierarchy of partitions: earlier layers carve coarse regions, later layers subdivide them.
