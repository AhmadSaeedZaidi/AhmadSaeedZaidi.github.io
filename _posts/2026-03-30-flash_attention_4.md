---
layout: single
title: "FlashAttention-4: When Tensor Cores Got Too Fast for Everything Else"
description: "A deep dive into how FlashAttention-4 co-designs algorithms and kernels to tame asymmetric hardware scaling on NVIDIA Blackwell GPUs"
date: 2026-03-30 20:43:00
categories: [mlsys, hardware]
author_profile: false
show_excerpts: false
sidebar:
  nav: "mlsys"
---

Every generation of GPU hardware promises faster matrix multiplication. Blackwell delivers on that promise — doubling tensor core throughput to 2.25 PFLOPS for BF16 compared to Hopper's 1 PFLOP. But here's the twist: **everything else stayed roughly the same.** Shared memory bandwidth? Unchanged. Exponential unit throughput? Unchanged. The result is a fundamentally lopsided machine where the matmul engine is screaming ahead while the rest of the chip struggles to keep up.

FlashAttention-4, from Tri Dao's group at Princeton (with collaborators at Meta, Colfax Research, and NVIDIA), is the response to this new reality. Rather than porting FlashAttention-3's Hopper-optimized kernels forward — which would either leave performance on the table or simply not compile — FA4 rebuilds from scratch around Blackwell's actual bottlenecks. The results: up to 1613 TFLOPS/s on B200 (71% utilization), 1.3× faster than cuDNN 9.13, and 2.7× faster than Triton.

Let's break down what changed and why.

---

## The Problem: Asymmetric Hardware Scaling

On Hopper (H100), the performance balance between tensor cores, shared memory, and special function units was tight enough that a well-pipelined kernel could keep most resources busy. On Blackwell (B200), that balance is shattered.

Here are the numbers per SM per clock cycle:

| Resource | Hopper (H100) | Blackwell (B200) | Scaling |
|---|---|---|---|
| Tensor core (BF16 MMA) | 4,096 ops | 8,192 ops | **2×** |
| Shared memory bandwidth | 128 bytes | 128 bytes | **1×** |
| Exponential unit (MUFU) | 16 ops | 16 ops | **1×** |

The implication is stark. On Hopper, the attention kernel was roughly compute-bound — the bottleneck was tensor core throughput. On Blackwell, the bottleneck has shifted to **shared memory traffic** and **exponential unit throughput**. The roofline analysis in the paper confirms this: for a typical 128³ tile configuration, SMEM and exponential each take ~1024 cycles while MMA also takes ~1024 cycles. They're neck-and-neck. But for larger tiles (256 × 128²), MMA and exponential both take 2048 cycles — meaning the kernel now has *two* co-dominant bottlenecks that are no longer matmul.

The forward pass bottleneck is the exponential unit. The backward pass bottleneck is shared memory bandwidth, exceeding MMA compute time by ~30%.

---

## What's New in Blackwell Hardware

Before diving into the algorithmic fixes, it's worth understanding the new hardware primitives that FA4 exploits.

### Tensor Memory (TMEM)

Blackwell introduces **256 KB of tensor memory per SM** — a new level in the memory hierarchy specifically coupled to the tensor cores. On Hopper, MMA results were written back to registers, which created enormous register pressure and limited tile sizes. On Blackwell, MMA outputs go directly to TMEM, asynchronously. This is a game-changer for two reasons:

1. **Register pressure relief.** Intermediate accumulator values no longer consume the limited 256-register-per-thread budget.
2. **Decoupled execution.** Since MMA writes to TMEM are fully async, the tensor cores don't block on register writeback, enabling better overlap with other operations.

TMEM is allocated in 32-column (16 KB) granules and requires explicit programmer management — it's not a transparent cache.

### Larger Tile Sizes

Blackwell MMA instructions operate on **128 × 128** tiles (doubled from Hopper's 64 × 128). This means fewer instructions per output tile, but also that each instruction touches more shared memory — amplifying the SMEM bandwidth bottleneck.

### 2-CTA MMA Mode

Blackwell supports a cooperative mode where two CTAs within a cluster jointly execute a single MMA. The accumulator is partitioned along the M dimension, and — critically — each CTA only stages **half** of operand B in its shared memory while the hardware consumes the combined B tile. This effectively halves redundant SMEM traffic for operand B.

---

## Fix #1: Redesigned Pipeline for MMA & Softmax Overlap

### The Core Idea

Since softmax computation (exponentials, max, normalization) is now a co-bottleneck with MMA, the pipeline must be designed so that **while one tile's MMA is running, another tile's softmax is being computed** — and vice versa. FA4 uses a **ping-pong schedule** (similar in spirit to FA3) where two output tiles are computed per thread block, but the implementation is fundamentally different due to Blackwell's TMEM and larger tiles.

### FA3 vs FA4: Thread Assignment

On Hopper (FA3), each row of an accumulator tile was distributed across **4 threads in an interleaved pattern**. Computing a row-wise operation like softmax required **inter-warp shuffles** — threads had to exchange partial results to reconstruct full rows before computing the row max and row sum. This shuffle overhead was a tax on every softmax step.

<!-- 🔄 CHANGED: Added FA3 code snippet showing quad_allreduce shuffle -->

Here's how FA3 handles it in C++ — notice the `quad_allreduce_` call that performs cross-thread reduction, required because each thread only holds a partial row:

```cpp
// FA3 (Hopper) — hopper/softmax.h
// Each thread holds a partial row. 4 threads per row in interleaved layout.
// After computing local row_sum, must shuffle across threads to get the full sum.

__forceinline__ __device__ TensorT finalize(float const final_scale=1.f) {
    SumOp<float> sum_op;
    quad_allreduce_(row_sum, row_sum, sum_op);  // ← Cross-thread shuffle required!
    TensorT scores_scale;
    #pragma unroll
    for (int mi = 0; mi < size(row_sum); ++mi) {
        float sum = row_sum(mi);
        float inv_sum = (sum == 0.f || sum != sum) ? 0.f : 1.f / sum;
        scores_scale(mi) = inv_sum * final_scale;
    }
    return scores_scale;
};
```

On Blackwell (FA4), the 128×128 tile and TMEM layout allow **each thread to own an entire row of 128 elements**. Two warpgroups of 128 threads each handle softmax, with each thread processing one complete row. This eliminates inter-warp shuffles entirely:

<!-- 🔄 CHANGED: Added FA4 code snippet showing no-shuffle softmax -->

```python
# FA4 (Blackwell) — flash_attn/cute/softmax.py
# Each thread owns an entire row → no shuffle needed.
# SoftmaxSm100 operates on 1 row per thread (num_rows=1).

@staticmethod
def create(
    scale_log2: Float32,
    rescale_threshold: cutlass.Constexpr[float] = 0.0,
    softmax_scale: Float32 | None = None,
):
    num_rows = 1  # ← Each thread owns exactly 1 full row. No shuffle.
    arch = 100
    row_max = cute.make_rmem_tensor(num_rows, Float32)
    row_sum = cute.make_rmem_tensor(num_rows, Float32)
    return SoftmaxSm100(
        scale_log2, num_rows, row_max, row_sum, arch, softmax_scale,
        rescale_threshold=rescale_threshold,
    )
```

The difference is structural: FA3's `kNRows` is typically 4–8 (partial rows distributed across threads), requiring `quad_allreduce_` at finalization. FA4's `num_rows = 1` means each thread computes the complete row max and row sum locally — zero cross-thread communication.

### Decoupled Rescaling via Correction Warpgroup

In FA3, the online softmax rescaling step — where previous output tiles are multiplied by `exp(m_old - m_new)` to account for updated row maxima — happened inline within the softmax warpgroup. This put rescaling squarely on the **critical path**.

<!-- 🔄 CHANGED: Added FA3 rescaling code snippet -->

```cpp
// FA3 (Hopper) — hopper/softmax.h
// Rescaling happens inline in the same warpgroup that computes softmax.
// This is on the critical path — must complete before the next MMA.

template<typename Tensor1>
__forceinline__ __device__ void rescale_o(Tensor1 &acc_o, TensorT const &scores_scale) {
    Tensor acc_o_rowcol = make_tensor(acc_o.data(),
        flash::convert_layout_acc_rowcol(acc_o.layout()));
    #pragma unroll
    for (int mi = 0; mi < size<0>(acc_o_rowcol); ++mi) {
        #pragma unroll
        for (int ni = 0; ni < size<1>(acc_o_rowcol); ++ni) {
            acc_o_rowcol(mi, ni) *= scores_scale(mi);  // ← On the critical path
        }
    }
};
```

In FA4, because intermediate results are communicated via TMEM rather than registers, the rescaling is offloaded to a separate **"correction" warpgroup**. The softmax warpgroups compute P and hand off rescale statistics through TMEM; the correction warpgroup applies the rescaling independently. This takes rescaling off the critical path of the main softmax + MMA pipeline.

<!-- 🔄 CHANGED: Added explanation of why TMEM enables decoupling -->

The key enabler is TMEM: on Hopper, accumulators live in registers which are thread-private — there's no way for a separate warpgroup to access another warpgroup's accumulator. On Blackwell, accumulators live in TMEM which is accessible by all warpgroups within the same SM. This allows the correction warpgroup to read rescale statistics and apply them to the output accumulator independently.

The overall warpgroup assignment per thread block:

| Warpgroup | Role |
|---|---|
| WG0 | Tensor core driver + TMA (data movement) |
| WG1 | Softmax (tile H — "high" Q tile) |
| WG2 | Softmax (tile L — "low" Q tile) |
| WG3 | Correction (rescaling) |

<!-- 🔄 CHANGED: Added FA3 warpgroup assignment comparison -->

Compare with FA3's warp specialization, where the division was simpler:

```cpp
// FA3 (Hopper) — hopper/flash_fwd_kernel_sm90.h
// Warp group 0 = Producer (TMA loads), Warp groups 1+ = Consumer (MMA + softmax)
// Softmax and rescaling happen in the SAME consumer warpgroup.

pipeline_params_k.role = warp_group_idx == 0
    ? MainloopPipelineK::ThreadCategory::Producer   // WG0: data movement
    : MainloopPipelineK::ThreadCategory::Consumer;   // WG1+: MMA + softmax + rescaling

if (warp_group_idx == 0) {  // Producer
    cutlass::arch::warpgroup_reg_dealloc<LoadRegisterRequirement>();
    // ... TMA loads only
} else {  // Consumer — does EVERYTHING: MMA, softmax, rescaling
    cutlass::arch::warpgroup_reg_alloc<MmaRegisterRequirement>();
    // ... MMA + softmax + rescale_o all interleaved in one warpgroup
}
```

FA4 splits the consumer into three specialized roles (softmax-H, softmax-L, correction), enabled by TMEM making the accumulator accessible across warpgroups.

The two softmax warpgroups are explicitly synchronized so their exponential-heavy critical sections don't overlap, ensuring they don't fight for the same MUFU resource.

### TMEM Partitioning Strategy

With 256 KB of TMEM per SM, the allocation must be carefully planned:

**Mandatory:** Two output tiles (O^H, O^L) for the ping-pong schedule. At head dimension 128 with FP32 accumulators, each output tile is 128 × 128 × 4 bytes = 64 KB, so two tiles consume 128 KB — exactly half.

**Remaining 128 KB** can store either:
- **Option A:** 1 S tile (FP32, 64 KB) + 2 P tiles (BF16, 32 KB each) = 128 KB
- **Option B:** 2 S tiles (FP32, 64 KB each) = 128 KB, with P overwriting S in-place

FA4 chooses **Option B**. Why?

- **Immediate compute start.** Two S tiles can be computed back-to-back at pipeline startup, filling the pipeline faster.
- **In-place conversion.** Once S is consumed by softmax and converted to P (which is BF16, half the size), P overwrites S. This is safe because S is no longer needed after softmax.
- **Statistics storage.** The leftover space (since P is smaller than S) is used to store rescale statistics for the correction warpgroup.
- **And two S tiles can be computed at once**, immediately kicking off the pipeline.

<!-- 🔄 CHANGED: Added FA4 TMEM allocation code -->

In the actual code, you can see the TMEM capacity check and 2-CTA tile configuration:

```python
# FA4 (Blackwell) — flash_attn/cute/flash_fwd_sm100.py
# TMEM capacity validation and tile configuration

self.cta_group_size = 2 if self.use_2cta_instrs else 1
# cta_tiler M includes only 1 CTA, scheduler accounts for cluster shape
self.cta_tiler = (self.q_stage * m_block_size, n_block_size, self.head_dim_padded)
# With 2CTA, the MMA tiler M covers both CTAs
self.mma_tiler_qk = (self.cta_group_size * m_block_size, n_block_size, self.head_dim_padded)
self.mma_tiler_pv = (self.cta_group_size * m_block_size, self.head_dim_v_padded, n_block_size)

# q_stage=2 means two Q tiles (ping-pong), each m_block_size=128 rows
# Total TMEM: 2 output tiles (O^H, O^L) + 2 S tiles (overwritten by P)
assert self.tmem_total <= SM100_TMEM_CAPACITY_COLUMNS  # 256 KB check
```

### Register Pressure Management

Here's where things get tight. Each SM has 65,536 registers shared across all threads. With 4 warpgroups × 128 threads = 512 threads per thread block, that's 128 registers per thread.

But each softmax thread needs:
- **128 registers** to hold a full row of input (128 BF16 values, packed 2 per register = 64, but with FP32 intermediates during computation, it's 128)
- **~64 registers** for output staging
- Additional registers for temporaries, loop variables, coefficients

This exceeds the budget. FA4's solution: **staged P storage.** The first 3/4 of P is computed, stored to TMEM (which triggers the corresponding MMA), and registers are freed. Then the remaining 1/4 is computed separately. This trades a small amount of instruction overhead for staying within register limits.

<!-- 🔄 CHANGED: Added code showing split_P_arrive and per-config register tuning -->

```python
# FA4 (Blackwell) — flash_attn/cute/flash_fwd_sm100.py
# Staged P storage: write 3/4 of P first to free registers, then the last 1/4.

self.split_P_arrive = n_block_size // 4 * 3           # = 96 for n_block_size=128
self.split_P_arrive = int(self.split_P_arrive / 32) * 32  # align to 32
assert self.split_P_arrive % 32 == 0
assert self.split_P_arrive < self.n_block_size

# Per-config register tuning — different configs need different register budgets
# Key: (is_causal, use_2cta, head_dim, is_sm103)
SM100_TUNING_CONFIGS = {
    (True, False, 128, False): {
        'ex2_emu_freq': 10,        # emit FMA-emulated exp2 every 10th element
        'ex2_emu_start_frg': 1,    # start emulation from fragment 1
        'num_regs_softmax': 176,   # register budget for softmax warpgroups
        'num_regs_correction': 88  # register budget for correction warpgroup
    },
    (False, True, 128, False): {
        'ex2_emu_freq': 16,
        'ex2_emu_start_frg': 1,
        'num_regs_softmax': 192,
        'num_regs_correction': 72
    },
    # ...
}
```

---

## Fix #2: Software-Emulated Exponential Function

### The Bottleneck

The MUFU (Multi-Function Unit) handles transcendental functions like `exp2(x)`. At 16 ops/cycle/SM, it's 512× slower than the tensor cores (8192 ops/cycle/SM). For attention, every element of the M × N score matrix needs an exponential evaluation during softmax. This makes MUFU a hard bottleneck.

### The Trick: Polynomial Approximation on FMA Units

FA4 implements `2^x` in software using the classical Cody-Waite range reduction + polynomial approximation:

```
2^x = 2^⌊x⌋ × 2^(x - ⌊x⌋)
```

The integer part `2^⌊x⌋` is computed via **bit manipulation** of the IEEE 754 exponent field — essentially a shift-and-add on integer ALU.

The fractional part `2^frac` (where frac ∈ [0, 1)) is approximated by a low-degree polynomial evaluated with **FMA (fused multiply-add) instructions**.

<!-- 🔄 CHANGED: Replaced pseudocode with actual FA4 source code for exp2 emulation -->

Here's the actual implementation — first the polynomial coefficients (computed via the Sollya package to minimize relative error):

```python
# FA4 (Blackwell) — flash_attn/cute/utils.py
# Minimax polynomial coefficients for 2^x on [0, 1)

POLY_EX2 = {
    3: (
        1.0,
        0.695146143436431884765625,     # p1
        0.227564394474029541015625,     # p2
        0.077119089663028717041015625,  # p3
    ),
    # Degree 3 matches hardware MUFU to within 1 BF16 ULP on 99% of inputs
}
```

And the full emulation algorithm — note the range reduction via bit tricks and polynomial evaluation via Horner's method with FMA:

```python
# FA4 (Blackwell) — flash_attn/cute/utils.py

@dsl_user_op
def ex2_emulation(x: Float32, *, poly_degree: int = 3) -> Float32:
    fp32_round_int = float(2**23 + 2**22)
    x_clamped = cute.arch.fmax(x, -127.0)  # Step 1: clamp to avoid underflow

    # Step 2: compute ⌊x⌋ via round-down trick
    x_rounded = add_round_down(x_clamped, fp32_round_int)
    x_rounded_back = x_rounded - fp32_round_int
    x_frac = x_clamped - x_rounded_back  # Step 3: fractional part ∈ [0, 1)

    # Step 4: evaluate polynomial (Horner's method using FMA)
    x_frac_ex2 = evaluate_polynomial(x_frac, POLY_EX2[poly_degree])

    # Step 5: combine via bit manipulation — shift ⌊x⌋ into exponent field
    return combine_int_frac_ex2(x_rounded, x_frac_ex2)
```

<!-- 🔄 CHANGED: Added the actual Horner evaluation and PTX bit manipulation code -->

The Horner evaluation is pure FMA — each step is a single fused multiply-add:

```python
# FA4 (Blackwell) — flash_attn/cute/utils.py

def evaluate_polynomial(x: Float32, poly: Tuple[Float32, ...]) -> Float32:
    deg = len(poly) - 1
    out = poly[deg]                          # Start from highest coefficient
    for i in cutlass.range_constexpr(deg - 1, -1, -1):
        out = out * x + poly[i]              # Each iteration = 1 FMA instruction
    return out
```

And the bit manipulation to combine the integer and fractional parts drops to raw PTX:

```python
# FA4 (Blackwell) — flash_attn/cute/utils.py
# Combines 2^⌊x⌋ (integer) and 2^frac (polynomial result) via bit ops

@dsl_user_op
def combine_int_frac_ex2(x_rounded: Float32, frac_ex2: Float32) -> Float32:
    return cutlass.Float32(llvm.inline_asm(
        T.f32(), [x_rounded.ir_value(), frac_ex2.ir_value()],
        "{\n\t"
        ".reg .s32 x_rounded_i, frac_ex_i, x_rounded_e, out_i;\n\t"
        "mov.b32 x_rounded_i, $1;\n\t"        # reinterpret float as int
        "mov.b32 frac_ex_i, $2;\n\t"
        "shl.b32 x_rounded_e, x_rounded_i, 23;\n\t"  # shift ⌊x⌋ into exponent field
        "add.s32 out_i, x_rounded_e, frac_ex_i;\n\t"  # combine: 2^⌊x⌋ × 2^frac
        "mov.b32 $0, out_i;\n\t"
        "}\n",
        "=f,f,f",
    ))
```

<!-- 🔄 CHANGED: Added comparison with FA3's pure hardware exp2 -->

Compare this entire machinery with FA3's approach — a single hardware instruction:

```cpp
// FA3 (Hopper) — hopper/softmax.h
// Pure hardware MUFU — simple but bottlenecked at 16 ops/cycle/SM
#pragma unroll
for (int mi = 0; mi < size<0>(tensor); ++mi) {
    #pragma unroll
    for (int ni = 0; ni < size<1>(tensor); ++ni) {
        tensor(mi, ni) = exp2f(tensor(mi, ni) * scale - max_scaled);
        //                ^^^^^ Always hardware MUFU. No alternative path.
    }
}
```

### Why Not Emulate Everything?

Full emulation has costs: more registers for polynomial coefficients and intermediates, higher latency per evaluation, and register bandwidth consumption that can cause spills. So FA4 takes a **hybrid approach**:

<!-- ✏️ CHANGED: Added actual FA4 code showing the hybrid MUFU/FMA split decision -->

```python
# FA4 — flash_attn/cute/softmax.py
# Hybrid exp2: some fragments use hardware MUFU, others use FMA emulation
@cute.jit
def apply_exp2_convert(self, acc_S_row, acc_S_row_converted,
                        ex2_emu_freq=0, ex2_emu_res=4, ex2_emu_start_frg=0):
    frg_tile = 32
    frg_cnt = cute.size(acc_S_row) // frg_tile
    acc_S_row_frg = cute.logical_divide(acc_S_row, cute.make_layout(frg_tile))

    for j in cutlass.range_constexpr(frg_cnt):
        for k in cutlass.range_constexpr(0, cute.size(acc_S_row_frg, mode=[0]), 2):
            if cutlass.const_expr(ex2_emu_freq == 0):
                # All hardware MUFU (fallback, or SM103/B300 with doubled MUFU)
                acc_S_row_frg[k, j] = cute.math.exp2(acc_S_row_frg[k, j], fastmath=True)
                acc_S_row_frg[k+1, j] = cute.math.exp2(acc_S_row_frg[k+1, j], fastmath=True)
            else:
                if cutlass.const_expr(
                    k % ex2_emu_freq < ex2_emu_freq - ex2_emu_res
                    or j >= frg_cnt - 1
                    or j < ex2_emu_start_frg
                ):
                    # Hardware MUFU path
                    acc_S_row_frg[k, j] = cute.math.exp2(...)
                    acc_S_row_frg[k+1, j] = cute.math.exp2(...)
                else:
                    # Software FMA emulation path (runs in parallel with MUFU!)
                    acc_S_row_frg[k, j], acc_S_row_frg[k+1, j] = \
                        utils.ex2_emulation_2(acc_S_row_frg[k, j],
                                              acc_S_row_frg[k+1, j])
```

The `ex2_emu_freq` and `ex2_emu_res` parameters control the split. For example, `ex2_emu_freq=10, ex2_emu_res=4` means every 10th pair of elements, 4 are emulated and 6 use hardware. The exact ratios are tuned per configuration.

### Accuracy

The degree-3 polynomial has a max relative error of ~8.8 × 10⁻⁵ in FP32. Sounds bad? It doesn't matter. After rounding to BF16, the quantization error (~3.9 × 10⁻³) completely dominates the polynomial approximation error. The degree-3 polynomial matches hardware MUFU to within 1 BF16 ULP on 99% of inputs. For attention, where softmax outputs are consumed at BF16 precision, this is more than sufficient.

---

## Fix #3: Conditional Softmax Rescaling

### The Problem

Standard online softmax (as used in all FlashAttention versions) maintains running statistics:

```
m_j = max(m_{j-1}, rowmax(S_j))
ℓ_j = exp(m_{j-1} - m_j) · ℓ_{j-1} + rowsum(exp(S_j - m_j))
O_j = exp(m_{j-1} - m_j) · O_{j-1} + exp(S_j - m_j) · V_j
```

The rescaling term `exp(m_{j-1} - m_j) · O_{j-1}` is a full vector multiplication that happens **every iteration**, even when the max barely changes.

### The Fix

<!-- ✏️ CHANGED: Replaced pseudocode with actual FA4 implementation -->

FA4 introduces a threshold τ (set to `log₂(256) = 8.0`). The implementation is embedded directly in the `update_row_max` method we saw earlier:

```python
# FA4 — flash_attn/cute/softmax.py
# Conditional rescaling: skip when max hasn't changed significantly
acc_scale_ = (row_max_old - row_max_safe) * self.scale_log2
acc_scale = cute.math.exp2(acc_scale_, fastmath=True)

if cutlass.const_expr(self.rescale_threshold > 0.0):
    if acc_scale_ >= -self.rescale_threshold:
        # New max is close to old max → skip rescaling
        row_max_new = row_max_old
        row_max_safe = row_max_old
        acc_scale = 1.0  # No-op rescale (multiply by 1)
```

Compare with FA3, where rescaling is **unconditional every iteration**:

```cpp
// FA3 (Hopper) — hopper/softmax.h
// Unconditional rescaling — no threshold check
#pragma unroll
for (int mi = 0; mi < size(row_max); ++mi) {
    float scores_max_cur = !Check_inf
        ? row_max(mi)
        : (row_max(mi) == -INFINITY ? 0.0f : row_max(mi));
    scores_scale(mi) = exp2f((scores_max_prev(mi) - scores_max_cur)
                             * softmax_scale_log2);  // ← Always computed
    row_sum(mi) *= scores_scale(mi);                 // ← Always applied
}
```

When the new max doesn't exceed the old max by more than τ, FA4 skips rescaling entirely. The key insight: **the final normalization step `Output = O_final / ℓ_final` corrects any accumulated drift.** As long as intermediate values don't overflow (which the threshold of 256× prevents), the final result is exact.

In practice, rescaling is needed only in the first few iterations when the running max is still being established. Once it stabilizes, most iterations skip rescaling entirely — saving a vector multiply per iteration.

---

## Fix #4: 2-CTA Backward Pass

The backward pass is where shared memory pressure is most severe. With five MMA operations (vs. two in the forward pass), SMEM traffic exceeds MMA compute by ~30% in the 1-CTA configuration.

### How 2-CTA Helps

In 2-CTA MMA mode, a CTA pair cooperatively executes each MMA with M = 256 (each CTA holds half the M dimension). The critical benefit: **each CTA only stages half of operand B** in shared memory, while the hardware reads the combined B tile from both CTAs. This roughly halves SMEM traffic for operand B across the five backward GEMMs.

The roofline improvement is significant:

| Resource | 1-CTA (M=128) | 2-CTA (M=256) |
|---|---|---|
| MMA compute | 2560 cycles | 2560 cycles |
| Total shared memory | **3328 cycles** | **2688 cycles** |
| Exponential unit | 1024 cycles | 1024 cycles |
| **SMEM overhead vs MMA** | **+30%** | **+5%** |

### The dQ Problem

There's a catch. The dQ computation accumulates along the KV sequence dimension (the outer loop), and its reduction axis is N — which is naturally split across the CTA pair. Each CTA still needs the **full reduction** for its rows.

FA4 solves this using **Distributed Shared Memory (DSMEM)** to exchange half of the dS tile between the two CTAs in the same cluster. After the exchange, each CTA holds an (M/2 × 2N) slice of dS, enabling a CTA-pair UMMA with doubled reduction dimension. The per-CTA dQ MMA becomes:

```
dQ tile shape: (M/2, 2N) × (2N, d) → (M/2, d)
```

This restructuring also **halves the number of global atomic reductions** for dQ, since each CTA writes only half the dQ tile. Atomics are expensive and introduce nondeterminism, so cutting them in half is a double win.

### Deterministic Mode

For reproducible training (critical for RL applications), FA4 provides a deterministic execution mode using semaphore-based serialization of global reductions. The performance overhead is minimized through careful CTA scheduling:
- Batches processed as the outermost dimension
- Heads swizzled within L2 cache capacity
- For causal masking: KV blocks launched in descending order, query blocks in ascending order from the diagonal, dQ reductions ordered by descending query block index ("shortest-processing-time-first")

This gets the deterministic backward pass to ~75% the speed of the nondeterministic 1-CTA version.

---

## Fix #5: LPT Scheduling

Load imbalance is inherent in attention — causal masking means tiles near the diagonal do more work than tiles far from it; variable-length batches have different sequence lengths.

<!-- ✏️ CHANGED: Added actual FA4 LPT scheduling code -->

FA4 applies **Longest-Processing-Time-First (LPT) scheduling**, a classical result from parallel processing theory. The implementation is clean — just reverse the block order:

```python
# FA4 — flash_attn/cute/tile_scheduler.py
# LPT: reverse block order so heaviest tiles (near diagonal) run first
@cute.jit
def get_current_work(self) -> WorkTileInfo:
    # ... L2-swizzled coordinate mapping ...

    # Longest-processing-time-first: one line does the trick
    if const_expr(params.lpt):
        block = params.num_block - 1 - block  # ← Simply reverse!

    is_valid = self._tile_idx < params.total_blocks
    return WorkTileInfo(
        (Int32(block), Int32(head_idx), Int32(batch_idx), Int32(self._split_idx)),
        is_valid
    )
```

But the real insight is in how it interacts with L2 cache locality. The scheduler divides heads into sections that fit in L2 cache, and traverses heads-per-section → mblocks (reversed) → sections → batches:

```python
# FA4 — flash_attn/cute/tile_scheduler.py
# L2-aware head swizzling: fit as many KV heads as possible in L2
size_one_kv_head = args.seqlen_k * (args.headdim + args.headdim_v) * args.element_size
size_l2 = 50 * 1024 * 1024  # 40 MB budget for K & V

# swizzle = how many heads fit in L2 (rounded to power of 2 for fast divmod)
log2_floor = lambda n: 31 - clz(n)
swizzle = 1 if size_l2 < size_one_head else (1 << log2_floor(size_l2 // size_one_head))
```

For GQA/MQA, all query heads per KV head are traversed before varying over mblocks — ensuring maximum KV reuse from L2 cache.

Empirical gains: **4–8% FLOPS improvement for MHA, 7–14% for MQA** on H200 (this optimization is architecture-agnostic and also benefits FA3 on Hopper).

---

## The Framework: CuTe-DSL

<!-- ✏️ CHANGED: Expanded CuTe-DSL section with full code comparison -->

Perhaps the most practically impactful contribution for the broader ecosystem: FA4 is implemented **entirely in CuTe-DSL embedded in Python** — no C++ template metaprogramming whatsoever.

### C++ Templates vs. Python DSL: A Side-by-Side

The same conceptual operation — computing softmax rescaling — looks fundamentally different in the two frameworks:

```cpp
// FA3 (C++ CUTLASS templates) — hopper/softmax.h
// 30+ lines of template metaprogramming
template<bool Is_first, bool Check_inf=false, typename Tensor0>
__forceinline__ __device__ TensorT max_get_scale(Tensor0 &acc_s) {
    Tensor scores = make_tensor(acc_s.data(),
        flash::convert_layout_acc_rowcol(acc_s.layout()));
    static_assert(CUTE_STATIC_V(size<0>(scores)) == kNRows);
    TensorT scores_scale;
    if constexpr (Is_first) {
        flash::template reduce_max</*zero_init=*/true>(scores, row_max);
        cute::fill(scores_scale, 1.f);
    } else {
        Tensor scores_max_prev = make_fragment_like(row_max);
        cute::copy(row_max, scores_max_prev);
        flash::template reduce_max</*zero_init=*/false>(scores, row_max);
        #pragma unroll
        for (int mi = 0; mi < size(row_max); ++mi) {
            float scores_max_cur = !Check_inf
                ? row_max(mi)
                : (row_max(mi) == -INFINITY ? 0.0f : row_max(mi));
            scores_scale(mi) = exp2f((scores_max_prev(mi) - scores_max_cur)
                                     * softmax_scale_log2);
            row_sum(mi) *= scores_scale(mi);
        }
    }
    return scores_scale;
};
```

```python
# FA4 (Python CuTe-DSL) — flash_attn/cute/softmax.py
# Same algorithm, same low-level control, readable Python
@cute.jit
def update_row_max(self, acc_S_row: cute.TensorSSA, is_first: int):
    if cutlass.const_expr(is_first):
        row_max_new = self._compute_row_max(acc_S_row)
        row_max_safe = row_max_new if row_max_new != -cutlass.Float32.inf else 0.0
        acc_scale = 0.0
    else:
        row_max_old = self.row_max[0]
        row_max_new = self._compute_row_max(acc_S_row, init_val=row_max_old)
        row_max_safe = row_max_new if row_max_new != -cutlass.Float32.inf else 0.0
        acc_scale_ = (row_max_old - row_max_safe) * self.scale_log2
        acc_scale = cute.math.exp2(acc_scale_, fastmath=True)
        if cutlass.const_expr(self.rescale_threshold > 0.0):
            if acc_scale_ >= -self.rescale_threshold:
                row_max_new = row_max_old
                row_max_safe = row_max_old
                acc_scale = 1.0
    self.row_max[0] = row_max_new
    return row_max_safe, acc_scale
```

Same algorithm, same low-level control (both compile down to PTX), but the Python version is more readable, doesn't require `template<bool Is_first, bool Check_inf=false, typename Tensor0>` incantations, and compiles in seconds instead of minutes. When CuTe-DSL's APIs don't cover something, raw PTX is available as an escape hatch — as we saw in the `combine_int_frac_ex2` function.

### Compile Time

| | Forward | Backward |
|---|---|---|
| FA3 (C++ templates) | 55s | 45s |
| FA4 (CuTe-DSL) | 2.5s | 1.4s |
| **Speedup** | **22×** | **32×** |

And FA3 required precompiling **hundreds of kernels** for different attention variants. FA4's JIT compilation means you compile only what you need, when you need it.

### Practical Impact

The practical impact is already visible: developers have built **FlexAttention and block-sparse attention variants** on top of FA4's framework without modifying core code. The barrier to entry drops from "years of C++ template metaprogramming expertise" to "a few months of GPU programming experience."

The design philosophy is modular: block-sparse patterns, masking strategies, variable-length handling, and work scheduling are all orthogonal, composable primitives. New attention variants get all existing optimizations for free.

---

## Results

On B200 with BF16, head dimension 128:

- **Forward pass:** 1.1–1.3× faster than cuDNN 9.13, 2.1–2.7× faster than Triton. Peak: **~1600 TFLOPS** (71% of theoretical max).
- **Backward pass:** Consistent speedups across sequence lengths for both causal and non-causal.
- **Deterministic backward:** Up to 75% the speed of the non-deterministic 1-CTA backward — a practical option for RL training that demands reproducibility.

Notably, since FA4's release, the cuDNN team has incorporated many of FA4's techniques into cuDNN 9.14+, converging to similar performance. This is perhaps the strongest validation of the algorithmic contributions.

---

## Key Takeaway

FlashAttention-4 is a case study in what happens when you take hardware asymmetry seriously. The tensor cores got faster, so the bottleneck moved. Rather than hoping the bottleneck would go away, the authors identified exactly where the cycles were being spent (roofline analysis with concrete cycle counts for every resource), then attacked each bottleneck with a targeted fix:

- SMEM too slow → 2-CTA MMA to halve operand B traffic
- Exponential unit too slow → Polynomial emulation on FMA units
- Rescaling wastes cycles → Conditional rescaling with deferred correction
- Pipeline bubbles → Ping-pong schedule with TMEM-based decoupling
- C++ compile times too slow → CuTe-DSL in Python

The code is open source at [github.com/Dao-AILab/flash-attention](https://github.com/Dao-AILab/flash-attention/tree/main/flash_attn/cute).

*The B300/GB300 will double MUFU throughput to 32 ops/cycle/SM — and FA4 already handles this: the tuning config sets `ex2_emu_freq: 0` for SM103, disabling emulation entirely. When the bottleneck map shifts again, the framework is ready.*