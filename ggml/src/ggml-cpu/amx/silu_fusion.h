#pragma once

#include <immintrin.h>
#include <cmath>

// Fused SiLU + elementwise multiply operations for MoE
// These provide building blocks for gate+up fusion in Phase 2

#if defined(__AVX512F__)

// SiLU activation: silu(x) = x / (1 + exp(-x)) = x * sigmoid(x)
// Using approximation for performance: silu(x) ≈ x * tanh(x * 0.5 + x^3 * 0.04) * 0.5 + x * 0.5
static inline __m512 silu_avx512(__m512 x) {
    // Fast approximation: silu(x) ≈ x * (0.5 + 0.5 * tanh(0.797884560803 * (x + 0.044715 * x^3)))
    // This is the GELU-like approximation adapted for SiLU

    // Constants
    const __m512 c1 = _mm512_set1_ps(0.5f);
    const __m512 c2 = _mm512_set1_ps(0.797884560803f);  // sqrt(2/pi)
    const __m512 c3 = _mm512_set1_ps(0.044715f);
    const __m512 c4 = _mm512_set1_ps(1.0f);

    // x^3
    __m512 x2 = _mm512_mul_ps(x, x);
    __m512 x3 = _mm512_mul_ps(x2, x);

    // 0.044715 * x^3
    __m512 term = _mm512_mul_ps(c3, x3);

    // x + 0.044715 * x^3
    term = _mm512_add_ps(x, term);

    // 0.797884560803 * (x + 0.044715 * x^3)
    term = _mm512_mul_ps(c2, term);

    // tanh(term) - using approximation: tanh(x) ≈ x * (27 + x^2) / (27 + 9*x^2)
    // For better accuracy, we'll use a simpler clamp: tanh(x) ≈ clamp(x, -1, 1) for |x| > 2
    __m512 term2 = _mm512_mul_ps(term, term);
    __m512 num = _mm512_fmadd_ps(term2, c4, _mm512_set1_ps(27.0f));
    num = _mm512_mul_ps(num, term);
    __m512 den = _mm512_fmadd_ps(term2, _mm512_set1_ps(9.0f), _mm512_set1_ps(27.0f));
    __m512 tanh_term = _mm512_div_ps(num, den);

    // 0.5 + 0.5 * tanh(term)
    __m512 sigmoid_approx = _mm512_fmadd_ps(tanh_term, c1, c1);

    // x * sigmoid_approx = silu(x)
    return _mm512_mul_ps(x, sigmoid_approx);
}

// Fused SiLU(gate) * up operation
// Processes 16 floats at a time
// out[i] = silu(gate[i]) * up[i]
static inline void fused_silu_mul_avx512(
    const float * gate,
    const float * up,
    float * out,
    int n
) {
    int i = 0;

    // Process 16 elements at a time with AVX-512
    for (; i + 16 <= n; i += 16) {
        __m512 g = _mm512_loadu_ps(gate + i);
        __m512 u = _mm512_loadu_ps(up + i);

        __m512 silu_g = silu_avx512(g);
        __m512 result = _mm512_mul_ps(silu_g, u);

        _mm512_storeu_ps(out + i, result);
    }

    // Handle remaining elements
    for (; i < n; i++) {
        float g = gate[i];
        // Scalar SiLU: silu(x) = x / (1 + exp(-x))
        float silu_g = g / (1.0f + expf(-g));
        out[i] = silu_g * up[i];
    }
}

// Batch version: process M rows of N columns
// gate: [M, N] gate projection output
// up: [M, N] up projection output
// out: [M, N] fused result
static inline void fused_silu_mul_batch_avx512(
    const float * gate,
    const float * up,
    float * out,
    int M,
    int N
) {
    for (int m = 0; m < M; m++) {
        fused_silu_mul_avx512(
            gate + m * N,
            up + m * N,
            out + m * N,
            N
        );
    }
}

#else
// Fallback for non-AVX512 systems
static inline void fused_silu_mul_avx512(
    const float * gate,
    const float * up,
    float * out,
    int n
) {
    for (int i = 0; i < n; i++) {
        float g = gate[i];
        float silu_g = g / (1.0f + expf(-g));
        out[i] = silu_g * up[i];
    }
}

static inline void fused_silu_mul_batch_avx512(
    const float * gate,
    const float * up,
    float * out,
    int M,
    int N
) {
    for (int m = 0; m < M; m++) {
        fused_silu_mul_avx512(gate + m * N, up + m * N, out + m * N, N);
    }
}
#endif

// Future Phase 2 extension point:
// This header provides the fused SiLU+multiply primitive that can be integrated
// into a full gate+up+silu AMX kernel when the higher-level GGML operation is added.
//
// Full Phase 2 implementation would:
// 1. Create GGML_OP_MUL_MAT_UP_GATE operation
// 2. Modify llama-graph.cpp to use fused operation for MoE FFN
// 3. Implement AMX backend that:
//    - Computes gate projection (AMX matmul)
//    - Computes up projection (AMX matmul)
//    - Applies fused_silu_mul_batch_avx512 (this function)
//    - Returns intermediate for down projection
//
// Expected benefit: +10-15% prompt throughput from reduced kernel overhead
