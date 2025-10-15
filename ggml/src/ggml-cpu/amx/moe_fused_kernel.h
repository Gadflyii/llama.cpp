#pragma once

#include "amx.h"
#include "mmq.h"
#include <immintrin.h>

// Macro for packed B matrix indexing (also defined in mmq.cpp)
#ifndef PACKED_INDEX
#define PACKED_INDEX(n, k, KB, tile_size) (n * KB + k) * tile_size
#endif

// AMX Native Fused Gate+Up+SiLU Kernel for MoE
// Based on SGlang's tinygemm_kernel_nn2 pattern:
//   - Single KB loop (load A tiles once)
//   - Dual accumulators (gate and up)
//   - Inline SiLU fusion
//
// Reference: /home/ron/src/sglang/sgl-kernel/csrc/cpu/moe.cpp:266-367

namespace {

// SiLU activation using AVX-512
// silu(x) = x / (1 + exp(-x)) = x * sigmoid(x)
static inline void silu_and_mul_avx512(
    const float* __restrict__ gate,
    const float* __restrict__ up,
    float* __restrict__ out,
    int n
) {
#if defined(__AVX512F__)
    const __m512 one = _mm512_set1_ps(1.0f);
    const __m512 neg_one = _mm512_set1_ps(-1.0f);

    int i = 0;
    // Process 16 elements at a time
    for (; i + 16 <= n; i += 16) {
        __m512 g = _mm512_loadu_ps(gate + i);
        __m512 u = _mm512_loadu_ps(up + i);

        // SiLU(g) = g / (1 + exp(-g))
        __m512 neg_g = _mm512_mul_ps(g, neg_one);
        __m512 exp_neg = _mm512_exp_ps(neg_g);
        __m512 silu_g = _mm512_div_ps(g, _mm512_add_ps(one, exp_neg));

        // result = silu(g) * u
        __m512 result = _mm512_mul_ps(silu_g, u);
        _mm512_storeu_ps(out + i, result);
    }

    // Handle remainder
    for (; i < n; ++i) {
        float g = gate[i];
        float silu_g = g / (1.0f + expf(-g));
        out[i] = silu_g * up[i];
    }
#else
    // Fallback for non-AVX512
    for (int i = 0; i < n; ++i) {
        float g = gate[i];
        float silu_g = g / (1.0f + expf(-g));
        out[i] = silu_g * up[i];
    }
#endif
}

}  // namespace

// AMX native fused gate+up+silu kernel for M=1 decode
// Processes gate and up projections in a single pass with inline SiLU fusion
// Note: type is the block type (e.g., block_q4_0), not ggml_type enum
template<typename vec_dot_type, typename type>
static void amx_fused_gate_up_silu_kernel_m1(
    int N,                     // Intermediate size (e.g., 768)
    int KB,                    // K blocks after quantization
    const char* A,             // Input: quantized [1, K]
    const void* B_gate,        // Gate weights: packed expert weights
    const void* B_up,          // Up weights: packed expert weights
    float* C_intermediate,     // Output: [1, N] - already SiLU fused!
    int64_t TILE_SIZE          // Packed tile size
) {
    constexpr int TILE_M = 1;   // M=1 for decode
    constexpr int TILE_N = 16;  // AMX tile width
    constexpr int TILE_K = 64;  // AMX tile K dimension
    constexpr int VNNI_BLK = 4; // VNNI block size for int8

    constexpr bool need_unpack = do_unpack<type>::value;
    const int prefetch_distance = get_prefetch_distance();

    const vec_dot_type* RESTRICT A_typed = reinterpret_cast<const vec_dot_type*>(A);
    const int lda = KB * sizeof(vec_dot_type);

    // Temporary buffers for unpacking
    static thread_local char tile_buf[TILE_N * TILE_K * 16] __attribute__((aligned(64)));

    // Process N in blocks of TILE_N
    const int NB_count = N / TILE_N;

    for (int nb = 0; nb < NB_count; ++nb) {
        const int nb_start = nb * TILE_N;

        // Temporary buffers for this N block
        float gate_tmp[TILE_N] __attribute__((aligned(64)));
        float up_tmp[TILE_N] __attribute__((aligned(64)));

        // Zero accumulators
        static thread_local int32_t acc_gate[TILE_M * TILE_N] __attribute__((aligned(64)));
        static thread_local int32_t acc_up[TILE_M * TILE_N] __attribute__((aligned(64)));
        memset(acc_gate, 0, sizeof(acc_gate));
        memset(acc_up, 0, sizeof(acc_up));

        // Get pointers to B for this N block
        // Assumes B is packed in [N/TILE_N, KB, TILE_N] format
        const char* B_gate_nb = (const char*)B_gate + PACKED_INDEX(nb, 0, KB, TILE_SIZE);
        const char* B_up_nb = (const char*)B_up + PACKED_INDEX(nb, 0, KB, TILE_SIZE);

        // ===== SINGLE KB LOOP - CRITICAL OPTIMIZATION =====
        // Load A tiles ONCE, use for both gate and up
        for (int k = 0; k < KB; ++k) {
            // Prefetch ahead
            if (prefetch_distance > 0 && k + prefetch_distance < KB) {
                _mm_prefetch(B_gate_nb + PACKED_INDEX(0, k + prefetch_distance, KB, TILE_SIZE), _MM_HINT_T0);
                _mm_prefetch(B_up_nb + PACKED_INDEX(0, k + prefetch_distance, KB, TILE_SIZE), _MM_HINT_T0);
                _mm_prefetch((const char*)&A_typed[k + prefetch_distance].qs, _MM_HINT_T0);
            }

            // Load A tile ONCE
            _tile_loadd(TMM2, A_typed[k].qs, lda);

            // ========== GATE PROJECTION ==========
            const char* B_gate_k = B_gate_nb + PACKED_INDEX(0, k, KB, TILE_SIZE);

            if (need_unpack) {
                unpack_B<type>(tile_buf, B_gate_k);
                _tile_loadd(TMM0, tile_buf, TILE_N * VNNI_BLK);
            } else {
                _tile_loadd(TMM0, B_gate_k, TILE_N * VNNI_BLK);
            }

            // Accumulate gate (using TMM2)
            _tile_dpbssd(TMM4, TMM2, TMM0);

            // ========== UP PROJECTION (reuse TMM2!) ==========
            const char* B_up_k = B_up_nb + PACKED_INDEX(0, k, KB, TILE_SIZE);

            if (need_unpack) {
                unpack_B<type>(tile_buf, B_up_k);
                _tile_loadd(TMM0, tile_buf, TILE_N * VNNI_BLK);
            } else {
                _tile_loadd(TMM0, B_up_k, TILE_N * VNNI_BLK);
            }

            // Accumulate up (STILL using TMM2!)
            _tile_dpbssd(TMM6, TMM2, TMM0);
        }

        // Store accumulators
        _tile_stored(TMM4, acc_gate, TILE_N * sizeof(int32_t));
        _tile_stored(TMM6, acc_up, TILE_N * sizeof(int32_t));

        // Dequantize gate and up using existing acc_C template
        // Note: We pass K=KB and proper pointers for scale calculation
        acc_C<vec_dot_type, type, false>::apply(
            gate_tmp, TILE_N, acc_gate, A_typed, KB, B_gate_nb, TILE_M);
        acc_C<vec_dot_type, type, false>::apply(
            up_tmp, TILE_N, acc_up, A_typed, KB, B_up_nb, TILE_M);

        // ========== INLINE SILU FUSION ==========
        // Apply silu(gate) * up → intermediate
        silu_and_mul_avx512(gate_tmp, up_tmp, C_intermediate + nb_start, TILE_N);
    }

    // Handle remainder if N is not a multiple of TILE_N
    const int remainder = N % TILE_N;
    if (remainder > 0) {
        // For simplicity, fall back to standard path for remainder
        // In practice, N is typically a multiple of TILE_N for MoE
        fprintf(stderr, "[WARNING] N=%d not multiple of TILE_N=%d, remainder not implemented\n",
                N, TILE_N);
    }
}

#endif  // AMX_MOE_FUSED_KERNEL_H
