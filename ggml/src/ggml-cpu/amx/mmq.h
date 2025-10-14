#pragma once
#include "common.h"

#ifdef __cplusplus
extern "C" {
#endif

// MoE token mapping structure (matches definition in ggml-cpu.c)
struct mmid_row_mapping {
    int32_t i1;  // Token ID (expert selection index)
    int32_t i2;  // Batch index
};

size_t ggml_backend_amx_desired_wsize(const struct ggml_tensor * dst);

size_t ggml_backend_amx_get_alloc_size(const struct ggml_tensor * tensor);

void ggml_backend_amx_convert_weight(struct ggml_tensor * tensor, const void * data, size_t offset, size_t size);

void ggml_backend_amx_mul_mat(const struct ggml_compute_params * params, struct ggml_tensor * dst);

void ggml_backend_amx_mul_mat_moe_expert(
    const struct ggml_compute_params * params,
    struct ggml_tensor * dst,
    const struct ggml_tensor * src0,
    const struct ggml_tensor * src1,
    const struct ggml_tensor * ids,
    const int64_t expert_id,
    const struct mmid_row_mapping * token_mappings,
    const int64_t num_tokens,
    const char * expert_weights,
    const void * wdata,
    const size_t row_size);

void ggml_backend_amx_mul_mat_moe_batch(
    const struct ggml_compute_params * params,
    struct ggml_tensor * dst,
    const struct ggml_tensor * src0,
    const struct ggml_tensor * src1,
    const struct ggml_tensor * ids,
    const int * activated_experts,
    const int activated_count,
    const struct mmid_row_mapping * matrix_rows,
    const int64_t * matrix_row_counts,
    const void * wdata,
    const size_t row_size,
    const int64_t ne10,
    const int64_t nb02);

// NUMA weight replication for CPU_REPACK backend
// Called after weights are repacked to replicate across NUMA groups
void ggml_backend_amx_numa_replicate_expert(int64_t expert_id, const void * data, size_t size);

// Check if NUMA weight replication is enabled
bool ggml_backend_amx_numa_is_enabled();

// Get NUMA-aware expert weight pointer for current thread
// Returns socket-local pointer if NUMA is enabled, otherwise returns fallback_ptr
const void * ggml_backend_amx_numa_get_expert_weight(int64_t expert_id, const void * fallback_ptr);

#ifdef __cplusplus
}
#endif
