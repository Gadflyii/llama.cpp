#include "amx.h"
#include "common.h"
#include "mmq.h"
#include "ggml-backend-impl.h"
#include "ggml-backend.h"
#include "ggml-impl.h"
#include "ggml-cpu.h"
#include "traits.h"

#if defined(__linux__)
#include <sys/syscall.h>
#include <unistd.h>
#endif

#include <cstdlib>
#include <cstring>
#include <memory>

#if defined(__AMX_INT8__) && defined(__AVX512VNNI__)

// AMX type_trais
namespace ggml::cpu::amx {
class tensor_traits : public ggml::cpu::tensor_traits {
    bool work_size(int /* n_threads */, const struct ggml_tensor * op, size_t & size) override {
        size = ggml_backend_amx_desired_wsize(op);
        return true;
    }

    bool compute_forward(struct ggml_compute_params * params, struct ggml_tensor * op) override {
        if (op->op == GGML_OP_MUL_MAT) {
            ggml_backend_amx_mul_mat(params, op);
            return true;
        }
        // MUL_MAT_GATE_UP_SILU handled by repack backend's dispatch
        // which checks buffer types and calls ggml_backend_amx_mul_mat_gate_up_silu_fused
        if (op->op == GGML_OP_MUL_MAT_GATE_UP_SILU) {
            // Delegate to repack backend - it will check if AMX path should be used
            // and call the appropriate implementation
            return false;  // Let repack backend handle it
        }
        return false;
    }
};

static ggml::cpu::tensor_traits * get_tensor_traits(ggml_backend_buffer_t, struct ggml_tensor *) {
    static tensor_traits traits;
    return &traits;
}
}  // namespace ggml::cpu::amx

// AMX buffer interface
static void ggml_backend_amx_buffer_free_buffer(ggml_backend_buffer_t buffer) {
    free(buffer->context);
}

static void * ggml_backend_amx_buffer_get_base(ggml_backend_buffer_t buffer) {
    return (void *) (buffer->context);
}

static enum ggml_status ggml_backend_amx_buffer_init_tensor(ggml_backend_buffer_t buffer, struct ggml_tensor * tensor) {
    // For MOE and FUSED_MOE gate/up expert weights, DON'T set AMX traits
    // This allows the repack backend to handle dispatch for MUL_MAT_GATE_UP_SILU
    // The repack backend will check buffer types and call AMX implementation (or baseline for M<=2)
    const enum ggml_amx_moe_arch arch = ggml_get_amx_moe_arch();
    if (arch == GGML_AMX_MOE_ARCH_MOE || arch == GGML_AMX_MOE_ARCH_FUSED_MOE) {
        const char * name = tensor->name;
        if ((strstr(name, ".ffn_gate_exps.weight") != nullptr) ||
            (strstr(name, ".ffn_up_exps.weight") != nullptr)) {
            // Don't set AMX traits for these tensors
            tensor->extra = nullptr;
            return GGML_STATUS_SUCCESS;
        }
    }

    tensor->extra = (void *) ggml::cpu::amx::get_tensor_traits(buffer, tensor);

    GGML_UNUSED(buffer);
    return GGML_STATUS_SUCCESS;
}

static void ggml_backend_amx_buffer_memset_tensor(ggml_backend_buffer_t buffer, struct ggml_tensor * tensor,
                                                  uint8_t value, size_t offset, size_t size) {
    memset((char *) tensor->data + offset, value, size);

    GGML_UNUSED(buffer);
}

// Forward declare mirror buffer structure for NUMA replication
#if defined(__linux__)
#define GGML_NUMA_MAX_NODES 8
#define GGML_MIRROR_BUFFER_MAGIC 0x4D49524E  // "MIRN" in hex
struct ggml_numa_mirror_buffer {
    uint32_t magic;                             // magic number for reliable identification
    uint32_t n_replicas;
    uint32_t active_nodes[GGML_NUMA_MAX_NODES];
    void *   replicas[GGML_NUMA_MAX_NODES];
    size_t   size;
    void *   original_base;
    bool     read_only;
};
#endif

static void ggml_backend_amx_buffer_set_tensor(ggml_backend_buffer_t buffer, struct ggml_tensor * tensor,
                                               const void * data, size_t offset, size_t size) {
#if defined(__linux__)
    // Check if this is a mirror buffer using magic number
    if (buffer->context) {
        struct ggml_numa_mirror_buffer * mirror = (struct ggml_numa_mirror_buffer *) buffer->context;
        // Verify magic number to ensure this is actually a mirror buffer
        // Only replicate if this is a read-only buffer (model weights)
        if (mirror->magic == GGML_MIRROR_BUFFER_MAGIC && mirror->n_replicas > 1 && mirror->read_only) {
            // This is a read-only mirror buffer - replicate to all nodes
            // Get the buffer base which is the first replica
            void * buffer_base = mirror->original_base ? mirror->original_base : mirror->replicas[mirror->active_nodes[0]];
            size_t tensor_offset = (char *)tensor->data - (char *)buffer_base;

            for (uint32_t i = 0; i < mirror->n_replicas; i++) {
                uint32_t node = mirror->active_nodes[i];
                // Temporarily update tensor->data to point to this replica
                void * original_data = tensor->data;
                tensor->data = (char *)mirror->replicas[node] + tensor_offset;

                // Convert/copy to this replica
                if (qtype_has_amx_kernels(tensor->type)) {
                    GGML_LOG_DEBUG("%s: amx repack tensor %s to node %u\n", __func__, tensor->name, node);
                    ggml_backend_amx_convert_weight(tensor, data, offset, size);
                } else {
                    memcpy((char *) tensor->data + offset, data, size);
                }

                // Restore original pointer
                tensor->data = original_data;
            }
            return;
        }
    }
#endif

    // Regular (non-mirror) buffer
    if (qtype_has_amx_kernels(tensor->type)) {
        GGML_LOG_DEBUG("%s: amx repack tensor %s of type %s\n", __func__, tensor->name, ggml_type_name(tensor->type));
        ggml_backend_amx_convert_weight(tensor, data, offset, size);
    } else {
        memcpy((char *) tensor->data + offset, data, size);
    }

    GGML_UNUSED(buffer);
}

/*
// need to figure what we need to do with buffer->extra.
static void ggml_backend_amx_buffer_get_tensor(ggml_backend_buffer_t buffer, const struct ggml_tensor * tensor, void * data, size_t offset, size_t size) {
    GGML_ASSERT(!qtype_has_amx_kernels(tensor->type));
    memcpy(data, (const char *)tensor->data + offset, size);

    GGML_UNUSED(buffer);
}

static bool ggml_backend_amx_buffer_cpy_tensor(ggml_backend_buffer_t buffer, const struct ggml_tensor * src, struct ggml_tensor * dst) {
    if (ggml_backend_buffer_is_host(src->buffer)) {
        if (qtype_has_amx_kernels(src->type)) {
            ggml_backend_amx_convert_weight(dst, src->data, 0, ggml_nbytes(dst));
        } else {
            memcpy(dst->data, src->data, ggml_nbytes(src));
        }
        return true;
    }
    return false;

    GGML_UNUSED(buffer);
}
*/

static void ggml_backend_amx_buffer_clear(ggml_backend_buffer_t buffer, uint8_t value) {
    memset(buffer->context, value, buffer->size);
}

static ggml_backend_buffer_i ggml_backend_amx_buffer_interface = {
    /* .free_buffer     = */ ggml_backend_amx_buffer_free_buffer,
    /* .get_base        = */ ggml_backend_amx_buffer_get_base,
    /* .init_tensor     = */ ggml_backend_amx_buffer_init_tensor,
    /* .memset_tensor   = */ ggml_backend_amx_buffer_memset_tensor,
    /* .set_tensor      = */ ggml_backend_amx_buffer_set_tensor,
    /* .get_tensor      = */ nullptr,
    /* .cpy_tensor      = */ nullptr,
    /* .clear           = */ ggml_backend_amx_buffer_clear,
    /* .reset           = */ nullptr,
};

static const char * ggml_backend_amx_buffer_type_get_name(ggml_backend_buffer_type_t buft) {
    return "AMX";

    GGML_UNUSED(buft);
}

#if defined(__linux__)
// Forward declaration for NUMA mirror support
extern "C" enum ggml_numa_strategy ggml_get_numa_strategy(void);
#endif

static ggml_backend_buffer_t ggml_backend_amx_buffer_type_alloc_buffer(ggml_backend_buffer_type_t buft, size_t size) {
#if defined(__linux__)
    // Check if NUMA mirror mode is active at allocation time
    if (ggml_get_numa_strategy() == GGML_NUMA_STRATEGY_MIRROR) {
        // Need to allocate mirror buffer and wrap it with AMX interface
        // Delegate to CPU buffer type which will create mirror
        ggml_backend_buffer_t buffer = ggml_backend_buft_alloc_buffer(ggml_backend_cpu_buffer_type(), size);

        if (buffer == nullptr) {
            return nullptr;
        }

        // Override the buffer type to report as AMX
        buffer->buft = buft;

        // CRITICAL: Override init_tensor to set AMX traits so AMX kernels are used
        // Without this, tensors won't have AMX traits and will fall back to regular CPU ops
        buffer->iface.init_tensor = ggml_backend_amx_buffer_init_tensor;
        buffer->iface.set_tensor = ggml_backend_amx_buffer_set_tensor;

        return buffer;
    }
#endif

    void * data = ggml_aligned_malloc(size);
    if (data == NULL) {
        fprintf(stderr, "%s: failed to allocate buffer of size %zu\n", __func__, size);
        return NULL;
    }

    return ggml_backend_buffer_init(buft, ggml_backend_amx_buffer_interface, data, size);
}

static size_t ggml_backend_amx_buffer_type_get_alignment(ggml_backend_buffer_type_t buft) {
    return TENSOR_ALIGNMENT;

    GGML_UNUSED(buft);
}

namespace ggml::cpu::amx {
class extra_buffer_type : ggml::cpu::extra_buffer_type {
    // Helper: Check if tensor is an MoE expert gate or up weight by name pattern
    static bool is_moe_expert_gate_or_up_weight(const struct ggml_tensor * tensor) {
        if (!tensor || !tensor->name) {
            return false;
        }

        const char * name = tensor->name;

        // Pattern for PACKED expert weights (all experts in one tensor):
        // - blk.X.ffn_gate_exps.weight → Gate weights for ALL experts
        // - blk.X.ffn_up_exps.weight   → Up weights for ALL experts
        //
        // Pattern for INDIVIDUAL expert weights (one tensor per expert):
        // - blk.X.ffn.experts.Y.gate_proj.weight
        // - blk.X.ffn.experts.Y.up_proj.weight

        // Check packed expert weights (most common for fused_moe)
        if (strstr(name, ".ffn_gate_exps.weight") != nullptr ||
            strstr(name, ".ffn_up_exps.weight") != nullptr) {
            return true;
        }

        // Check individual expert weights (alternative format)
        bool is_expert_weight = (strstr(name, ".ffn.experts.") != nullptr) ||
                                (strstr(name, ".ffn_experts.") != nullptr);
        bool is_gate_or_up = (strstr(name, ".gate_proj.weight") != nullptr) ||
                             (strstr(name, ".up_proj.weight") != nullptr);

        return is_expert_weight && is_gate_or_up;
    }

    bool supports_op(ggml_backend_dev_t, const struct ggml_tensor * op) override {
        // Only FUSED_MOE uses AMX buffer for MoE expert weights
        // BASE and MOE keep expert weights in CPU_REPACK buffer
        const enum ggml_amx_moe_arch arch = ggml_get_amx_moe_arch();
        if (arch == GGML_AMX_MOE_ARCH_FUSED_MOE) {
            // Check src[0] (weight tensor) - during model loading, dummy ops are created to test buffer support
            if (op->src[0]) {
                if (is_moe_expert_gate_or_up_weight(op->src[0])) {
                    if (qtype_has_amx_kernels(op->src[0]->type)) {
                        static bool first_detection = true;
                        if (first_detection) {
                            fprintf(stderr, "[AMX BUFFER] ✓ Detected MoE expert gate/up weight: %s\n", op->src[0]->name);
                            fprintf(stderr, "[AMX BUFFER] ✓ Selecting AMX buffer for fused_moe architecture\n");
                            first_detection = false;
                        }
                        return true;  // Use AMX buffer for this weight
                    }
                }
            }
        }

        // handle only 2d gemm for now
        auto is_contiguous_2d = [](const struct ggml_tensor * t) {
            return ggml_is_contiguous(t) && t->ne[3] == 1 && t->ne[2] == 1;
        };

        if (op->op == GGML_OP_MUL_MAT && is_contiguous_2d(op->src[0]) &&  // src0 must be contiguous
            is_contiguous_2d(op->src[1]) &&                               // src1 must be contiguous
            op->src[0]->buffer && op->src[0]->buffer->buft == ggml_backend_amx_buffer_type() &&
            op->src[0]->ne[0] % (TILE_K * 2 * 32) == 0 && // TODO: not sure if correct (https://github.com/ggml-org/llama.cpp/pull/16315)
            op->ne[0] % (TILE_N * 2) == 0 &&                              // out_features is 32x
            (qtype_has_amx_kernels(op->src[0]->type) || (op->src[0]->type == GGML_TYPE_F16))) {
            // src1 must be host buffer
            if (op->src[1]->buffer && !ggml_backend_buft_is_host(op->src[1]->buffer->buft)) {
                return false;
            }
            // src1 must be float32
            if (op->src[1]->type == GGML_TYPE_F32) {
                return true;
            }
        }

        // Support MUL_MAT_GATE_UP_SILU for buffer selection only (not runtime dispatch)
        // At runtime, repack backend handles dispatch to AMX or baseline
        if (op->op == GGML_OP_MUL_MAT_GATE_UP_SILU) {
            // Enable for both MOE (hybrid) and FUSED_MOE architectures
            const enum ggml_amx_moe_arch arch = ggml_get_amx_moe_arch();
            if (arch == GGML_AMX_MOE_ARCH_MOE || arch == GGML_AMX_MOE_ARCH_FUSED_MOE) {
                // During model loading (buffer selection): buffers are null
                // During runtime dispatch: buffers are allocated
                // We only want to claim this operation for buffer selection, not dispatch
                if (op->src[0] && op->src[1] && !op->src[0]->buffer && !op->src[1]->buffer) {
                    // Buffer selection phase - claim it to use AMX buffer
                    if (qtype_has_amx_kernels(op->src[0]->type) &&
                        qtype_has_amx_kernels(op->src[1]->type) &&
                        op->ne[0] % (TILE_N * 2) == 0) {
                        return true;
                    }
                }
                // Runtime dispatch phase - return false to let repack backend handle it
                if (op->src[0] && op->src[1] && op->src[0]->buffer && op->src[1]->buffer) {
                    return false;  // Let repack backend dispatch to AMX or baseline
                }
            }
        }

        return false;
    }

    ggml::cpu::tensor_traits * get_tensor_traits(const struct ggml_tensor * op) override {
        if (op->op == GGML_OP_MUL_MAT && op->src[0]->buffer &&
            op->src[0]->buffer->buft == ggml_backend_amx_buffer_type()) {
            return (ggml::cpu::tensor_traits *) op->src[0]->extra;
        }

        return nullptr;
    }
};
}  // namespace ggml::cpu::amx

static size_t ggml_backend_amx_buffer_type_get_alloc_size(ggml_backend_buffer_type_t buft, const ggml_tensor * tensor) {
    return ggml_backend_amx_get_alloc_size(tensor);

    GGML_UNUSED(buft);
}

#define ARCH_GET_XCOMP_PERM     0x1022
#define ARCH_REQ_XCOMP_PERM     0x1023
#define XFEATURE_XTILECFG       17
#define XFEATURE_XTILEDATA      18

static bool ggml_amx_init() {
#if defined(__linux__)
    if (syscall(SYS_arch_prctl, ARCH_REQ_XCOMP_PERM, XFEATURE_XTILEDATA)) {
        fprintf(stderr, "AMX is not ready to be used!\n");
        return false;
    }
    return true;
#elif defined(_WIN32)
    return true;
#else
    return false;
#endif
}

ggml_backend_buffer_type_t ggml_backend_amx_buffer_type() {
    static struct ggml_backend_buffer_type ggml_backend_buffer_type_amx = {
        /* .iface = */ {
                        /* .get_name         = */ ggml_backend_amx_buffer_type_get_name,
                        /* .alloc_buffer     = */ ggml_backend_amx_buffer_type_alloc_buffer,
                        /* .get_alignment    = */ ggml_backend_amx_buffer_type_get_alignment,
                        /* .get_max_size     = */ nullptr,  // defaults to SIZE_MAX
                        /* .get_alloc_size   = */ ggml_backend_amx_buffer_type_get_alloc_size,
                        /* .is_host          = */ nullptr,
                        },
        /* .device  = */ ggml_backend_reg_dev_get(ggml_backend_cpu_reg(), 0),
        /* .context = */ new ggml::cpu::amx::extra_buffer_type(),
    };

    if (!ggml_amx_init()) {
        return nullptr;
    }

    return &ggml_backend_buffer_type_amx;
}

#endif  // defined(__AMX_INT8__) && defined(__AVX512VNNI__)
