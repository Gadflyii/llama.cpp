// Continuous Batching for llama.cpp
// Efficient multi-request inference with dynamic batch sizing

#pragma once

#include "llama.h"
#include "common.h"

#include <vector>
#include <queue>
#include <unordered_map>
#include <cstdint>

namespace llama_continuous_batch {

// Request states
enum class state_t {
    WAITING,     // Queued, not yet started
    PREFILLING,  // Processing prompt tokens
    GENERATING,  // Generating output tokens
    COMPLETED,   // Finished or failed
};

// Individual inference request
struct request_t {
    uint64_t request_id;
    llama_seq_id seq_id;

    std::vector<llama_token> prompt_tokens;
    std::vector<llama_token> output_tokens;

    state_t state;
    int max_tokens;
    int prompt_pos;  // Current position in prompt (for prefill)

    // Sampling params
    llama_sampler * sampler;

    // Timing
    int64_t t_start_us;
    int64_t t_first_token_us;
    int64_t t_end_us;
};

// Scheduler configuration
struct scheduler_config_t {
    int target_batch_size = 16;
    int max_batch_size = 32;
    int min_batch_size = 8;
    bool dynamic_batching = true;
    bool prioritize_prefill = true;
    int max_context_length = 8192;
    int max_sequences = 64;
};

// Statistics
struct batch_stats_t {
    size_t total_requests = 0;
    size_t completed_requests = 0;
    size_t failed_requests = 0;
    int64_t total_tokens_generated = 0;

    double avg_batch_size = 0.0;
    double tokens_per_second = 0.0;
    double requests_per_second = 0.0;

    int64_t total_time_us = 0;

    // Latency percentiles (ms)
    double latency_p50_ms = 0.0;
    double latency_p90_ms = 0.0;
    double latency_p99_ms = 0.0;
};

// Main continuous batching scheduler
class continuous_batch_scheduler {
public:
    continuous_batch_scheduler(
        llama_context* ctx,
        const llama_vocab* vocab,
        const scheduler_config_t& config
    );

    ~continuous_batch_scheduler();

    // Add a new request to the queue
    uint64_t add_request(
        const std::vector<llama_token>& prompt,
        int max_tokens,
        float temperature = 0.8f,
        float top_p = 0.95f,
        int top_k = 40
    );

    // Run one step of the scheduler
    bool step();

    // Run until all requests complete
    void run();

    // Get request output
    std::vector<llama_token> get_output_tokens(uint64_t request_id) const;

    // Statistics
    const batch_stats_t& get_stats() const { return stats_; }
    void print_stats() const;

private:
    void form_batch();
    void process_batch();
    void update_stats();
    llama_seq_id allocate_seq_id();
    void free_seq_id(llama_seq_id seq_id);

    llama_context* ctx_;
    const llama_vocab* vocab_;
    scheduler_config_t config_;

    llama_batch batch_;

    std::unordered_map<uint64_t, request_t> requests_;
    std::queue<uint64_t> waiting_queue_;
    std::vector<request_t*> active_requests_;

    std::vector<llama_seq_id> free_seq_ids_;
    llama_seq_id next_seq_id_;

    uint64_t next_request_id_;
    batch_stats_t stats_;

    int64_t batch_count_;
    int64_t total_batch_size_;
    int64_t t_start_us_;
};

} // namespace llama_continuous_batch
