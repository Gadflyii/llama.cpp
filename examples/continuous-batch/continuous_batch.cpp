// Continuous Batching Implementation
// Maintains optimal batch size for AMX

#include "continuous_batch.h"
#include "common.h"
#include "log.h"

#include <algorithm>
#include <cstdio>
#include <cstring>

namespace llama_continuous_batch {

continuous_batch_scheduler::continuous_batch_scheduler(
    llama_context* ctx,
    const llama_vocab* vocab,
    const scheduler_config_t& config
)
    : ctx_(ctx)
    , vocab_(vocab)
    , config_(config)
    , next_request_id_(1)
    , batch_count_(0)
    , total_batch_size_(0)
    , t_start_us_(0)
    , next_seq_id_(0)
{
    // Initialize batch
    batch_ = llama_batch_init(config_.max_batch_size, 0, config_.max_sequences);
}

continuous_batch_scheduler::~continuous_batch_scheduler() {
    // Free all samplers
    for (auto& pair : requests_) {
        if (pair.second.sampler) {
            llama_sampler_free(pair.second.sampler);
        }
    }

    llama_batch_free(batch_);
}

uint64_t continuous_batch_scheduler::add_request(
    const std::vector<llama_token>& prompt,
    int max_tokens,
    float temperature,
    float top_p,
    int top_k
) {
    uint64_t req_id = next_request_id_++;

    request_t req;
    req.request_id = req_id;
    req.prompt_tokens = prompt;
    req.max_tokens = max_tokens;
    req.state = state_t::WAITING;
    req.prompt_pos = 0;
    req.t_start_us = llama_time_us();
    req.t_first_token_us = 0;
    req.t_end_us = 0;
    req.seq_id = -1;

    // Create sampler for this request
    auto sparams = llama_sampler_chain_default_params();
    sparams.no_perf = true;

    req.sampler = llama_sampler_chain_init(sparams);
    llama_sampler_chain_add(req.sampler, llama_sampler_init_top_k(top_k));
    llama_sampler_chain_add(req.sampler, llama_sampler_init_top_p(top_p, 1));
    llama_sampler_chain_add(req.sampler, llama_sampler_init_temp(temperature));
    llama_sampler_chain_add(req.sampler, llama_sampler_init_dist(LLAMA_DEFAULT_SEED));

    requests_[req_id] = req;
    waiting_queue_.push(req_id);

    stats_.total_requests++;

    return req_id;
}

void continuous_batch_scheduler::run() {
    t_start_us_ = llama_time_us();

    while (!requests_.empty()) {
        // Remove completed requests
        for (auto it = requests_.begin(); it != requests_.end(); ) {
            if (it->second.state == state_t::COMPLETED) {
                if (it->second.sampler) {
                    llama_sampler_free(it->second.sampler);
                    it->second.sampler = nullptr;
                }
                it = requests_.erase(it);
            } else {
                ++it;
            }
        }

        if (requests_.empty() || (waiting_queue_.empty() && active_requests_.empty())) {
            break;
        }

        if (!step()) {
            break;
        }
    }

    int64_t t_end_us = llama_time_us();
    stats_.total_time_us = t_end_us - t_start_us_;

    update_stats();
}

bool continuous_batch_scheduler::step() {
    form_batch();

    if (batch_.n_tokens == 0) {
        return false;  // No work to do
    }

    process_batch();
    return true;
}

void continuous_batch_scheduler::form_batch() {
    common_batch_clear(batch_);
    active_requests_.clear();

    int tokens_added = 0;

    // Phase 1: Add prefilling requests (priority)
    if (config_.prioritize_prefill) {
        for (auto& pair : requests_) {
            request_t& req = pair.second;
            if (req.state == state_t::PREFILLING && tokens_added < config_.target_batch_size) {
                // Add one token from prompt
                if (req.prompt_pos < (int)req.prompt_tokens.size()) {
                    llama_token token = req.prompt_tokens[req.prompt_pos];
                    bool is_last = (req.prompt_pos == (int)req.prompt_tokens.size() - 1);

                    common_batch_add(batch_, token, req.prompt_pos, {req.seq_id}, is_last);

                    req.prompt_pos++;
                    tokens_added++;

                    // Check if prefill is complete
                    if (req.prompt_pos >= (int)req.prompt_tokens.size()) {
                        req.state = state_t::GENERATING;
                    }

                    active_requests_.push_back(&req);
                }
            }
        }
    }

    // Phase 2: Add generating requests
    for (auto& pair : requests_) {
        request_t& req = pair.second;
        if (req.state == state_t::GENERATING && tokens_added < config_.max_batch_size) {
            // Add last generated token
            if (!req.output_tokens.empty()) {
                llama_token token = req.output_tokens.back();
                llama_pos pos = req.prompt_tokens.size() + req.output_tokens.size() - 1;

                common_batch_add(batch_, token, pos, {req.seq_id}, true);

                tokens_added++;
                active_requests_.push_back(&req);
            }
        }
    }

    // Phase 3: Start new waiting requests
    while (!waiting_queue_.empty() && tokens_added < config_.target_batch_size) {
        uint64_t req_id = waiting_queue_.front();
        waiting_queue_.pop();

        auto it = requests_.find(req_id);
        if (it == requests_.end()) {
            continue;
        }

        request_t& req = it->second;

        // Allocate sequence ID
        req.seq_id = allocate_seq_id();
        req.state = state_t::PREFILLING;
        req.prompt_pos = 0;
        req.t_first_token_us = llama_time_us();

        // Add first prompt token
        if (!req.prompt_tokens.empty()) {
            llama_token token = req.prompt_tokens[0];
            bool is_last = (req.prompt_tokens.size() == 1);

            common_batch_add(batch_, token, 0, {req.seq_id}, is_last);

            req.prompt_pos = 1;
            tokens_added++;

            if (req.prompt_pos >= (int)req.prompt_tokens.size()) {
                req.state = state_t::GENERATING;
            }

            active_requests_.push_back(&req);
        }
    }

    batch_count_++;
    total_batch_size_ += batch_.n_tokens;
}

void continuous_batch_scheduler::process_batch() {
    if (batch_.n_tokens == 0) {
        return;
    }

    // Decode the batch
    if (llama_decode(ctx_, batch_) != 0) {
        LOG_ERR("%s: llama_decode failed\n", __func__);
        // Mark all active requests as failed
        for (request_t* req : active_requests_) {
            req->state = state_t::COMPLETED;
            stats_.failed_requests++;
        }
        return;
    }

    // Sample next token for each generating request
    int batch_idx = 0;
    for (size_t i = 0; i < active_requests_.size(); ++i) {
        request_t* req = active_requests_[i];

        if (req->state == state_t::GENERATING) {
            // Find the batch index for this request's logits
            // The logits are in the order they were added with logits=true
            while (batch_idx < batch_.n_tokens && !batch_.logits[batch_idx]) {
                batch_idx++;
            }

            if (batch_idx >= batch_.n_tokens) {
                // No more logits available
                break;
            }

            // Sample next token
            llama_token next_token = llama_sampler_sample(req->sampler, ctx_, batch_idx);

            batch_idx++;

            // Check for end of generation
            if (llama_vocab_is_eog(vocab_, next_token) ||
                (int)req->output_tokens.size() >= req->max_tokens) {
                req->state = state_t::COMPLETED;
                req->t_end_us = llama_time_us();
                stats_.completed_requests++;

                // Free sequence ID
                free_seq_id(req->seq_id);
                continue;
            }

            req->output_tokens.push_back(next_token);
            stats_.total_tokens_generated++;
        }
    }
}

llama_seq_id continuous_batch_scheduler::allocate_seq_id() {
    if (!free_seq_ids_.empty()) {
        llama_seq_id id = free_seq_ids_.back();
        free_seq_ids_.pop_back();
        return id;
    }
    return next_seq_id_++;
}

void continuous_batch_scheduler::free_seq_id(llama_seq_id seq_id) {
    llama_memory_t mem = llama_get_memory(ctx_);
    llama_memory_seq_rm(mem, seq_id, -1, -1);
    free_seq_ids_.push_back(seq_id);
}

std::vector<llama_token> continuous_batch_scheduler::get_output_tokens(uint64_t request_id) const {
    auto it = requests_.find(request_id);
    if (it != requests_.end()) {
        return it->second.output_tokens;
    }
    return {};
}

void continuous_batch_scheduler::update_stats() {
    if (batch_count_ > 0) {
        stats_.avg_batch_size = (double)total_batch_size_ / batch_count_;
    }

    if (stats_.total_time_us > 0) {
        stats_.tokens_per_second = (double)stats_.total_tokens_generated / (stats_.total_time_us / 1000000.0);
        stats_.requests_per_second = (double)stats_.completed_requests / (stats_.total_time_us / 1000000.0);
    }

    // Calculate latency percentiles
    std::vector<int64_t> latencies;
    for (const auto& pair : requests_) {
        const request_t& req = pair.second;
        if (req.state == state_t::COMPLETED && req.t_end_us > req.t_start_us) {
            latencies.push_back(req.t_end_us - req.t_start_us);
        }
    }

    if (!latencies.empty()) {
        std::sort(latencies.begin(), latencies.end());
        stats_.latency_p50_ms = latencies[latencies.size() * 50 / 100] / 1000.0;
        stats_.latency_p90_ms = latencies[latencies.size() * 90 / 100] / 1000.0;
        stats_.latency_p99_ms = latencies[latencies.size() * 99 / 100] / 1000.0;
    }
}

void continuous_batch_scheduler::print_stats() const {
    fprintf(stderr, "\n=== Continuous Batching Statistics ===\n");
    fprintf(stderr, "Total Requests:    %zu\n", stats_.total_requests);
    fprintf(stderr, "Completed:         %zu\n", stats_.completed_requests);
    fprintf(stderr, "Failed:            %zu\n", stats_.failed_requests);
    fprintf(stderr, "Tokens Generated:  %ld\n", stats_.total_tokens_generated);
    fprintf(stderr, "\n");
    fprintf(stderr, "Total Time:        %.2f s\n", stats_.total_time_us / 1000000.0);
    fprintf(stderr, "Throughput:        %.2f tokens/sec\n", stats_.tokens_per_second);
    fprintf(stderr, "Request Rate:      %.2f requests/sec\n", stats_.requests_per_second);
    fprintf(stderr, "\n");
    fprintf(stderr, "Avg Batch Size:    %.2f\n", stats_.avg_batch_size);
    fprintf(stderr, "\n");
    fprintf(stderr, "Latency p50:       %.2f ms\n", stats_.latency_p50_ms);
    fprintf(stderr, "Latency p90:       %.2f ms\n", stats_.latency_p90_ms);
    fprintf(stderr, "Latency p99:       %.2f ms\n", stats_.latency_p99_ms);
    fprintf(stderr, "=====================================\n");
}

} // namespace llama_continuous_batch
