// Continuous Batching Demo
// Demonstrates high-throughput multi-request inference

#include "continuous_batch.h"
#include "arg.h"
#include "common.h"
#include "log.h"
#include "llama.h"
#include "ggml-cpu.h"

#include <cstdio>
#include <cstring>
#include <string>
#include <vector>

using namespace llama_continuous_batch;

static void print_usage(int, char ** argv) {
    LOG("\nContinuous Batching Demo - High-throughput multi-request inference\n");
    LOG("\nUsage:\n");
    LOG("    %s -m model.gguf [options]\n", argv[0]);
    LOG("\nOptions:\n");
    LOG("    -m,    --model PATH         Model path (required)\n");
    LOG("    -p,    --prompt TEXT        Base prompt (default: \"Once upon a time\")\n");
    LOG("    -n,    --predict N          Tokens to generate per request (default: 64)\n");
    LOG("    -cb-req, --cb-requests N    Number of concurrent requests (default: 16)\n");
    LOG("    -cb-batch, --cb-batch N     Target batch size M (default: 16)\n");
    LOG("           --cb-max-batch N     Maximum batch size (default: 32)\n");
    LOG("    -t,    --threads N          Number of threads (default: 4)\n");
    LOG("           --temp N             Temperature (default: 0.8)\n");
    LOG("           --top-p N            Top-p sampling (default: 0.95)\n");
    LOG("           --top-k N            Top-k sampling (default: 40)\n");
    LOG("    -cb-v, --cb-verbose         Show sample outputs\n");
    LOG("\n");
}

int main(int argc, char ** argv) {
    common_params params;

    params.prompt = "Once upon a time";
    params.n_predict = 64;

    // Custom args for continuous batching
    int n_requests = 16;
    int target_batch = 16;
    int max_batch = 32;
    bool verbose = false;

    // Parse custom arguments FIRST (using -cb- prefix to avoid conflicts)
    // We need to do this before common_params_parse to avoid errors
    std::vector<char*> filtered_argv;
    filtered_argv.push_back(argv[0]);

    for (int i = 1; i < argc; ++i) {
        std::string arg = argv[i];
        if (arg == "-cb-req" || arg == "--cb-requests") {
            if (i + 1 < argc) n_requests = std::atoi(argv[++i]);
        } else if (arg == "-cb-batch" || arg == "--cb-batch") {
            if (i + 1 < argc) target_batch = std::atoi(argv[++i]);
        } else if (arg == "--cb-max-batch") {
            if (i + 1 < argc) max_batch = std::atoi(argv[++i]);
        } else if (arg == "-cb-v" || arg == "--cb-verbose") {
            verbose = true;
        } else {
            // Not a custom arg, pass to common_params_parse
            filtered_argv.push_back(argv[i]);
        }
    }

    // Parse arguments with filtered argv
    if (!common_params_parse(filtered_argv.size(), filtered_argv.data(), params, LLAMA_EXAMPLE_COMMON, print_usage)) {
        return 1;
    }

    common_init();

    // Initialize backend
    llama_backend_init();
    llama_numa_init(params.numa);
    ggml_amx_moe_init(params.amx_arch);

    // Initialize NUMA weight replication for MoE models
    if (params.numa_replication.replicate != NUMA_REPLICATE_NONE) {
        // Convert groups to comma-separated string
        std::string groups_str;
        if (params.numa_replication.replicate == NUMA_REPLICATE_GROUPS &&
            !params.numa_replication.groups.empty() &&
            !params.numa_replication.groups[0].empty()) {
            for (size_t i = 0; i < params.numa_replication.groups[0].size(); i++) {
                if (i > 0) groups_str += ",";
                groups_str += std::to_string(params.numa_replication.groups[0][i]);
            }
        }

        ggml_backend_amx_numa_init(
            static_cast<int>(params.numa_replication.replicate),
            static_cast<int>(params.numa_replication.alloc),
            groups_str.empty() ? nullptr : groups_str.c_str()
        );
    }

    // Load model
    llama_model_params model_params = common_model_params_to_llama(params);
    llama_model * model = llama_model_load_from_file(params.model.path.c_str(), model_params);

    if (model == NULL) {
        LOG_ERR("%s: error: unable to load model\n", __func__);
        return 1;
    }

    const llama_vocab * vocab = llama_model_get_vocab(model);

    // Initialize context
    llama_context_params ctx_params = common_context_params_to_llama(params);
    ctx_params.n_ctx = 8192;
    ctx_params.n_batch = max_batch;
    ctx_params.n_ubatch = max_batch;
    ctx_params.n_seq_max = n_requests + 8;

    llama_context * ctx = llama_init_from_model(model, ctx_params);

    if (ctx == NULL) {
        LOG_ERR("%s: error: failed to create context\n", __func__);
        llama_model_free(model);
        return 1;
    }

    LOG("\n=== Continuous Batching Demo ===\n");
    LOG("Model:          %s\n", params.model.path.c_str());
    LOG("Prompt:         %s\n", params.prompt.c_str());
    LOG("Requests:       %d\n", n_requests);
    LOG("Tokens/request: %d\n", params.n_predict);
    LOG("Target batch:   %d\n", target_batch);
    LOG("Max batch:      %d\n", max_batch);
    LOG("Threads:        %d\n", params.cpuparams.n_threads);
    LOG("================================\n\n");

    // Tokenize prompt
    std::vector<llama_token> prompt_tokens = common_tokenize(vocab, params.prompt, true);
    LOG("Prompt tokens:  %zu\n\n", prompt_tokens.size());

    // Create scheduler
    scheduler_config_t config;
    config.target_batch_size = target_batch;
    config.max_batch_size = max_batch;
    config.min_batch_size = target_batch / 2;
    config.dynamic_batching = true;
    config.prioritize_prefill = true;
    config.max_context_length = 8192;
    config.max_sequences = n_requests + 8;

    continuous_batch_scheduler scheduler(ctx, vocab, config);

    // Add requests
    LOG("Adding %d requests...\n", n_requests);
    std::vector<uint64_t> request_ids;
    for (int i = 0; i < n_requests; ++i) {
        uint64_t req_id = scheduler.add_request(
            prompt_tokens,
            params.n_predict,
            params.sampling.temp,
            params.sampling.top_p,
            params.sampling.top_k
        );
        request_ids.push_back(req_id);
    }

    LOG("\n=== Starting continuous batching ===\n\n");

    // Run scheduler
    int64_t t_start = llama_time_us();
    scheduler.run();
    int64_t t_end = llama_time_us();

    double elapsed_s = (t_end - t_start) / 1000000.0;

    LOG("\n=== Processing complete ===\n");
    LOG("Total time: %.2f seconds\n\n", elapsed_s);

    // Print statistics
    scheduler.print_stats();

    // Print sample outputs if verbose
    if (verbose) {
        LOG("\n=== Sample Outputs ===\n");
        for (size_t i = 0; i < std::min(size_t(3), request_ids.size()); ++i) {
            uint64_t req_id = request_ids[i];
            auto output = scheduler.get_output_tokens(req_id);

            LOG("\nRequest %lu (%zu tokens):\n", req_id, output.size());
            LOG("%s", params.prompt.c_str());

            for (llama_token token : output) {
                LOG("%s", common_token_to_piece(ctx, token).c_str());
            }
            LOG("\n");
        }
    }

    // Cleanup
    llama_free(ctx);
    llama_model_free(model);
    llama_backend_free();

    return 0;
}
