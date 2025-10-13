#ifndef NUMA_TOPOLOGY_H
#define NUMA_TOPOLOGY_H

#include <vector>
#include <map>

struct numa_topology {
    int n_numa_nodes;
    int n_sockets;
    std::map<int, int> numa_to_socket;              // NUMA node → socket ID
    std::map<int, std::vector<int>> socket_to_numas; // Socket → [NUMA nodes]
};

// NUMA replication strategies
enum numa_replicate_strategy {
    NUMA_REPLICATE_NONE,        // No replication (default for backward compatibility)
    NUMA_REPLICATE_AUTO,        // Auto-detect sockets and replicate per socket
    NUMA_REPLICATE_PER_NODE,    // Replicate on every NUMA node
    NUMA_REPLICATE_GROUPS,      // User-defined groups
};

enum numa_alloc_strategy {
    NUMA_ALLOC_INTERLEAVED,     // Interleave pages across NUMA nodes in group (default)
    NUMA_ALLOC_STRIPED,         // Stripe experts across NUMA nodes
};

struct numa_replication_config {
    numa_replicate_strategy replicate = NUMA_REPLICATE_NONE;
    numa_alloc_strategy alloc = NUMA_ALLOC_INTERLEAVED;
    std::vector<std::vector<int>> groups;  // For NUMA_REPLICATE_GROUPS
};

// Detect NUMA topology from system
// Returns topology with socket detection from sysfs
numa_topology detect_numa_topology();

// Print NUMA topology for diagnostics
void print_numa_topology(const numa_topology& topo);

// Print NUMA replication configuration
void print_numa_replication_config(const numa_replication_config& config, const numa_topology& topo);

// Check if NUMA is available on this system
bool is_numa_available();

#endif // NUMA_TOPOLOGY_H
