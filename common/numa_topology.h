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

// Detect NUMA topology from system
// Returns topology with socket detection from sysfs
numa_topology detect_numa_topology();

// Print NUMA topology for diagnostics
void print_numa_topology(const numa_topology& topo);

// Check if NUMA is available on this system
bool is_numa_available();

#endif // NUMA_TOPOLOGY_H
