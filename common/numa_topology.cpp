#include "numa_topology.h"
#include "common.h"
#include <fstream>
#include <iostream>
#include <sstream>
#include <cstring>

#ifdef __linux__
#include <numa.h>
#include <numaif.h>
#else
// Dummy NUMA functions for non-Linux platforms
static int numa_available() { return -1; }
static int numa_num_configured_nodes() { return 1; }
#endif

bool is_numa_available() {
#ifdef __linux__
    return numa_available() != -1;
#else
    return false;
#endif
}

numa_topology detect_numa_topology() {
    numa_topology topo;

    // Check if NUMA is available
    if (!is_numa_available()) {
        // Single NUMA node, single socket fallback
        topo.n_numa_nodes = 1;
        topo.n_sockets = 1;
        topo.numa_to_socket[0] = 0;
        topo.socket_to_numas[0] = {0};
        return topo;
    }

#ifdef __linux__
    topo.n_numa_nodes = numa_num_configured_nodes();

    // Detect socket affinity for each NUMA node
    for (int numa = 0; numa < topo.n_numa_nodes; numa++) {
        // Try to read physical_package_id from sysfs
        // Format: /sys/devices/system/node/nodeN/cpuX/topology/physical_package_id
        // We need to find a CPU in this NUMA node first

        std::string cpulist_path = "/sys/devices/system/node/node" +
                                   std::to_string(numa) + "/cpulist";

        std::ifstream cpulist_file(cpulist_path);
        if (!cpulist_file) {
            // Fallback: assume one socket per NUMA node
            topo.numa_to_socket[numa] = numa;
            topo.socket_to_numas[numa].push_back(numa);
            continue;
        }

        // Parse cpulist (format: "0-31,64-95" or "0-31")
        std::string cpulist_str;
        std::getline(cpulist_file, cpulist_str);
        cpulist_file.close();

        // Extract first CPU number
        int first_cpu = -1;
        size_t dash_pos = cpulist_str.find('-');
        size_t comma_pos = cpulist_str.find(',');

        if (dash_pos != std::string::npos) {
            // Format: "X-Y" - get X
            first_cpu = std::stoi(cpulist_str.substr(0, dash_pos));
        } else if (comma_pos != std::string::npos) {
            // Format: "X,Y" - get X
            first_cpu = std::stoi(cpulist_str.substr(0, comma_pos));
        } else {
            // Single CPU
            first_cpu = std::stoi(cpulist_str);
        }

        if (first_cpu < 0) {
            // Fallback
            topo.numa_to_socket[numa] = numa;
            topo.socket_to_numas[numa].push_back(numa);
            continue;
        }

        // Read physical_package_id for this CPU
        std::string socket_path = "/sys/devices/system/cpu/cpu" +
                                  std::to_string(first_cpu) +
                                  "/topology/physical_package_id";

        std::ifstream socket_file(socket_path);
        if (!socket_file) {
            // Fallback
            topo.numa_to_socket[numa] = numa;
            topo.socket_to_numas[numa].push_back(numa);
            continue;
        }

        int socket_id;
        socket_file >> socket_id;
        socket_file.close();

        topo.numa_to_socket[numa] = socket_id;
        topo.socket_to_numas[socket_id].push_back(numa);
    }

    topo.n_sockets = topo.socket_to_numas.size();
#else
    // Non-Linux fallback
    topo.n_numa_nodes = 1;
    topo.n_sockets = 1;
    topo.numa_to_socket[0] = 0;
    topo.socket_to_numas[0] = {0};
#endif

    return topo;
}

void print_numa_topology(const numa_topology& topo) {
    std::cout << "NUMA Topology:" << std::endl;
    std::cout << "  NUMA nodes: " << topo.n_numa_nodes << std::endl;
    std::cout << "  Sockets: " << topo.n_sockets << std::endl;

    for (const auto& [socket_id, numa_nodes] : topo.socket_to_numas) {
        std::cout << "  Socket " << socket_id << ": NUMA [";
        for (size_t i = 0; i < numa_nodes.size(); i++) {
            std::cout << numa_nodes[i];
            if (i < numa_nodes.size() - 1) std::cout << ",";
        }
        std::cout << "]" << std::endl;
    }

    // Also print memory available per node
#ifdef __linux__
    if (is_numa_available()) {
        std::cout << "  Memory per node:" << std::endl;
        for (int numa = 0; numa < topo.n_numa_nodes; numa++) {
            long long free_mem, total_mem;
            if (numa_node_size64(numa, &free_mem) == 0) {
                // Convert bytes to GB
                double total_gb = total_mem / (1024.0 * 1024.0 * 1024.0);
                double free_gb = free_mem / (1024.0 * 1024.0 * 1024.0);
                std::cout << "    NUMA " << numa << ": "
                          << total_gb << " GB total, "
                          << free_gb << " GB free" << std::endl;
            }
        }
    }
#endif
}

void print_numa_replication_config(const numa_replication_config& config, const numa_topology& topo) {
    std::cout << "NUMA Replication Configuration:" << std::endl;

    std::cout << "  Strategy: ";
    switch (config.replicate) {
        case NUMA_REPLICATE_NONE:
            std::cout << "none (no replication)" << std::endl;
            return;  // No need to print more if disabled
        case NUMA_REPLICATE_AUTO:
            std::cout << "auto (socket-grouped)" << std::endl;
            std::cout << "  Detected groups from topology:" << std::endl;
            for (const auto& [socket_id, numa_nodes] : topo.socket_to_numas) {
                std::cout << "    Group (Socket " << socket_id << "): NUMA [";
                for (size_t i = 0; i < numa_nodes.size(); i++) {
                    std::cout << numa_nodes[i];
                    if (i < numa_nodes.size() - 1) std::cout << ",";
                }
                std::cout << "]" << std::endl;
            }
            break;
        case NUMA_REPLICATE_PER_NODE:
            std::cout << "per-node (replicate on every NUMA node)" << std::endl;
            std::cout << "  Groups: " << topo.n_numa_nodes << " groups (one per NUMA node)" << std::endl;
            for (int numa = 0; numa < topo.n_numa_nodes; numa++) {
                std::cout << "    Group " << numa << ": NUMA [" << numa << "]" << std::endl;
            }
            break;
        case NUMA_REPLICATE_GROUPS:
            std::cout << "groups (user-defined)" << std::endl;
            std::cout << "  User-defined groups:" << std::endl;
            for (size_t g = 0; g < config.groups.size(); g++) {
                std::cout << "    Group " << g << ": NUMA [";
                for (size_t i = 0; i < config.groups[g].size(); i++) {
                    std::cout << config.groups[g][i];
                    if (i < config.groups[g].size() - 1) std::cout << ",";
                }
                std::cout << "]" << std::endl;
            }
            break;
    }

    std::cout << "  Allocation strategy: ";
    switch (config.alloc) {
        case NUMA_ALLOC_INTERLEAVED:
            std::cout << "interleaved (pages distributed across group)" << std::endl;
            break;
        case NUMA_ALLOC_STRIPED:
            std::cout << "striped (experts mapped to NUMA nodes)" << std::endl;
            break;
    }

    // Estimate memory multiplier
    if (config.replicate != NUMA_REPLICATE_NONE) {
        int n_groups = 0;
        switch (config.replicate) {
            case NUMA_REPLICATE_AUTO:
                n_groups = topo.n_sockets;
                break;
            case NUMA_REPLICATE_PER_NODE:
                n_groups = topo.n_numa_nodes;
                break;
            case NUMA_REPLICATE_GROUPS:
                n_groups = config.groups.size();
                break;
            default:
                break;
        }
        std::cout << "  Memory multiplier: " << n_groups << "× (weight replication)" << std::endl;
    }
}
