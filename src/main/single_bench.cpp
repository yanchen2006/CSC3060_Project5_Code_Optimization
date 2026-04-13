#include <chrono>
#include <cstdint>
#include <iomanip>
#include <iostream>
#include <print>
#include <vector>

#include "bench.h"
#include "graph.h"

int main() {
    std::cout << "Benchmark setup\n";

    constexpr std::size_t graph_node_count = 1024000;
    constexpr int graph_avg_degree = 8;
    constexpr std::uint_fast64_t graph_seed = 42;

    graph_args graph_args_ref;
    graph_args graph_args_stu;
    initialize_graph(&graph_args_ref,
                      graph_node_count,
                      graph_avg_degree,
                      graph_seed);
    initialize_graph(&graph_args_stu,
                      graph_node_count,
                      graph_avg_degree,
                      graph_seed);

    std::cout << "\tGraph: node_count=" << graph_node_count
              << ", avg_degree=" << graph_avg_degree
              << ", random_seed=" << graph_seed << '\n';

    std::vector<bench_t> benchmarks = {{"Graph (Naive)",
                                        naive_graph_wrapper,
                                        naive_graph_wrapper,
                                        graph_check,
                                        &graph_args_ref,
                                        &graph_args_ref,
                                        BASELINE_GRAPH},
                                       {"Graph (Stu)",
                                        stu_graph_wrapper,
                                        naive_graph_wrapper,
                                        graph_check,
                                        &graph_args_stu,
                                        &graph_args_ref,
                                        BASELINE_GRAPH}};

    std::cout << "\nRunning Benchmarks...\n";
    std::cout << "--------------------------------------------------------\n";
    std::cout << std::left << std::setw(25) << "Benchmark" << std::setw(12)
              << "Status" << std::right << std::setw(15) << "Nanoseconds"
              << "\n";
    std::cout << "--------------------------------------------------------\n";

    for (const auto &bench : benchmarks) {
        std::chrono::nanoseconds avg_time{0};
        const int k_best = 20;

        for (int i = 0; i < k_best; ++i) {
            flush_cache();
            const auto elapsed = measure_time([&] { bench.tfunc(bench.args); });

            avg_time += elapsed;
            debug_log("DEBUG: {}-th measurement: {} ns\n",
                      i,
                      static_cast<std::uint64_t>(elapsed.count()));
        }
        avg_time /= static_cast<uint64_t>(k_best);

        bool correct =
            bench.checkFunc(bench.args, bench.ref_args, bench.naiveFunc);

        std::cout << std::left << std::setw(25) << bench.description;
        if (!correct) {
            std::cout << "\033[1;31mFAILED\033[0m" << std::right
                      << std::setw(15) << "N/A" << "\n";
            std::cout
                << "  Error: Results do not match naive implementation!\n";
        } else {
            std::cout << "\033[1;32mPASSED\033[0m" << std::right
                      << std::setw(15) << avg_time.count() << " ns";
            if (avg_time.count() > bench.baseline_time.count() * 1.1) {
                std::cout << " (SLOW)";
            }
            std::cout << "\n";
        }
    }

    return 0;
}
