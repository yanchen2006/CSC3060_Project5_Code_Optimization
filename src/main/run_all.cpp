#include <chrono>
#include <cstdint>
#include <iomanip>
#include <iostream>
#include <print>
#include <vector>

#include "bench.h"
#include "bitwise.h"
#include "blackscholes.h"
#include "filter_gradient.h"
#include "graph.h"
#include "grff.h"
#include "image_proc.h"
#include "matmul.h"
#include "relu.h"
#include "sparse_spmm.h"
#include "trace_replay.h"

#define GEOMETRIC_MEAN 1

int main() {
    std::cout << "Benchmark setup\n";

    std::uint32_t seed = 12345u;

    // Black-Scholes
    blackscholes_args black_args_naive;
    initialize_blackscholes(black_args_naive, 81920, seed);
    blackscholes_args black_args_student;
    initialize_blackscholes(black_args_student, 81920, seed);
    std::cout << "\tBlack-Scholes options: " << black_args_naive.spot_price.size() << '\n';

    // Sparse SpMM
    sparse_spmm_args sparse_args_naive;
    initialize_spmm(sparse_args_naive, 512, 512, -1, {}, seed);
    sparse_spmm_args sparse_args_student;
    initialize_spmm(sparse_args_student, 512, 512, -1, {}, seed);
    std::cout << "\tSparse A (CSR): " << sparse_args_naive.csr.rows << " x "
              << sparse_args_naive.csr.cols << ", nnz=" << sparse_args_naive.csr.values.size() << '\n';

    // ReLU
    constexpr size_t relu_size = 1024000;
    relu_args relu_args_naive;
    initialize_relu(&relu_args_naive, relu_size, seed);
    relu_args relu_args_student;
    initialize_relu(&relu_args_student, relu_size, seed);
    std::println("\tReLU: vector length={}", relu_size);

    // Bitwise
    constexpr size_t bitwise_size = 1024000;
    bitwise_args bitwise_args_naive;
    initialize_bitwise(&bitwise_args_naive, bitwise_size, seed);
    bitwise_args bitwise_args_student;
    initialize_bitwise(&bitwise_args_student, bitwise_size, seed);
    std::println("\tBitwise: vector length={}", bitwise_size);

    // MatMul
    matmul_args matmul_args_naive;
    initialize_matmul(matmul_args_naive, 512, seed);
    matmul_args matmul_args_student;
    initialize_matmul(matmul_args_student, 512, seed);
    std::cout << "\tMatMul: n=" << matmul_args_naive.n << '\n';

    // Trace Replay
    trace_replay_args trace_args_naive;
    initialize_trace_replay(trace_args_naive, 1 << 16, 1 << 20, seed);
    trace_replay_args trace_args_student;
    initialize_trace_replay(trace_args_student, 1 << 16, 1 << 20, seed);
    std::cout << "\tTrace Replay: records=" << trace_args_naive.records.size()
              << ", trace_length=" << trace_args_naive.trace.size() << '\n';

    // Graph
    constexpr std::size_t graph_node_count = 1024000;
    constexpr int graph_avg_degree = 8;
    graph_args graph_args_naive;
    initialize_graph(&graph_args_naive, graph_node_count, graph_avg_degree, seed);
    graph_args graph_args_student;
    initialize_graph(&graph_args_student, graph_node_count, graph_avg_degree, seed);
    std::cout << "\tGraph: node_count=" << graph_node_count << ", avg_degree=" << graph_avg_degree << '\n';

    // GRFF
    constexpr std::size_t grff_size = 1024000;
    grff_args grff_args_naive;
    initialize_grff(&grff_args_naive, grff_size, seed);
    grff_args grff_args_student;
    initialize_grff(&grff_args_student, grff_size, seed);
    std::cout << "\tGRFF: feature size=" << grff_args_naive.a_features.size() << '\n';

    // Image Proc
    constexpr std::size_t image_width = 1024;
    constexpr std::size_t image_height = 1000;
    image_proc_args image_args_naive;
    initialize_image_proc(&image_args_naive, image_width, image_height, seed);
    image_proc_args image_args_student;
    initialize_image_proc(&image_args_student, image_width, image_height, seed);
    std::cout << "\tImage Proc: " << image_args_naive.width << " x " << image_args_naive.height << '\n';

    // Filter Gradient
    const std::size_t WIDTH = 1024;
    const std::size_t HEIGHT = 1024;
    filter_gradient_args filter_args_naive;
    initialize_filter_gradient(&filter_args_naive, WIDTH, HEIGHT, seed);
    filter_gradient_args filter_args_student;
    initialize_filter_gradient(&filter_args_student, WIDTH, HEIGHT, seed);
    std::cout << "\tFilter Gradient: " << HEIGHT << " x " << WIDTH << '\n';

    // Benchmarks: each kernel has both Naive and Student entries
    std::vector<bench_t> benchmarks = {
        // Black-Scholes
        {"Black-Scholes (Naive)",
         naive_BlkSchls_wrapper,
         naive_BlkSchls_wrapper,
         BlkSchls_check,
         &black_args_naive,
         &black_args_naive,
         BASELINE_BLACKSCHOLES},
        {"Black-Scholes (Student)",
         stu_BlkSchls_wrapper,
         naive_BlkSchls_wrapper,
         BlkSchls_check,
         &black_args_student,
         &black_args_student,
         BASELINE_BLACKSCHOLES},

        // Sparse SpMM
        {"Sparse SpMM (Naive)",
         naive_sparse_spmm_wrapper,
         naive_sparse_spmm_wrapper,
         sparse_spmm_check,
         &sparse_args_naive,
         &sparse_args_naive,
         BASELINE_SPARSE_SPMM},
        {"Sparse SpMM (Student)",
         stu_sparse_spmm_wrapper,
         naive_sparse_spmm_wrapper,
         sparse_spmm_check,
         &sparse_args_student,
         &sparse_args_student,
         BASELINE_SPARSE_SPMM},

        // ReLU
        {"ReLU (Naive)",
         naive_relu_wrapper,
         naive_relu_wrapper,
         relu_check,
         &relu_args_naive,
         &relu_args_naive,
         BASELINE_RELU},
        {"ReLU (Student)",
         stu_relu_wrapper,
         naive_relu_wrapper,
         relu_check,
         &relu_args_student,
         &relu_args_student,
         BASELINE_RELU},

        // Bitwise
        {"Bitwise (Naive)",
         naive_bitwise_wrapper,
         naive_bitwise_wrapper,
         bitwise_check,
         &bitwise_args_naive,
         &bitwise_args_naive,
         BASELINE_BITWISE},
        {"Bitwise (Student)",
         stu_bitwise_wrapper,
         naive_bitwise_wrapper,
         bitwise_check,
         &bitwise_args_student,
         &bitwise_args_student,
         BASELINE_BITWISE},

        // MatMul
        {"MatMul (Naive)",
         naive_matmul_wrapper,
         naive_matmul_wrapper,
         matmul_check,
         &matmul_args_naive,
         &matmul_args_naive,
         BASELINE_MATMUL},
        {"MatMul (Student)",
         stu_matmul_wrapper,
         naive_matmul_wrapper,
         matmul_check,
         &matmul_args_student,
         &matmul_args_student,
         BASELINE_MATMUL},

        // Trace Replay
        {"Trace Replay (Naive)",
         naive_trace_replay_wrapper,
         naive_trace_replay_wrapper,
         trace_replay_check,
         &trace_args_naive,
         &trace_args_naive,
         BASELINE_TRACE_REPLAY},
        {"Trace Replay (Student)",
         stu_trace_replay_wrapper,
         naive_trace_replay_wrapper,
         trace_replay_check,
         &trace_args_student,
         &trace_args_student,
         BASELINE_TRACE_REPLAY},

        // Graph
        {"Graph (Naive)",
         naive_graph_wrapper,
         naive_graph_wrapper,
         graph_check,
         &graph_args_naive,
         &graph_args_naive,
         BASELINE_GRAPH},
        {"Graph (Student)",
         stu_graph_wrapper,
         naive_graph_wrapper,
         graph_check,
         &graph_args_student,
         &graph_args_student,
         BASELINE_GRAPH},

        // GRFF
        {"GRFF (Naive)",
         naive_grff_wrapper,
         naive_grff_wrapper,
         grff_check,
         &grff_args_naive,
         &grff_args_naive,
         BASELINE_GRFF},
        {"GRFF (Student)",
         stu_grff_wrapper,
         naive_grff_wrapper,
         grff_check,
         &grff_args_student,
         &grff_args_student,
         BASELINE_GRFF},

        // Image Proc
        {"Image Proc (Naive)",
         naive_image_proc_wrapper,
         naive_image_proc_wrapper,
         image_proc_check,
         &image_args_naive,
         &image_args_naive,
         BASELINE_IMAGE_PROC},
        {"Image Proc (Student)",
         stu_image_proc_wrapper,
         naive_image_proc_wrapper,
         image_proc_check,
         &image_args_student,
         &image_args_student,
         BASELINE_IMAGE_PROC},

        // Filter Gradient
        {"Filter Gradient (Naive)",
         naive_filter_gradient_wrapper,
         naive_filter_gradient_wrapper,
         filter_gradient_check,
         &filter_args_naive,
         &filter_args_naive,
         BASELINE_FILTER_GRADIENT},
        {"Filter Gradient (Student)",
         stu_filter_gradient_wrapper,
         naive_filter_gradient_wrapper,
         filter_gradient_check,
         &filter_args_student,
         &filter_args_student,
         BASELINE_FILTER_GRADIENT},
    };

    std::cout << "\nRunning Benchmarks...\n";
    std::cout << "-----------------------------------------------------------------------\n";
    std::cout << std::left << std::setw(25) << "Benchmark" << std::setw(12)
              << "Status" << std::right << std::setw(15) << "Nanoseconds"
              << std::setw(12) << "Speedup" << '\n';
    std::cout << "-----------------------------------------------------------------------\n";

#if GEOMETRIC_MEAN
    std::vector<std::chrono::nanoseconds> measured_times;
    measured_times.reserve(benchmarks.size());
    bool gm_enable = true;
#endif

    constexpr int k_best = 20;
    for (const auto &bench : benchmarks) {
        std::chrono::nanoseconds avg_time{0};

        for (int i = 0; i < k_best; ++i) {
            flush_cache();
            const auto elapsed = measure_time([&] { bench.tfunc(bench.args); });
            avg_time += elapsed;
            debug_log("\tDEBUG: {}-th measurement: {} ns\n",
                      i,
                      static_cast<std::uint64_t>(elapsed.count()));
        }
        avg_time /= static_cast<uint64_t>(k_best);

        const bool correct =
            bench.checkFunc(bench.args, bench.ref_args, bench.naiveFunc);

        std::cout << std::left << std::setw(25) << bench.description;
        if (!correct) {
            std::cout << "\033[1;31mFAILED\033[0m" << std::right
                      << std::setw(15) << "N/A" << std::setw(12) << "N/A"
                      << '\n';
            std::cout << "  Error: Results do not match naive implementation!\n";
#if GEOMETRIC_MEAN
            gm_enable = false;
#endif
            continue;
        }

        const double speedup = calculate_speedup(bench, avg_time);
        std::cout << "\033[1;32mPASSED\033[0m" << std::right << std::setw(15)
                  << avg_time.count() << " ns" << std::setw(11) << std::fixed
                  << std::setprecision(3) << speedup << "x";
        if (avg_time.count() > bench.baseline_time.count() * 1.1) {
            std::cout << " (SLOWER)";
        }
        std::cout << '\n';
        std::cout.unsetf(std::ios::floatfield);
        std::cout << std::setprecision(6);

#if GEOMETRIC_MEAN
        measured_times.push_back(avg_time);
#endif
    }

#if GEOMETRIC_MEAN
    if (gm_enable && measured_times.size() == benchmarks.size()) {
        const double geometric_mean_speedup =
            calculate_geometric_mean_speedup(measured_times, benchmarks);
        std::println("\nGeometric mean speedup: {:.3f}x",
                     geometric_mean_speedup);
    } else {
        std::println("\nGeometric mean speedup: N/A");
    }
#endif

    return 0;
}
