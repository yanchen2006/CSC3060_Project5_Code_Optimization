#include <chrono>
#include <cstdint>
#include <iomanip>
#include <iostream>
#include <print>
#include <vector>

#include "bench.h"
#include "image_proc.h"

int main() {
    std::uint32_t seed = 12345u;
    constexpr size_t width = 1024;
    constexpr size_t height = 1000;   // 与 run_all 中 Image Proc 尺寸一致
    image_proc_args args_naive;
    initialize_image_proc(&args_naive, width, height, seed);
    std::println("\tImage Proc: {} x {}", width, height);

    // Student 版本需要独立的参数实例
    image_proc_args args_student;
    initialize_image_proc(&args_student, width, height, seed);

    std::vector<bench_t> benchmarks = {
        {"Image Proc (Naive)",
         naive_image_proc_wrapper,
         naive_image_proc_wrapper,
         image_proc_check,
         &args_naive,
         &args_naive,
         BASELINE_IMAGE_PROC},
        {"Image Proc (Student)",
         stu_image_proc_wrapper,
         naive_image_proc_wrapper,
         image_proc_check,
         &args_student,
         &args_student,
         BASELINE_IMAGE_PROC},
    };

    std::cout << "\nRunning Benchmarks...\n";
    std::cout << "--------------------------------------------------------\n";
    std::cout << std::left << std::setw(25) << "Benchmark" << std::setw(12)
              << "Status" << std::right << std::setw(15) << "Nanoseconds"
              << std::setw(12) << "Speedup" << "\n";
    std::cout << "--------------------------------------------------------\n";

    for (const auto &bench : benchmarks) {
        std::chrono::nanoseconds avg_time{0};
        const int k_best = 20;

        for (int i = 0; i < k_best; ++i) {
            flush_cache();
            const auto elapsed = measure_time([&] { bench.tfunc(bench.args); });
            avg_time += elapsed;
            debug_log("\tDEBUG: {}-th measurement: {} ns\n",
                      i,
                      static_cast<std::uint64_t>(elapsed.count()));
        }
        avg_time /= static_cast<uint64_t>(k_best);

        bool correct =
            bench.checkFunc(bench.args, bench.ref_args, bench.naiveFunc);

        std::cout << std::left << std::setw(25) << bench.description;
        if (!correct) {
            std::cout << "\033[1;31mFAILED\033[0m" << std::right
                      << std::setw(15) << "N/A" << std::setw(12) << "N/A"
                      << "\n";
            std::cout << "  Error: Results do not match naive implementation!\n";
            continue;
        }

        double speedup = static_cast<double>(bench.baseline_time.count()) /
                         static_cast<double>(avg_time.count());
        std::cout << "\033[1;32mPASSED\033[0m" << std::right << std::setw(15)
                  << avg_time.count() << " ns" << std::setw(11) << std::fixed
                  << std::setprecision(3) << speedup << "x";
        if (avg_time.count() > bench.baseline_time.count() * 1.1) {
            std::cout << " (SLOW)";
        }
        std::cout << "\n";
    }

    return 0;
}
