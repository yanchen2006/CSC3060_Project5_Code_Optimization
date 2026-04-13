# CSC3060 Project 5 - Code optimization

This repository is for the design of project 5 code optimization. Students are required to optimize some functions and make them not worse than baseline.

## Code structure

```bash
code_opt_dev/
├── CMakeLists.txt
├── README.md
├── include
│   ├── bench.h # Necessary data structure for running benchmarks
│   ├── inverseSqrt.h # Example benchmark for inversed square root
│   └── kernels.h # Example code for matrix rotation
└── src
    ├── kernel # The implementation of all benchmarks
    │   ├── inverseSqrt.cpp
    │   └── kernels.cpp
    └── main
        ├── run_all.cpp # One-off calling all benchmarks
        └── single_bench.cpp # Measure benchmarks one by one

4 directories, 9 files
```

`bench.h` defines `bench_t` which includes the functions `tfunc` to test, corresponding naive versions `naiveFunc`, correctness verifying function `checkFunc`, arguments of `tfunc` and `naiveFunc`, as well as baseline in cycle count `baseline_cyc`. It also defines `get_cycles()` function to measure performance, `flush_cache()` to avoid the impact of cache before benchmarking.

## What is needed to add new functions

`kernels.h` describes what is needed for each single benchmark. The newly-added benchmark have files named as `<KERNEL_NAME.h>` and `<KERNEL_NAME.cpp>` (here `kernels.h` is just an example for demonstration, and you can see `kernels.h` and its `kernels.cpp` for more details), in each `<KERNEL_NAME>.h` and `<KERNEL_NAME>.cpp`, they should define necessary data structure for its corresponding benchmark function, benchmark declaration (naive and optimized), baseline cycle count, its correctness checking function as well as wrapper functions.

- The arguments in `bench_t::tfunc` and `bench_t::naiveFunc` are not necessarily the same, and the `bench_t::checkFunc` needs to check over the argument (i.e. output) at specific position, which is the first argument in `bench_t::args` and `bench_t::ref_args`.
- The wrapper function is used to unrolling arguments to call specific kernel implementation (e.g. optimized kernel or naive one), and this wrapper function would become `bench_t::tfunc` or `bench_t::naiveFunc`.

In each `<KERNEL_NAME>.cpp`, the implementation of specific benchmark, wrapper functions and correctness checking function should be here (like `kernels.cpp` and `inverseSqrt.cpp`).

## To fine-tune each benchmark

There are two target executable files: `single_bench.cpp` and `run_all`.cpp:

- In `single_bench.cpp`, you can add arbitrary benchmarks into `std::vector<bench_t> benchmarks` (but you must prepare all initialized input and output for all benchmarks before that), then you can measure all benchmarks in `benchmarks`. Current implementation would run each registered benchmark for 20 times and get average cycle count as final result.
- `run_all.cpp` should call all benchmarks at once, so it measures the total cycles to run all functions. The purpose of it is, the student should run it firstly with `perf` to find the hottest function, and optimize it. Then run all benchmarks again to find the next hottest function and optimize, and so on.

## Debug and build

Use the following command to build project:

```bash
$ cd code_opt_dev
$ cmake -B build .
$ cmake --build build
```

To enable `DEBUG` macro:

```bash
$ cmake -DCMAKE_CXX_FLAGS="-DDEBUG" -B build .
$ cmake --build build
```
