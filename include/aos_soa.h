#ifndef AOS_SOA_H
#define AOS_SOA_H

#include <chrono>
#include <vector>
#include "bench.h"

const std::chrono::nanoseconds BASELINE_AOS_SOA{900000};
namespace aos_soa_bench {

struct Naive {
    float ax, ay, az;
    float bx, by, bz;
    float cx, cy, cz;
};

} 

void naive_aos_soa(float& out, const std::vector<aos_soa_bench::Naive>& data);
void stu_aos_soa(float& out, const std::vector<aos_soa_bench::Naive>& data);

void naive_aos_soa_wrapper(void* ctx);
void stu_aos_soa_wrapper(void* ctx);

bool aos_soa_check(void* stu_ctx, void* ref_ctx, lab_test_func naive_func);

#endif // AOS_SOA_H
