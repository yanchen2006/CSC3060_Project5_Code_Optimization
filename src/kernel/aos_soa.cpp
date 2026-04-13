#include "aos_soa.h"

#include <cmath>
#include <cstddef>

void naive_aos_soa(float& out, const std::vector<aos_soa_bench::Naive>& data) {
    float sum = 0.0f;
    for (size_t i = 0; i < data.size(); ++i) {
        const aos_soa_bench::Naive& e = data[i];
        sum += e.ax * e.bz + e.cy;
    }
    out = sum;
}

void stu_aos_soa(float& out, const std::vector<aos_soa_bench::Naive>& data) {
    float sum = 0.0f;
    for (size_t i = 0; i < data.size(); ++i) {
        const auto& e = data[i];
        sum += e.ax * e.bz + e.cy;
    }
    out = sum;
}

void naive_aos_soa_wrapper(void* ctx) {
    auto** params = static_cast<void**>(ctx);
    auto* out = static_cast<float*>(params[0]);
    auto* data = static_cast<std::vector<aos_soa_bench::Naive>*>(params[1]);
    naive_aos_soa(*out, *data);
}

void stu_aos_soa_wrapper(void* ctx) {
    auto** params = static_cast<void**>(ctx);
    auto* out = static_cast<float*>(params[0]);
    auto* data = static_cast<std::vector<aos_soa_bench::Naive>*>(params[1]);
    stu_aos_soa(*out, *data);
}

bool aos_soa_check(void* stu_ctx, void* ref_ctx, lab_test_func naive_func) {
    // Compute reference.
    naive_func(ref_ctx);

    auto** stu_params = static_cast<void**>(stu_ctx);
    auto** ref_params = static_cast<void**>(ref_ctx);
    auto* out_stu = static_cast<float*>(stu_params[0]);
    auto* out_ref = static_cast<float*>(ref_params[0]);

    if (!out_stu || !out_ref) {
        return false;
    }

    const double s = static_cast<double>(*out_stu);
    const double r = static_cast<double>(*out_ref);
    const double abs_diff = std::abs(s - r);
    const double rel_diff = (std::abs(r) > 1e-12) ? abs_diff / std::abs(r) : abs_diff;

    debug_log("DEBUG: aos_soa stu={} ref={} abs_diff={} rel_diff={}\n",
              *out_stu,
              *out_ref,
              abs_diff,
              rel_diff);

    return abs_diff < 1e-5f;
}
