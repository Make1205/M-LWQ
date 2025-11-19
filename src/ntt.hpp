#pragma once
#include <vector>
#include <cstdint>
#include "params.hpp"

namespace ntt {

    // 初始化 NTT 表 (zetas)
    void init_tables();

    // NTT 加速的多项式乘法
    // 内部根据 params::USE_AVX2 选择路径
    std::vector<int32_t> poly_mul_ntt(const std::vector<int32_t>& a, const std::vector<int32_t>& b);

}