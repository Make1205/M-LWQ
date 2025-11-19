// src/ntt.hpp
#pragma once
#include <vector>
#include <cstdint>
#include "params.hpp" // 获取 N, Q

namespace ntt {

    // 初始化 NTT 需要的预计算表 (zetas)
    void init_tables();

    // 使用 NTT 加速的多项式乘法
    // 输入: a, b (系数在 [0, Q) 或更宽松范围内)
    // 输出: a * b mod (X^N + 1) mod Q (系数在 [0, Q) 范围内)
    std::vector<int32_t> poly_mul_ntt(const std::vector<int32_t>& a, const std::vector<int32_t>& b);

}