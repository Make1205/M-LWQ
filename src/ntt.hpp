#pragma once
#include <vector>
#include <cstdint>
#include "params.hpp"

namespace ntt {

    void init_tables();

    // 输入输出改为 int16_t
    std::vector<int16_t> poly_mul_ntt(const std::vector<int16_t>& a, const std::vector<int16_t>& b);

}