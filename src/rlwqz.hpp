#pragma once
#include "poly.hpp"
#include <vector>
#include <cstdint>
#include <utility> // for std::pair

// 公钥 (a, b, d)
struct rlwqz_pk {
    poly a;
    poly b;
    poly d;
};

// 私钥 (s)
using rlwqz_sk = poly;

/**
 * @brief 生成 RLWQZ 密钥对
 * * @param seed_ad 用于生成 a 和 d 的公共种子
 * @return std::pair<rlwqz_pk, rlwqz_sk> 
 */
std::pair<rlwqz_pk, rlwqz_sk> rlwqz_keygen(const std::vector<uint8_t>& seed_ad);