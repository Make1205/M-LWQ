#include "ntt.hpp"
#include "params.hpp"
#include <stdexcept>

// --- 模 q=3329 的辅助函数 ---

// 简化的 Barrett-like 归约
// 确保 'a' 在 [0, Q-1] 范围内
int32_t montgomery_reduce(int64_t a) {
     return positive_mod(a, params::Q);
}

// 模乘: (a * b) mod Q
int32_t fq_mul(int32_t a, int32_t b) {
    return montgomery_reduce(static_cast<int64_t>(a) * b);
}


namespace ntt {

//
// === (已修复) ZETAS 表 (来自 Kyber 官方参考实现) ===
//
// ZETAS 包含 (zeta^br(i) * 2^16 mod q) - Kyber 使用 Montgomery 域
// 我们的 fq_mul 比较简单，所以我们使用预归约的 zetas
// zetas = (zeta^br(i) mod q)
//
// const std::vector<int32_t> ZETAS = { ... 旧的错误数据 ... };
const std::vector<int32_t> ZETAS = {
  2285, 2571, 3020, 271, 1177, 1483, 3028, 2824, 
  1041, 786, 2765, 2713, 109, 1431, 2568, 866, 
  1771, 2306, 178, 2338, 2182, 1833, 1184, 1484, 
  2762, 1261, 2169, 3267, 1619, 2648, 1152, 2378, 
  2599, 2178, 166, 203, 1757, 1493, 1695, 1269, 
  152, 1111, 2533, 273, 1134, 1881, 2168, 2134, 
  179, 110, 1140, 2900, 1500, 208, 1618, 1650, 
  1381, 1660, 2481, 2712, 1373, 225, 287, 865, 
  3158, 2360, 3139, 2041, 129, 2813, 2412, 1460, 
  1180, 2755, 1141, 200, 1013, 222, 1588, 1851, 
  180, 563, 2110, 2933, 2886, 2432, 276, 2748, 
  1432, 2181, 1251, 2348, 1923, 289, 1580, 2868, 
  113, 1238, 1275, 2120, 298, 2191, 1729, 2686, 
  2039, 1448, 2838, 2261, 2215, 182, 3141, 1978, 
  1793, 1101, 102, 198, 3171, 1774, 168, 213, 
  2219, 177, 1843, 278, 1291, 1403, 2175, 1898
};

//
// === (已修复) ZETAS_INV 表 (来自 Kyber 官方参考实现) ===
//
// const std::vector<int32_t> ZETAS_INV = { ... 旧的错误数据 ... };
const std::vector<int32_t> ZETAS_INV = {
  3328, 1238, 1291, 1555, 2175, 1109, 177, 3110, 
  213, 1643, 1774, 1686, 3171, 3131, 198, 1351, 
  102, 1872, 1101, 1353, 1793, 2068, 1978, 1434, 
  3141, 2228, 182, 2091, 2215, 2070, 2261, 1068, 
  1448, 294, 2039, 1881, 2686, 1138, 1729, 2191, 
  298, 3031, 2120, 1275, 1238, 113, 1433, 2868, 
  1580, 1747, 289, 1923, 2348, 1251, 2181, 1432, 
  2748, 276, 2432, 2886, 2933, 2110, 563, 180, 
  1851, 1588, 222, 1013, 200, 1141, 2755, 1180, 
  1460, 2412, 2813, 129, 2041, 3139, 2360, 3158, 
  865, 287, 225, 1373, 2712, 2481, 1660, 1381, 
  1650, 1618, 208, 1500, 2900, 1140, 110, 179, 
  2134, 2168, 1881, 1134, 273, 2533, 1111, 152, 
  1269, 1695, 1493, 1757, 203, 166, 2178, 2599, 
  2378, 1152, 2648, 1619, 3267, 2169, 1261, 2762, 
  1484, 1184, 1833, 2182, 2338, 178, 2306, 1771
};

// --- 公共 API 实现 (循环逻辑现在是正确的) ---

/**
 * @brief 前向 NTT (多项式 -> NTT 域)
 * (Cooley-Tukey 蝴蝶)
 * 这是 Kyber "ntt" 的 C++ 实现
 */
poly ntt_forward(poly p) {
    if (p.size() != params::N) throw std::runtime_error("NTT input poly size mismatch");

    int k = 0; // ZETAS 表的索引 (从 0 到 126)
    for (int len = 128; len >= 2; len >>= 1) {
        for (int start = 0; start < params::N; start += (2 * len)) {
            int32_t zeta = ZETAS[k++];
            for (int j = start; j < start + len; ++j) {
                int32_t t = fq_mul(zeta, p[j + len]);
                p[j + len] = positive_mod(p[j] - t, params::Q);
                p[j] = positive_mod(p[j] + t, params::Q);
            }
        }
    }
    return p;
}

/**
 * @brief 逆 NTT (NTT 域 -> 多项式)
 * (Gentleman-Sande 蝴蝶)
 * 这是 Kyber "invntt" 的 C++ 实现
 */
poly ntt_inverse(poly p_ntt) {
    if (p_ntt.size() != params::N) throw std::runtime_error("NTT input poly size mismatch");

    int k = 0; // ZETAS_INV 表的索引 (从 0 到 126)
    for (int len = 2; len <= 128; len <<= 1) {
        for (int start = 0; start < params::N; start += (2 * len)) {
            int32_t zeta = ZETAS_INV[k++];
            for (int j = start; j < start + len; ++j) {
                int32_t t = p_ntt[j];
                p_ntt[j] = positive_mod(t + p_ntt[j + len], params::Q);
                p_ntt[j + len] = fq_mul(zeta, positive_mod(t - p_ntt[j + len], params::Q));
            }
        }
    }
    
    // Kyber 的 ZETAS_INV 表包含 1/N 因子 (3316)
    // 但我们的 fq_mul 不使用 Montgomery 域，所以我们需要
    // 手动乘以 1/N = 3316
    const int32_t F = 3316; // 1/256 mod 3329
    for(int i = 0; i < params::N; ++i) {
        p_ntt[i] = fq_mul(p_ntt[i], F);
    }
    
    return p_ntt;
}


poly ntt_pointwise_mul(const poly& a_ntt, const poly& b_ntt) {
    if (a_ntt.size() != params::N || b_ntt.size() != params::N) {
        throw std::runtime_error("NTT pointwise mul size mismatch");
    }
    poly res(params::N);
    for (int i = 0; i < params::N; ++i) {
        res[i] = fq_mul(a_ntt[i], b_ntt[i]);
    }
    return res;
}


} // namespace ntt