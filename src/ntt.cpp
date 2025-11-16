#include "ntt.hpp"
#include "params.hpp"
#include <stdexcept>

// --- Montgomery 算术 ---
// 
// Q = 3329
// R = 2^16 = 65536
// R_INV = 1 (mod Q)  (这不对, 65536 mod 3329 = 2087)
// Q_INV = -Q_inv (mod R) = 62209
// 
// 为简单起见，我们将避免使用 Montgomery 算术，
// 而是在 C++ 中使用 64-bit 整数执行标准模乘。
// 性能会稍慢，但逻辑更清晰。

// --- 辅助函数 ---

// a * b mod q
int32_t montgomery_reduce(int64_t a) {
     // 这是一个简化的、非 Montgomery 的 Barrett-like 归约
     // 对于我们的 Q=3329 来说已经足够了
     return positive_mod(a, params::Q);
}

int32_t fq_mul(int32_t a, int32_t b) {
    return montgomery_reduce(static_cast<int64_t>(a) * b);
}


namespace ntt {

// ZETAS 表 (N=256, Q=3329, ZETA=17)
// 预计算和位逆序的 "twiddle factors"
const std::vector<int32_t> ZETAS = {
    17, 140, 1632, 2187, 240, 2029, 3012, 2811, 
    2759, 290, 80, 2197, 1863, 179, 1373, 2568, 
    3249, 1827, 2246, 2135, 1260, 2697, 2125, 2004, 
    1678, 1726, 706, 3192, 1799, 1177, 112, 1470, 
    1600, 2872, 1171, 3110, 2816, 2162, 2415, 2307, 
    138, 2012, 2736, 1782, 3302, 1718, 1228, 2710, 
    500, 3143, 2061, 2223, 137, 814, 2110, 2623, 
    1830, 1699, 3112, 1361, 3182, 1367, 155, 3236, 
    2035, 3111, 2618, 2410, 712, 1603, 151, 1074, 
    1567, 278, 214, 1891, 2368, 2981, 2772, 2901, 
    1421, 2366, 2809, 2519, 1558, 2253, 915, 2917, 
    2282, 2038, 3077, 2526, 2769, 149, 1152, 2588, 
    1288, 1982, 267, 1533, 1805, 3086, 1759, 233, 
    3062, 1007, 959, 2576, 2777, 185, 3254, 1303, 
    552, 2147, 50, 1018, 2154, 210, 882, 2115, 
    274, 1072, 708, 1459, 1735, 960, 2353, 3300
};

// 逆 NTT 因子 (zeta_inv * 1/N)
// F = 1/N = 1/256 mod 3329 = 3316
// ZETAS_INV[i] = zetas[127-i] * F
const std::vector<int32_t> ZETAS_INV = {
    3316, 2369, 1594, 2257, 1595, 225, 2496, 2900, 
    2150, 2642, 856, 1395, 1020, 1610, 1500, 1720, 
    594, 230, 2217, 1818, 1182, 1802, 2901, 104, 
    188, 3072, 1570, 245, 1222, 1146, 2796, 603, 
    1325, 2147, 2652, 211, 1968, 2153, 281, 177, 
    1962, 1176, 2154, 1530, 962, 1106, 1322, 593, 
    3093, 2106, 1528, 2621, 1969, 282, 28, 1630, 
    708, 922, 2829, 2223, 706, 1629, 2367, 3152, 
    1961, 502, 2112, 1217, 1527, 2361, 2309, 1022, 
    321, 3183, 1934, 1762, 2307, 323, 2132, 1223, 
    963, 1197, 1829, 3110, 3144, 1362, 604, 1553, 
    1179, 1935, 3215, 1021, 1780, 2736, 2306, 157, 
    1023, 2152, 194, 1512, 1195, 934, 111, 3184, 
    2206, 1183, 2133, 1112, 1735, 87, 726, 1105, 
    1552, 1360, 2214, 1178, 1797, 3008, 2151, 1967, 
    232, 707, 1175, 2308, 215, 964, 3143, 1767
};


// --- NTT 核心 "蝴蝶" 运算 ---

/**
 * @brief Forward NTT butterfly layer
 * (基于 Kyber/Dilithium 的 "CT" (Cooley-Tukey) 蝴蝶)
 * * @param p 多项式系数
 * @param len 当前子多项式的长度 (从 128 递减到 2)
 * @param offset 当前处理块的起始索引
 * @param zetas_ptr 指向当前层级 zetas 表的指针
 */
void ntt_layer_forward(poly& p, int len, int offset, const int32_t* zetas_ptr) {
    for (int i = 0; i < len; ++i) {
        int32_t zeta = *zetas_ptr++;
        int idx1 = offset + i;
        int idx2 = offset + i + len;
        
        int32_t t = fq_mul(zeta, p[idx2]);
        p[idx2] = positive_mod(p[idx1] - t, params::Q);
        p[idx1] = positive_mod(p[idx1] + t, params::Q);
    }
}

/**
 * @brief Inverse NTT butterfly layer
 * (基于 Kyber/Dilithium 的 "GS" (Gentleman-Sande) 蝴蝶)
 * * @param p_ntt NTT 域系数
 * @param len 当前子多项式的长度 (从 2 递增到 128)
 * @param offset 当前处理块的起始索引
 * @param zetas_ptr 指向当前层级 zetas_inv 表的指针
 */
void ntt_layer_inverse(poly& p_ntt, int len, int offset, const int32_t* zetas_ptr) {
    for (int i = 0; i < len; ++i) {
        int32_t zeta = *zetas_ptr++;
        int idx1 = offset + i;
        int idx2 = offset + i + len;
        
        int32_t t = positive_mod(p_ntt[idx1] - p_ntt[idx2], params::Q);
        p_ntt[idx1] = positive_mod(p_ntt[idx1] + p_ntt[idx2], params::Q);
        p_ntt[idx2] = fq_mul(zeta, t);
    }
}

// --- 公共 API 实现 ---

poly ntt_forward(poly p) {
    if (p.size() != params::N) throw std::runtime_error("NTT input poly size mismatch");
    
    int zeta_ptr = 0;
    
    // 7 层 CT 蝴蝶
    ntt_layer_forward(p, 128, 0, &ZETAS[zeta_ptr]);
    zeta_ptr += 128;
    
    ntt_layer_forward(p, 64, 0, &ZETAS[zeta_ptr]);
    ntt_layer_forward(p, 64, 128, &ZETAS[zeta_ptr]);
    zeta_ptr += 64;

    ntt_layer_forward(p, 32, 0, &ZETAS[zeta_ptr]);
    ntt_layer_forward(p, 32, 64, &ZETAS[zeta_ptr]);
    ntt_layer_forward(p, 32, 128, &ZETAS[zeta_ptr]);
    ntt_layer_forward(p, 32, 192, &ZETAS[zeta_ptr]);
    zeta_ptr += 32;

    for(int i = 0; i < 8; ++i) ntt_layer_forward(p, 16, i*32, &ZETAS[zeta_ptr]);
    zeta_ptr += 16;
    for(int i = 0; i < 16; ++i) ntt_layer_forward(p, 8, i*16, &ZETAS[zeta_ptr]);
    zeta_ptr += 8;
    for(int i = 0; i < 32; ++i) ntt_layer_forward(p, 4, i*8, &ZETAS[zeta_ptr]);
    zeta_ptr += 4;
    for(int i = 0; i < 64; ++i) ntt_layer_forward(p, 2, i*4, &ZETAS[zeta_ptr]);

    return p;
}


poly ntt_inverse(poly p_ntt) {
    if (p_ntt.size() != params::N) throw std::runtime_error("NTT input poly size mismatch");

    int zeta_ptr = 0;

    // 7 层 GS 蝴蝶
    for(int i = 0; i < 64; ++i) ntt_layer_inverse(p_ntt, 2, i*4, &ZETAS_INV[zeta_ptr]);
    zeta_ptr += 2;
    for(int i = 0; i < 32; ++i) ntt_layer_inverse(p_ntt, 4, i*8, &ZETAS_INV[zeta_ptr]);
    zeta_ptr += 4;
    for(int i = 0; i < 16; ++i) ntt_layer_inverse(p_ntt, 8, i*16, &ZETAS_INV[zeta_ptr]);
    zeta_ptr += 8;
    for(int i = 0; i < 8; ++i) ntt_layer_inverse(p_ntt, 16, i*32, &ZETAS_INV[zeta_ptr]);
    zeta_ptr += 16;
    
    ntt_layer_inverse(p_ntt, 32, 0, &ZETAS_INV[zeta_ptr]);
    ntt_layer_inverse(p_ntt, 32, 64, &ZETAS_INV[zeta_ptr]);
    ntt_layer_inverse(p_ntt, 32, 128, &ZETAS_INV[zeta_ptr]);
    ntt_layer_inverse(p_ntt, 32, 192, &ZETAS_INV[zeta_ptr]);
    zeta_ptr += 32;

    ntt_layer_inverse(p_ntt, 64, 0, &ZETAS_INV[zeta_ptr]);
    ntt_layer_inverse(p_ntt, 64, 128, &ZETAS_INV[zeta_ptr]);
    zeta_ptr += 64;

    ntt_layer_inverse(p_ntt, 128, 0, &ZETAS_INV[zeta_ptr]);

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