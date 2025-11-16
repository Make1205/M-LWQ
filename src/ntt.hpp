#pragma once
#include "poly.hpp"
#include <vector>
#include <cstdint>

/**
 * @brief Kyber/M-LWQ NTT (Number-Theoretic Transform) 实现
 * * N = 256, Q = 3329.
 * * 乘法在环 R_q = Z_q[X] / (X^256 + 1) 中进行。
 * * 我们使用 (N=256)-th  primitive root of unity zeta = 17.
 * 17^128 = -1 (mod 3329).
 * * NTT (a_poly) -> a_ntt
 * INTT(a_ntt) -> a_poly
 * * a * b (在 R_q) == INTT( NTT(a) .* NTT(b) )
 * 其中 '.*' 是按元素乘法 (pointwise multiplication)。
 */

namespace ntt {

    // --- NTT 预计算常量 (来自 Kyber) ---

    // ZETAS 包含 (zeta^br(i) mod q)
    // br(i) 是 i 的 7-bit 位逆序 (bit-reversal)
    // zeta = 17 (N=256-th root of unity)
    extern const std::vector<int32_t> ZETAS;

    // --- NTT 核心函数 ---

    /**
     * @brief 前向 NTT (多项式 -> NTT 域)
     * @param p 输入多项式 (长度 N)
     * @return 变换后的多项式 (NTT 域)
     */
    poly ntt_forward(poly p);

    /**
     * @brief 逆 NTT (NTT 域 -> 多项式)
     * @param p_ntt NTT 域的多项式 (长度 N)
     * @return 变换后的多项式 (标准系数)
     */
    poly ntt_inverse(poly p_ntt);

    /**
     * @brief 在 NTT 域中执行按元素乘法 (a .* b)
     * @param a_ntt 变换后的多项式 a
     * @param b_ntt 变换后的多项式 b
     * @return a .* b (mod q)
     */
    poly ntt_pointwise_mul(const poly& a_ntt, const poly& b_ntt);

} // namespace ntt