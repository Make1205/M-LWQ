// src/ntt.cpp
#include "ntt.hpp"
#include <vector>
#include <cstdint>
#include <algorithm>

namespace ntt {

    // 使用 int32_t 以匹配 mlwq 的 poly 定义
    using i32 = int32_t;
    using i16 = int16_t;

    std::vector<i16> zetas(128);
    
    // Kyber constants
    // Q = 3329, ZETA = 17
    // Barrett Reduction: 2^26 / 3329 = 20159
    const i32 Q_BARRETT_MUL = 20159; 
    const i32 Q_BARRETT_SHIFT = 26;
    const i16 ZETA = 17;

    // --- 快速模约简 (Barrett Reduction) ---
    inline i16 barrett_reduce(i32 a) {
        i32 v = ((int64_t)a * Q_BARRETT_MUL) >> Q_BARRETT_SHIFT;
        v = a - v * params::Q;
        return (i16)v;
    }
    
    // 模幂运算
    i16 mod_pow(i16 base, i16 exp) {
        i32 res = 1; i32 b = base;
        while (exp > 0) {
            if (exp & 1) res = barrett_reduce(res * b);
            b = barrett_reduce(b * b);
            exp >>= 1;
        }
        return (i16)res;
    }

    i16 mod_inv(i16 a) { return mod_pow(a, params::Q - 2); }

    // Bit Reversal
    uint8_t bitrev7(uint8_t n) {
        uint8_t r = 0;
        for(int i=0; i<7; i++) if((n >> i) & 1) r |= (1 << (6-i));
        return r;
    }

    void init_tables() {
        static bool initialized = false;
        if (initialized) return;
        for(int i = 0; i < 128; i++) zetas[i] = mod_pow(ZETA, bitrev7(i));
        initialized = true;
    }

    // --- NTT 实现 (In-Place) ---
    void ntt_forward(std::vector<i32>& a) {
        int k = 1;
        for (int len = 128; len >= 2; len >>= 1) {
            for (int start = 0; start < params::N; start += 2 * len) {
                i16 zeta = zetas[k++];
                for (int j = start; j < start + len; j++) {
                    i16 t = barrett_reduce(zeta * a[j + len]);
                    a[j + len] = barrett_reduce(a[j] - t);
                    a[j] = barrett_reduce(a[j] + t);
                }
            }
        }
    }

    void ntt_inverse(std::vector<i32>& a) {
        for (int len = 2; len <= 128; len <<= 1) {
            int k_start = 128 / len;
            for (int start = 0; start < params::N; start += 2 * len) {
                int k = k_start + (start / (2 * len));
                i16 zeta = zetas[k];
                i16 zeta_inv = mod_inv(zeta); 
                for (int j = start; j < start + len; j++) {
                    i16 t = a[j];
                    a[j] = barrett_reduce(t + a[j + len]);
                    i32 diff = t - a[j + len];
                    a[j + len] = barrett_reduce(diff * (i32)zeta_inv);
                }
            }
        }
        i16 f = mod_inv(128);
        for(int i=0; i<params::N; i++) a[i] = barrett_reduce(a[i] * f);
    }

    // Kyber 风格 Base Multiplication (Pointwise in NTT domain)
    void basemul(std::vector<i32>& r, const std::vector<i32>& a, const std::vector<i32>& b) {
        for (int i = 0; i < params::N / 4; i++) {
            i16 zeta = zetas[64 + i];
            
            // Block 1
            i32 a0 = a[4*i], a1 = a[4*i+1];
            i32 b0 = b[4*i], b1 = b[4*i+1];
            r[4*i]   = barrett_reduce(a0*b0 + barrett_reduce(a1*b1)*zeta);
            r[4*i+1] = barrett_reduce(a0*b1 + a1*b0);

            // Block 2
            i32 a2 = a[4*i+2], a3 = a[4*i+3];
            i32 b2 = b[4*i+2], b3 = b[4*i+3];
            i16 m_zeta = barrett_reduce(-zeta);
            r[4*i+2] = barrett_reduce(a2*b2 + barrett_reduce(a3*b3)*m_zeta);
            r[4*i+3] = barrett_reduce(a2*b3 + a3*b2);
        }
    }

    // --- 公共接口 ---
    std::vector<i32> poly_mul_ntt(const std::vector<i32>& a, const std::vector<i32>& b) {
        // 1. 确保表已初始化
        init_tables();

        // 2. 复制数据
        std::vector<i32> fa = a;
        std::vector<i32> fb = b;
        std::vector<i32> res(params::N);

        // 3. NTT 变换
        ntt_forward(fa);
        ntt_forward(fb);

        // 4. 点乘 (BaseMul)
        basemul(res, fa, fb);

        // 5. 逆变换
        ntt_inverse(res);

        // 6. 标准化到 [0, Q) (Important for M-LWQ logic!)
        for(int i=0; i<params::N; i++) {
            i16 v = res[i];
            res[i] = (v < 0) ? v + params::Q : v;
        }

        return res;
    }
}