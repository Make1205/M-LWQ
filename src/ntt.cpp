// 最终完整的 ntt.cpp (包含标量和AVX路径)
// 请使用此代码覆盖
#include "ntt.hpp"
#include <vector>
#include <algorithm>
#include <immintrin.h>

namespace ntt {
    using i16 = int16_t;
    using i32 = int32_t;
    std::vector<i16> zetas(128);
    alignas(32) i16 zetas_avx_storage[16 * 16]; 
    const i16 Q = 3329;
    const i16 ZETA = 17;
    const i32 Q_BARRETT_MUL = 20159; 
    const i32 Q_BARRETT_SHIFT = 26;

    inline i16 barrett_reduce(i32 a) {
        i32 v = ((int64_t)a * Q_BARRETT_MUL) >> Q_BARRETT_SHIFT;
        v = a - v * Q;
        return (i16)v;
    }
    i16 mod_pow(i16 base, i16 exp) {
        i32 res = 1; i32 b = base;
        while (exp > 0) {
            if (exp & 1) res = barrett_reduce(res * b);
            b = barrett_reduce(b * b);
            exp >>= 1;
        }
        return (i16)res;
    }
    uint8_t bitrev7(uint8_t n) {
        uint8_t r = 0;
        for(int i=0; i<7; i++) if((n >> i) & 1) r |= (1 << (6-i));
        return r;
    }
    void init_tables() {
        static bool initialized = false;
        if (initialized) return;
        for(int i = 0; i < 128; i++) zetas[i] = mod_pow(ZETA, bitrev7(i));
        for(int i = 0; i < 16; i++) { 
            i16* vec_ptr = &zetas_avx_storage[i * 16];
            i16 z0 = zetas[64 + 4*i + 0]; i16 z1 = zetas[64 + 4*i + 1];
            i16 z2 = zetas[64 + 4*i + 2]; i16 z3 = zetas[64 + 4*i + 3];
            vec_ptr[0] = z0; vec_ptr[1] = z0; vec_ptr[2] = -z0; vec_ptr[3] = -z0;
            vec_ptr[4] = z1; vec_ptr[5] = z1; vec_ptr[6] = -z1; vec_ptr[7] = -z1;
            vec_ptr[8] = z2; vec_ptr[9] = z2; vec_ptr[10] = -z2; vec_ptr[11] = -z2;
            vec_ptr[12] = z3; vec_ptr[13] = z3; vec_ptr[14] = -z3; vec_ptr[15] = -z3;
        }
        initialized = true;
    }

    inline __m256i vec_mul_mod_safe(__m256i a, __m256i b) {
        __m128i a_lo = _mm256_castsi256_si128(a);
        __m128i a_hi = _mm256_extracti128_si256(a, 1);
        __m128i b_lo = _mm256_castsi256_si128(b);
        __m128i b_hi = _mm256_extracti128_si256(b, 1);
        __m256i a_lo_32 = _mm256_cvtepi16_epi32(a_lo);
        __m256i a_hi_32 = _mm256_cvtepi16_epi32(a_hi);
        __m256i b_lo_32 = _mm256_cvtepi16_epi32(b_lo);
        __m256i b_hi_32 = _mm256_cvtepi16_epi32(b_hi);
        __m256i prod_lo = _mm256_mullo_epi32(a_lo_32, b_lo_32);
        __m256i prod_hi = _mm256_mullo_epi32(a_hi_32, b_hi_32);
        __m256i v_q = _mm256_set1_epi32(Q);
        auto reduce = [&](__m256i val) {
            __m256 vf = _mm256_cvtepi32_ps(val);
            __m256 iq = _mm256_set1_ps(1.0f / 3329.0f);
            __m256i q = _mm256_cvttps_epi32(_mm256_mul_ps(vf, iq));
            __m256i r = _mm256_sub_epi32(val, _mm256_mullo_epi32(q, v_q));
            __m256i m1 = _mm256_cmpgt_epi32(_mm256_setzero_si256(), r);
            r = _mm256_add_epi32(r, _mm256_and_si256(m1, v_q));
            __m256i m2 = _mm256_cmpgt_epi32(r, _mm256_sub_epi32(v_q, _mm256_set1_epi32(1)));
            r = _mm256_sub_epi32(r, _mm256_and_si256(m2, v_q));
            return r;
        };
        __m256i rl = reduce(prod_lo);
        __m256i rh = reduce(prod_hi);
        return _mm256_permute4x64_epi64(_mm256_packus_epi32(rl, rh), 0xD8);
    }

    void ntt_forward(std::vector<i16>& a) {
        int k = 1;
        if (params::USE_AVX2) {
            const __m256i vq = _mm256_set1_epi16(Q);
            for (int len = 128; len >= 16; len >>= 1) {
                for (int start = 0; start < params::N; start += 2 * len) {
                    i16 zeta = zetas[k++];
                    __m256i v_zeta = _mm256_set1_epi16(zeta);
                    for (int j = start; j < start + len; j += 16) {
                        __m256i aj = _mm256_loadu_si256((__m256i*)&a[j]);
                        __m256i aj_len = _mm256_loadu_si256((__m256i*)&a[j + len]);
                        __m256i t = vec_mul_mod_safe(aj_len, v_zeta);
                        __m256i v_sub = _mm256_sub_epi16(aj, t);
                        __m256i mask_neg = _mm256_cmpgt_epi16(_mm256_setzero_si256(), v_sub);
                        v_sub = _mm256_add_epi16(v_sub, _mm256_and_si256(mask_neg, vq));
                        __m256i v_add = _mm256_add_epi16(aj, t);
                        __m256i v_limit = _mm256_sub_epi16(vq, _mm256_set1_epi16(1));
                        __m256i mask_ge = _mm256_cmpgt_epi16(v_add, v_limit);
                        v_add = _mm256_sub_epi16(v_add, _mm256_and_si256(mask_ge, vq));
                        _mm256_storeu_si256((__m256i*)&a[j + len], v_sub);
                        _mm256_storeu_si256((__m256i*)&a[j], v_add);
                    }
                }
            }
            for (int len = 8; len >= 2; len >>= 1) {
                for (int start = 0; start < params::N; start += 2 * len) {
                    i16 zeta = zetas[k++];
                    for (int j = start; j < start + len; j++) {
                        i16 t = barrett_reduce((i32)zeta * a[j + len]);
                        a[j + len] = barrett_reduce(a[j] - t);
                        a[j] = barrett_reduce(a[j] + t);
                    }
                }
            }
        } else {
            // Pure Scalar
            for (int len = 128; len >= 2; len >>= 1) {
                for (int start = 0; start < params::N; start += 2 * len) {
                    i16 zeta = zetas[k++];
                    for (int j = start; j < start + len; j++) {
                        i16 t = barrett_reduce((i32)zeta * a[j + len]);
                        a[j + len] = barrett_reduce(a[j] - t);
                        a[j] = barrett_reduce(a[j] + t);
                    }
                }
            }
        }
    }

    void ntt_inverse(std::vector<i16>& a) {
        if (params::USE_AVX2) {
            const __m256i vq = _mm256_set1_epi16(Q);
            for (int len = 2; len <= 8; len <<= 1) {
                int k_start = 128 / len;
                for (int start = 0; start < params::N; start += 2 * len) {
                    int k = k_start + (start / (2 * len));
                    i16 zeta = zetas[k];
                    i16 zeta_inv = mod_pow(zeta, Q - 2);
                    for (int j = start; j < start + len; j++) {
                        i16 t = a[j];
                        a[j] = barrett_reduce(t + a[j + len]);
                        i32 diff = t - a[j + len];
                        a[j + len] = barrett_reduce(diff * (i32)zeta_inv);
                    }
                }
            }
            for (int len = 16; len <= 128; len <<= 1) {
                int k_start = 128 / len;
                for (int start = 0; start < params::N; start += 2 * len) {
                    int k = k_start + (start / (2 * len));
                    i16 zeta = zetas[k];
                    i16 zeta_inv = mod_pow(zeta, Q - 2);
                    __m256i v_zeta_inv = _mm256_set1_epi16(zeta_inv);
                    for (int j = start; j < start + len; j += 16) {
                        __m256i aj = _mm256_loadu_si256((__m256i*)&a[j]);
                        __m256i aj_len = _mm256_loadu_si256((__m256i*)&a[j + len]);
                        __m256i v_add = _mm256_add_epi16(aj, aj_len);
                        __m256i v_limit = _mm256_sub_epi16(vq, _mm256_set1_epi16(1));
                        __m256i mask_ge = _mm256_cmpgt_epi16(v_add, v_limit);
                        v_add = _mm256_sub_epi16(v_add, _mm256_and_si256(mask_ge, vq));
                        __m256i v_sub = _mm256_sub_epi16(aj, aj_len);
                        __m256i mask_neg = _mm256_cmpgt_epi16(_mm256_setzero_si256(), v_sub);
                        v_sub = _mm256_add_epi16(v_sub, _mm256_and_si256(mask_neg, vq));
                        __m256i v_mul = vec_mul_mod_safe(v_sub, v_zeta_inv);
                        _mm256_storeu_si256((__m256i*)&a[j], v_add);
                        _mm256_storeu_si256((__m256i*)&a[j + len], v_mul);
                    }
                }
            }
            i16 f = mod_pow(128, Q - 2);
            __m256i v_f = _mm256_set1_epi16(f);
            for(int i = 0; i < params::N; i += 16) {
                __m256i val = _mm256_loadu_si256((__m256i*)&a[i]);
                __m256i scaled = vec_mul_mod_safe(val, v_f);
                _mm256_storeu_si256((__m256i*)&a[i], scaled);
            }
        } else {
            // Scalar Inverse
            for (int len = 2; len <= 128; len <<= 1) {
                int k_start = 128 / len;
                for (int start = 0; start < params::N; start += 2 * len) {
                    int k = k_start + (start / (2 * len));
                    i16 zeta = zetas[k];
                    i16 zeta_inv = mod_pow(zeta, Q - 2);
                    for (int j = start; j < start + len; j++) {
                        i16 t = a[j];
                        a[j] = barrett_reduce(t + a[j + len]);
                        i32 diff = t - a[j + len];
                        a[j + len] = barrett_reduce(diff * (i32)zeta_inv);
                    }
                }
            }
            i16 f = mod_pow(128, Q - 2);
            for(int i=0; i<params::N; i++) a[i] = barrett_reduce((i32)a[i] * f);
        }
    }

    void basemul_avx(std::vector<i16>& r, const std::vector<i16>& a, const std::vector<i16>& b) {
        const __m256i vq = _mm256_set1_epi16(Q);
        for (int i = 0; i < params::N / 32; i++) {
            int idx1 = 32 * i;
            __m256i va1 = _mm256_loadu_si256((__m256i*)&a[idx1]);
            __m256i vb1 = _mm256_loadu_si256((__m256i*)&b[idx1]);
            __m256i v_z1 = _mm256_load_si256((__m256i*)&zetas_avx_storage[(2*i) * 16]);
            __m256i prod1_1 = vec_mul_mod_safe(va1, vb1);
            __m256i va_swap1 = _mm256_shuffle_epi8(va1, _mm256_setr_epi8(2,3, 0,1, 6,7, 4,5, 10,11, 8,9, 14,15, 12,13, 18,19, 16,17, 22,23, 20,21, 26,27, 24,25, 30,31, 28,29));
            __m256i prod2_1 = vec_mul_mod_safe(va_swap1, vb1);
            __m256i prod1_swap1 = _mm256_shuffle_epi8(prod1_1, _mm256_setr_epi8(2,3, 0,1, 6,7, 4,5, 10,11, 8,9, 14,15, 12,13, 18,19, 16,17, 22,23, 20,21, 26,27, 24,25, 30,31, 28,29));
            __m256i term_zeta1 = vec_mul_mod_safe(prod1_swap1, v_z1);
            __m256i r_even1 = _mm256_add_epi16(prod1_1, term_zeta1);
            __m256i prod2_swap1 = _mm256_shuffle_epi8(prod2_1, _mm256_setr_epi8(2,3, 0,1, 6,7, 4,5, 10,11, 8,9, 14,15, 12,13, 18,19, 16,17, 22,23, 20,21, 26,27, 24,25, 30,31, 28,29));
            __m256i r_odd1 = _mm256_add_epi16(prod2_1, prod2_swap1);
            __m256i res1 = _mm256_blend_epi16(r_even1, r_odd1, 0xAA);
            __m256i mask1 = _mm256_cmpgt_epi16(res1, _mm256_sub_epi16(vq, _mm256_set1_epi16(1)));
            res1 = _mm256_sub_epi16(res1, _mm256_and_si256(mask1, vq));
            _mm256_storeu_si256((__m256i*)&r[idx1], res1);

            int idx2 = 32 * i + 16;
            __m256i va2 = _mm256_loadu_si256((__m256i*)&a[idx2]);
            __m256i vb2 = _mm256_loadu_si256((__m256i*)&b[idx2]);
            __m256i v_z2 = _mm256_load_si256((__m256i*)&zetas_avx_storage[(2*i + 1) * 16]);
            __m256i prod1_2 = vec_mul_mod_safe(va2, vb2);
            __m256i va_swap2 = _mm256_shuffle_epi8(va2, _mm256_setr_epi8(2,3, 0,1, 6,7, 4,5, 10,11, 8,9, 14,15, 12,13, 18,19, 16,17, 22,23, 20,21, 26,27, 24,25, 30,31, 28,29));
            __m256i prod2_2 = vec_mul_mod_safe(va_swap2, vb2);
            __m256i prod1_swap2 = _mm256_shuffle_epi8(prod1_2, _mm256_setr_epi8(2,3, 0,1, 6,7, 4,5, 10,11, 8,9, 14,15, 12,13, 18,19, 16,17, 22,23, 20,21, 26,27, 24,25, 30,31, 28,29));
            __m256i term_zeta2 = vec_mul_mod_safe(prod1_swap2, v_z2);
            __m256i r_even2 = _mm256_add_epi16(prod1_2, term_zeta2);
            __m256i prod2_swap2 = _mm256_shuffle_epi8(prod2_2, _mm256_setr_epi8(2,3, 0,1, 6,7, 4,5, 10,11, 8,9, 14,15, 12,13, 18,19, 16,17, 22,23, 20,21, 26,27, 24,25, 30,31, 28,29));
            __m256i r_odd2 = _mm256_add_epi16(prod2_2, prod2_swap2);
            __m256i res2 = _mm256_blend_epi16(r_even2, r_odd2, 0xAA);
            __m256i mask2 = _mm256_cmpgt_epi16(res2, _mm256_sub_epi16(vq, _mm256_set1_epi16(1)));
            res2 = _mm256_sub_epi16(res2, _mm256_and_si256(mask2, vq));
            _mm256_storeu_si256((__m256i*)&r[idx2], res2);
        }
    }
    void basemul(std::vector<i16>& r, const std::vector<i16>& a, const std::vector<i16>& b) {
        for (int i = 0; i < params::N / 4; i++) {
            i16 zeta = zetas[64 + i];
            i32 a0 = a[4*i], a1 = a[4*i+1], b0 = b[4*i], b1 = b[4*i+1];
            r[4*i]   = barrett_reduce(a0*b0 + barrett_reduce(a1*b1)*zeta);
            r[4*i+1] = barrett_reduce(a0*b1 + a1*b0);
            i32 a2 = a[4*i+2], a3 = a[4*i+3], b2 = b[4*i+2], b3 = b[4*i+3];
            i16 m_zeta = barrett_reduce(-zeta);
            r[4*i+2] = barrett_reduce(a2*b2 + barrett_reduce(a3*b3)*m_zeta);
            r[4*i+3] = barrett_reduce(a2*b3 + a3*b2);
        }
    }
    std::vector<i16> poly_mul_ntt(const std::vector<i16>& a, const std::vector<i16>& b) {
        init_tables();
        std::vector<i16> fa = a;
        std::vector<i16> fb = b;
        std::vector<i16> res(params::N);
        ntt_forward(fa);
        ntt_forward(fb);
        if (params::USE_AVX2) basemul_avx(res, fa, fb);
        else basemul(res, fa, fb);
        ntt_inverse(res);
        for(int i=0; i<params::N; i++) {
            i16 v = res[i];
            res[i] = (v < 0) ? v + Q : v;
        }
        return res;
    }
}