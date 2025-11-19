#include "poly.hpp"
#include "ntt.hpp"
#include <iostream>
#include <cmath>
#include <stdexcept>
#include <numeric>
#include <array>
#include <immintrin.h> // AVX2

// === 基础模运算 ===
int16_t positive_mod(int64_t val, int16_t q) {
    int16_t res = val % q;
    return (res < 0) ? (res + q) : static_cast<int16_t>(res);
}

// ==========================================
// AVX2 Helper Functions (手动 AVX2 实现)
// ==========================================

poly poly_add_avx(const poly& a, const poly& b) {
    poly res(params::N);
    for (int i = 0; i < params::N; i += 16) { 
        __m256i va = _mm256_loadu_si256((__m256i*)&a[i]);
        __m256i vb = _mm256_loadu_si256((__m256i*)&b[i]);
        __m256i vsum = _mm256_add_epi16(va, vb); 
        _mm256_storeu_si256((__m256i*)&res[i], vsum);
    }
    for(int i=0; i<params::N; ++i) if (res[i] >= params::Q) res[i] -= params::Q;
    return res;
}

poly poly_sub_avx(const poly& a, const poly& b) {
    poly res(params::N);
    for (int i = 0; i < params::N; i += 16) { 
        __m256i va = _mm256_loadu_si256((__m256i*)&a[i]);
        __m256i vb = _mm256_loadu_si256((__m256i*)&b[i]);
        __m256i vsub = _mm256_sub_epi16(va, vb); 
        _mm256_storeu_si256((__m256i*)&res[i], vsub);
    }
    for(int i=0; i<params::N; ++i) if (res[i] < 0) res[i] += params::Q;
    return res;
}

poly poly_quantize_avx_explicit(const poly& val, const poly& d, int32_t P_param) {
    poly res(params::N);
    float scale_val = (float)P_param / (float)params::Q;
    __m256 v_scale = _mm256_set1_ps(scale_val);
    __m256i v_mask = _mm256_set1_epi32(P_param - 1); // P needs to be 2^k

    for (int i = 0; i < params::N; i += 8) {
        __m128i v_val_128 = _mm_loadu_si128((__m128i*)&val[i]);
        __m128i v_d_128   = _mm_loadu_si128((__m128i*)&d[i]);
        __m256i v_val = _mm256_cvtepi16_epi32(v_val_128);
        __m256i v_d   = _mm256_cvtepi16_epi32(v_d_128);
        __m256i v_sum = _mm256_add_epi32(v_val, v_d);
        __m256 v_sum_f = _mm256_cvtepi32_ps(v_sum);
        __m256 v_res_f = _mm256_mul_ps(v_sum_f, v_scale);
        __m256 v_floor_f = _mm256_floor_ps(v_res_f);
        __m256i v_res_i32 = _mm256_cvtps_epi32(v_floor_f);
        v_res_i32 = _mm256_and_si256(v_res_i32, v_mask);
        
        int32_t tmp[8];
        _mm256_storeu_si256((__m256i*)tmp, v_res_i32);
        for(int k=0; k<8; ++k) res[i+k] = (int16_t)tmp[k];
    }
    return res;
}

// [新增] AVX2 反量化
// res = (b * Q) / P
poly poly_dequantize_avx(const poly& b, int32_t P_param) {
    poly res(params::N);
    float scale_val = (float)params::Q / (float)P_param;
    __m256 v_scale = _mm256_set1_ps(scale_val);

    for (int i = 0; i < params::N; i += 8) {
        __m128i v_b_128 = _mm_loadu_si128((__m128i*)&b[i]);
        __m256i v_b = _mm256_cvtepi16_epi32(v_b_128);
        __m256 v_b_f = _mm256_cvtepi32_ps(v_b);
        __m256 v_res_f = _mm256_mul_ps(v_b_f, v_scale);
        
        // round: floor(x + 0.5)
        __m256 v_round_f = _mm256_floor_ps(_mm256_add_ps(v_res_f, _mm256_set1_ps(0.5f)));
        __m256i v_res_i32 = _mm256_cvtps_epi32(v_round_f);
        
        // Store
        int32_t tmp[8];
        _mm256_storeu_si256((__m256i*)tmp, v_res_i32);
        for(int k=0; k<8; ++k) res[i+k] = positive_mod(tmp[k], params::Q);
    }
    return res;
}

// [新增] AVX2 消息解码
// res = round( (val * 2) / Q ) mod 2
poly poly_message_decode_avx(const poly& val) {
    poly res(params::N);
    // Logic:
    // centered = val > Q/2 ? val - Q : val
    // scaled = centered * (2/Q)
    // round(scaled) mod 2
    
    // 简化逻辑用于 2/Q:
    // 直接观察 val. 如果 val 在 [-Q/4, Q/4] -> 0, else -> 1
    // 即: 如果 val \in [0, Q/4] or [3Q/4, Q) -> 0
    //     如果 val \in [Q/4, 3Q/4] -> 1
    
    int16_t q_4 = params::Q / 4;
    int16_t q_34 = (params::Q * 3) / 4;
    
    __m256i v_q4 = _mm256_set1_epi16(q_4);
    __m256i v_q34 = _mm256_set1_epi16(q_34);
    __m256i v_one = _mm256_set1_epi16(1);
    __m256i v_zero = _mm256_setzero_si256();

    for (int i = 0; i < params::N; i += 16) {
        __m256i v = _mm256_loadu_si256((__m256i*)&val[i]);
        
        // mask1: v > q_4  (signed compare works if q_4 positive)
        __m256i m1 = _mm256_cmpgt_epi16(v, v_q4);
        // mask2: v < q_34 (use gt: q_34 > v)
        __m256i m2 = _mm256_cmpgt_epi16(v_q34, v);
        
        // res = m1 & m2 (即 v > Q/4 AND v < 3Q/4)
        __m256i mask = _mm256_and_si256(m1, m2);
        
        // mask is 0xFFFF (-1) if true, 0 if false.
        // result = 1 & mask = (1 if true, 0 if false) -- but mask is -1.
        // -1 is 1111...1.  1 & 111...1 = 1.
        // Wait, we want result to be 0 or 1.
        // _mm256_and_si256(v_one, mask) -> 1 if true, 0 if false.
        
        __m256i r = _mm256_and_si256(v_one, mask);
        _mm256_storeu_si256((__m256i*)&res[i], r);
    }
    return res;
}

// ==========================================
// 多项式运算 (mod Q)
// ==========================================

poly poly_add(const poly& a, const poly& b) {
    if (params::USE_AVX2) return poly_add_avx(a, b);
    poly res(params::N);
    for (size_t i = 0; i < params::N; ++i) res[i] = positive_mod((int32_t)a[i] + b[i], params::Q);
    return res;
}

poly poly_sub(const poly& a, const poly& b) {
    if (params::USE_AVX2) return poly_sub_avx(a, b);
    poly res(params::N);
    for (size_t i = 0; i < params::N; ++i) res[i] = positive_mod((int32_t)a[i] - b[i], params::Q);
    return res;
}

poly poly_mul_mod(const poly& a, const poly& b) {
    return ntt::poly_mul_ntt(a, b);
}

// ==========================================
// 向量/矩阵运算
// ==========================================
// (保持不变，会自动调用上面的 add/mul)
poly_vec poly_vec_add(const poly_vec& a, const poly_vec& b) {
    if (a.size() != b.size()) throw std::runtime_error("size mismatch");
    poly_vec res(a.size());
    for (size_t i = 0; i < a.size(); ++i) res[i] = poly_add(a[i], b[i]);
    return res;
}
poly_vec poly_vec_sub(const poly_vec& a, const poly_vec& b) {
    if (a.size() != b.size()) throw std::runtime_error("size mismatch");
    poly_vec res(a.size());
    for (size_t i = 0; i < a.size(); ++i) res[i] = poly_sub(a[i], b[i]);
    return res;
}
poly_vec poly_matrix_vec_mul(const poly_matrix& A, const poly_vec& s) {
    poly_vec res(params::K);
    for(int i = 0; i < params::K; ++i) {
        poly acc(params::N, 0);
        for(int j = 0; j < params::K; ++j) {
            poly prod = poly_mul_mod(A[i][j], s[j]);
            acc = poly_add(acc, prod);
        }
        res[i] = acc;
    }
    return res;
}
poly poly_vec_transpose_mul(const poly_vec& a_t, const poly_vec& b) {
    poly res(params::N, 0);
    for(int i = 0; i < params::K; ++i) {
        poly prod = poly_mul_mod(a_t[i], b[i]);
        res = poly_add(res, prod);
    }
    return res;
}
poly_matrix poly_matrix_transpose(const poly_matrix& A) {
    poly_matrix A_t(params::K, poly_vec(params::K));
    for (int i = 0; i < params::K; ++i) {
        for (int j = 0; j < params::K; ++j) A_t[j][i] = A[i][j];
    }
    return A_t;
}

// ==========================================
// 量化部分 (适配更新)
// ==========================================

// D8 代码 (省略，保持不变)
void quantize_d8_block(std::array<int16_t, 8>& b_block, const std::array<int16_t, 8>& val_block, const std::array<int16_t, 8>& d_block, int32_t P_param) {
    const double scale_pq = static_cast<double>(P_param) / params::Q;
    std::array<double, 8> x_scaled;
    std::array<int16_t, 8> z_rounded;
    int32_t sum = 0;
    for (int i = 0; i < 8; ++i) {
        int16_t added_val = positive_mod((int32_t)val_block[i] + d_block[i], params::Q);
        x_scaled[i] = static_cast<double>(added_val) * scale_pq;
        z_rounded[i] = static_cast<int16_t>(std::round(x_scaled[i]));
        sum += z_rounded[i];
    }
    if (sum % 2 == 0) {
        for (int i = 0; i < 8; ++i) b_block[i] = positive_mod(z_rounded[i], (int16_t)P_param);
        return;
    }
    int j_max_dist = 0; double max_dist = -1.0;
    for (int i = 0; i < 8; ++i) {
        double dist = std::fabs(x_scaled[i] - z_rounded[i]);
        if (dist > max_dist) { max_dist = dist; j_max_dist = i; }
    }
    if (x_scaled[j_max_dist] > z_rounded[j_max_dist]) z_rounded[j_max_dist] += 1;
    else z_rounded[j_max_dist] -= 1;
    for (int i = 0; i < 8; ++i) b_block[i] = positive_mod(z_rounded[i], (int16_t)P_param);
}

poly poly_quantize(const poly& val, const poly& d, int32_t P_param) {
    if (params::USE_AVX2 && params::Q_MODE == params::QUANT_SCALAR) {
        return poly_quantize_avx_explicit(val, d, P_param);
    }
    // Scalar Fallback
    poly b(params::N);
    if constexpr (params::Q_MODE == params::QUANT_SCALAR) {
        const double scale_pq = static_cast<double>(P_param) / params::Q;
        for(int i = 0; i < params::N; ++i) {
            int16_t added_val = positive_mod((int32_t)val[i] + d[i], params::Q);
            double scaled_val = scale_pq * added_val;
            int16_t floored_val = static_cast<int16_t>(std::floor(scaled_val));
            b[i] = positive_mod(floored_val, (int16_t)P_param);
        }
    } else if constexpr (params::Q_MODE == params::QUANT_D8) {
        for (int i = 0; i < params::N; i += 8) {
            std::array<int16_t, 8> val_block, d_block, b_block;
            for (int j = 0; j < 8; ++j) { val_block[j] = val[i + j]; d_block[j] = d[i + j]; }
            quantize_d8_block(b_block, val_block, d_block, P_param);
            for (int j = 0; j < 8; ++j) b[i + j] = b_block[j];
        }
    }
    return b;
}

poly_vec poly_vec_quantize(const poly_vec& val, const poly_vec& d, int32_t P_param) {
    poly_vec res(val.size());
    for(size_t i = 0; i < val.size(); ++i) res[i] = poly_quantize(val[i], d[i], P_param);
    return res;
}

poly poly_dequantize(const poly& b, int32_t P_param) {
    // [修改] 使用 AVX2
    if (params::USE_AVX2) return poly_dequantize_avx(b, P_param);
    
    poly res(params::N);
    const double scale_qp = static_cast<double>(params::Q) / P_param;
    for(int i = 0; i < params::N; ++i) {
        double scaled_val = scale_qp * b[i];
        int16_t rounded_val = static_cast<int16_t>(std::round(scaled_val));
        res[i] = positive_mod(rounded_val, params::Q);
    }
    return res;
}

poly_vec poly_vec_dequantize(const poly_vec& b, int32_t P_param) {
    poly_vec res(b.size());
    for(size_t i = 0; i < b.size(); ++i) res[i] = poly_dequantize(b[i], P_param);
    return res;
}

poly poly_message_encode(const poly& m) {
    poly res(params::N);
    const int32_t scale = params::Q / params::MSG_MODULUS;
    for (int i = 0; i < params::N; ++i) res[i] = positive_mod((int32_t)m[i] * scale, params::Q);
    return res;
}

poly poly_message_decode(const poly& val) {
    // [修改] 使用 AVX2
    if (params::USE_AVX2) return poly_message_decode_avx(val);

    poly res(params::N);
    const double scale = static_cast<double>(params::MSG_MODULUS) / params::Q;
    const int32_t q_half = params::Q / 2;
    for (int i = 0; i < params::N; ++i) {
        int16_t centered_val = positive_mod(val[i], params::Q);
        if (centered_val > q_half) centered_val -= params::Q;
        double scaled_val = static_cast<double>(centered_val) * scale;
        int16_t rounded_val = static_cast<int16_t>(std::round(scaled_val));
        res[i] = positive_mod(rounded_val, (int16_t)params::MSG_MODULUS);
    }
    return res;
}

// 辅助函数 (Print/Check) ...
void print_poly(const std::string& name, const poly& p, size_t count) {
    std::cout << "  " << name << " [";
    size_t n = std::min(count, p.size());
    for(size_t i = 0; i < n; ++i) std::cout << p[i] << (i==n-1?"":", ");
    std::cout << (p.size()>n ? ", ...]" : "]") << std::endl;
}
void print_poly_vec(const std::string& name, const poly_vec& pv, size_t count) {
    std::cout << "  " << name << " (" << pv.size() << "):" << std::endl;
    for(size_t i=0; i<pv.size(); ++i) print_poly(" ["+std::to_string(i)+"]", pv[i], count);
}
bool check_poly_eq(const poly& a, const poly& b) {
    if (a.size() != b.size()) return false;
    for (size_t i = 0; i < a.size(); ++i) if (a[i] != b[i]) return false;
    return true;
}