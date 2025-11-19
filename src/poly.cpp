#include "poly.hpp"
#include "ntt.hpp"
#include <iostream>
#include <cmath>
#include <immintrin.h> // AVX2

// === 基础模运算 ===
int16_t positive_mod(int64_t val, int16_t q) {
    int16_t res = val % q;
    return (res < 0) ? (res + q) : static_cast<int16_t>(res);
}

// ==========================================
// AVX2 Helper Functions
// ==========================================

// [加速] 16路并行加法
poly poly_add_avx(const poly& a, const poly& b) {
    poly res(params::N);
    for (int i = 0; i < params::N; i += 16) { 
        __m256i va = _mm256_loadu_si256((__m256i*)&a[i]);
        __m256i vb = _mm256_loadu_si256((__m256i*)&b[i]);
        __m256i vsum = _mm256_add_epi16(va, vb); 
        _mm256_storeu_si256((__m256i*)&res[i], vsum);
    }
    // 简单修正 (因为 Q=3329 很小，int16 不会溢出，只需减Q)
    for(int i=0; i<params::N; ++i) if (res[i] >= params::Q) res[i] -= params::Q;
    return res;
}

// [加速] 16路并行减法
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

// [核心加速] AVX2 显式浮点量化
// 逻辑: res = floor( (val + d) * (P / Q) ) mod P
// 优势: 使用 FMA (Fused Multiply-Add) 替代整数除法，速度极快
poly poly_quantize_avx_explicit(const poly& val, const poly& d, int32_t P_param) {
    poly res(params::N);
    
    float scale_val = (float)P_param / (float)params::Q;
    __m256 v_scale = _mm256_set1_ps(scale_val);
    __m256i v_mask = _mm256_set1_epi32(P_param - 1); // P 必须是 2^k

    // 每次处理 8 个 (因为要转 float)
    for (int i = 0; i < params::N; i += 8) {
        // 1. Load & Convert to int32
        __m128i v_val_128 = _mm_loadu_si128((__m128i*)&val[i]);
        __m128i v_d_128   = _mm_loadu_si128((__m128i*)&d[i]);
        __m256i v_val = _mm256_cvtepi16_epi32(v_val_128);
        __m256i v_d   = _mm256_cvtepi16_epi32(v_d_128);
        
        // 2. Add
        __m256i v_sum = _mm256_add_epi32(v_val, v_d);
        
        // 3. Convert to float & Scale
        __m256 v_sum_f = _mm256_cvtepi32_ps(v_sum);
        __m256 v_res_f = _mm256_mul_ps(v_sum_f, v_scale);
        
        // 4. Floor & Convert back
        __m256 v_floor_f = _mm256_floor_ps(v_res_f);
        __m256i v_res_i32 = _mm256_cvtps_epi32(v_floor_f);
        
        // 5. Mod P (Bitwise AND)
        v_res_i32 = _mm256_and_si256(v_res_i32, v_mask);
        
        // 6. Store (手动 pack 以避免 shuffle 开销)
        int32_t tmp[8];
        _mm256_storeu_si256((__m256i*)tmp, v_res_i32);
        for(int k=0; k<8; ++k) res[i+k] = (int16_t)tmp[k];
    }
    return res;
}

// [加速] AVX2 反量化
poly poly_dequantize_avx(const poly& b, int32_t P_param) {
    poly res(params::N);
    float scale_val = (float)params::Q / (float)P_param;
    __m256 v_scale = _mm256_set1_ps(scale_val);
    __m256 v_half = _mm256_set1_ps(0.5f);

    for (int i = 0; i < params::N; i += 8) {
        __m128i v_b_128 = _mm_loadu_si128((__m128i*)&b[i]);
        __m256i v_b = _mm256_cvtepi16_epi32(v_b_128);
        __m256 v_b_f = _mm256_cvtepi32_ps(v_b);
        
        // x * scale
        __m256 v_res_f = _mm256_mul_ps(v_b_f, v_scale);
        // round(x) = floor(x + 0.5)
        __m256 v_round_f = _mm256_floor_ps(_mm256_add_ps(v_res_f, v_half));
        __m256i v_res_i32 = _mm256_cvtps_epi32(v_round_f);
        
        int32_t tmp[8];
        _mm256_storeu_si256((__m256i*)tmp, v_res_i32);
        for(int k=0; k<8; ++k) res[i+k] = positive_mod(tmp[k], params::Q);
    }
    return res;
}

// [加速] AVX2 解码
poly poly_message_decode_avx(const poly& val) {
    poly res(params::N);
    // 逻辑: if (val > Q/4 && val < 3Q/4) return 1 else 0
    __m256i v_q4 = _mm256_set1_epi16(params::Q / 4);
    __m256i v_q34 = _mm256_set1_epi16((params::Q * 3) / 4);
    __m256i v_one = _mm256_set1_epi16(1);

    for (int i = 0; i < params::N; i += 16) {
        __m256i v = _mm256_loadu_si256((__m256i*)&val[i]);
        __m256i m1 = _mm256_cmpgt_epi16(v, v_q4);
        __m256i m2 = _mm256_cmpgt_epi16(v_q34, v);
        __m256i mask = _mm256_and_si256(m1, m2);
        __m256i r = _mm256_and_si256(v_one, mask);
        _mm256_storeu_si256((__m256i*)&res[i], r);
    }
    return res;
}

// ==========================================
// 公共接口 (Dispatcher)
// ==========================================

poly poly_add(const poly& a, const poly& b) {
    if (params::USE_AVX2) return poly_add_avx(a, b);
    // Scalar Fallback
    poly res(params::N);
    for (size_t i = 0; i < params::N; ++i) res[i] = positive_mod((int32_t)a[i] + b[i], params::Q);
    return res;
}

poly poly_sub(const poly& a, const poly& b) {
    if (params::USE_AVX2) return poly_sub_avx(a, b);
    // Scalar Fallback
    poly res(params::N);
    for (size_t i = 0; i < params::N; ++i) res[i] = positive_mod((int32_t)a[i] - b[i], params::Q);
    return res;
}

poly poly_mul_mod(const poly& a, const poly& b) {
    return ntt::poly_mul_ntt(a, b);
}

poly poly_quantize(const poly& val, const poly& d, int32_t P_param) {
    // 始终优先使用 AVX2 标量量化 (最快且正确)
    if (params::USE_AVX2) {
        return poly_quantize_avx_explicit(val, d, P_param);
    }
    // Scalar Fallback
    poly b(params::N);
    const double scale_pq = static_cast<double>(P_param) / params::Q;
    for(int i = 0; i < params::N; ++i) {
        int16_t added_val = positive_mod((int32_t)val[i] + d[i], params::Q);
        double scaled_val = scale_pq * added_val;
        int16_t floored_val = static_cast<int16_t>(std::floor(scaled_val));
        b[i] = positive_mod(floored_val, (int16_t)P_param);
    }
    return b;
}

// 封装函数保持不变
poly_vec poly_vec_quantize(const poly_vec& val, const poly_vec& d, int32_t P_param) {
    poly_vec res(val.size());
    for(size_t i = 0; i < val.size(); ++i) res[i] = poly_quantize(val[i], d[i], P_param);
    return res;
}
poly poly_dequantize(const poly& b, int32_t P_param) {
    if (params::USE_AVX2) return poly_dequantize_avx(b, P_param);
    // Scalar Fallback
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
    if (params::USE_AVX2) return poly_message_decode_avx(val);
    // Scalar Fallback
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

// 辅助函数
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
void print_poly(const std::string& name, const poly& p, size_t count) { /*...*/ }
void print_poly_vec(const std::string& name, const poly_vec& pv, size_t count) { /*...*/ }
bool check_poly_eq(const poly& a, const poly& b) {
    if (a.size() != b.size()) return false;
    for (size_t i = 0; i < a.size(); ++i) if (a[i] != b[i]) return false;
    return true;
}