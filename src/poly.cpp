#include "poly.hpp"
#include "ntt.hpp"
#include <iostream>
#include <cmath>
#include <stdexcept>
#include <numeric>
#include <array>
#include <immintrin.h> // AVX2

// === 基础模运算 ===
int32_t positive_mod(int64_t val, int32_t q) {
    int64_t res = val % q;
    return (res < 0) ? (res + q) : static_cast<int32_t>(res);
}

// === AVX2 Helper Functions (手动 AVX2 实现) ===

poly poly_add_avx(const poly& a, const poly& b) {
    poly res(params::N);
    // 每次处理 8 个 32位整数
    for (int i = 0; i < params::N; i += 8) {
        __m256i va = _mm256_loadu_si256((__m256i*)&a[i]);
        __m256i vb = _mm256_loadu_si256((__m256i*)&b[i]);
        __m256i vsum = _mm256_add_epi32(va, vb); // 并行加法
        _mm256_storeu_si256((__m256i*)&res[i], vsum);
    }
    // 后处理取模 (混合模式：AVX算，Scalar修)
    // 由于输入都在 [0, Q) 范围，sum < 2Q，修正很快
    for(int i=0; i<params::N; ++i) {
        if (res[i] >= params::Q) res[i] -= params::Q;
    }
    return res;
}

poly poly_sub_avx(const poly& a, const poly& b) {
    poly res(params::N);
    for (int i = 0; i < params::N; i += 8) {
        __m256i va = _mm256_loadu_si256((__m256i*)&a[i]);
        __m256i vb = _mm256_loadu_si256((__m256i*)&b[i]);
        __m256i vsub = _mm256_sub_epi32(va, vb); // 并行减法
        _mm256_storeu_si256((__m256i*)&res[i], vsub);
    }
    // 后处理取模: 如果结果为负，加 Q
    for(int i=0; i<params::N; ++i) {
        if (res[i] < 0) res[i] += params::Q;
    }
    return res;
}

// === 多项式运算 (mod Q) ===

poly poly_add(const poly& a, const poly& b) {
    if (params::USE_AVX2) return poly_add_avx(a, b);

    // 标量 fallback
    poly res(params::N);
    for (size_t i = 0; i < params::N; ++i) {
        res[i] = positive_mod(static_cast<int64_t>(a[i]) + b[i], params::Q);
    }
    return res;
}

poly poly_sub(const poly& a, const poly& b) {
    if (params::USE_AVX2) return poly_sub_avx(a, b);

    // 标量 fallback
    poly res(params::N);
    for (size_t i = 0; i < params::N; ++i) {
        res[i] = positive_mod(static_cast<int64_t>(a[i]) - b[i], params::Q);
    }
    return res;
}

// 核心修改：使用 NTT 加速乘法
poly poly_mul_mod(const poly& a, const poly& b) {
    // params::USE_AVX2 的检查在 ntt::poly_mul_ntt 内部处理
    return ntt::poly_mul_ntt(a, b);
}

// === 向量/矩阵运算 (保持不变) ===
// 这些函数会调用上面的 poly_add/poly_mul_mod，从而自动获得加速
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

// === 量化部分 ===

void quantize_d8_block(std::array<int32_t, 8>& b_block, const std::array<int32_t, 8>& val_block, const std::array<int32_t, 8>& d_block, int32_t P_param) {
    // D8 Lattice CVP
    const double scale_pq = static_cast<double>(P_param) / params::Q;
    std::array<double, 8> x_scaled;
    std::array<int32_t, 8> z_rounded;
    int32_t sum = 0;
    for (int i = 0; i < 8; ++i) {
        int32_t added_val = positive_mod(static_cast<int64_t>(val_block[i]) + d_block[i], params::Q);
        x_scaled[i] = static_cast<double>(added_val) * scale_pq;
        z_rounded[i] = static_cast<int32_t>(std::round(x_scaled[i]));
        sum += z_rounded[i];
    }
    if (sum % 2 == 0) {
        for (int i = 0; i < 8; ++i) b_block[i] = positive_mod(z_rounded[i], P_param);
        return;
    }
    int j_max_dist = 0; double max_dist = -1.0;
    for (int i = 0; i < 8; ++i) {
        double dist = std::fabs(x_scaled[i] - z_rounded[i]);
        if (dist > max_dist) { max_dist = dist; j_max_dist = i; }
    }
    if (x_scaled[j_max_dist] > z_rounded[j_max_dist]) z_rounded[j_max_dist] += 1;
    else z_rounded[j_max_dist] -= 1;
    for (int i = 0; i < 8; ++i) b_block[i] = positive_mod(z_rounded[i], P_param);
}

poly poly_quantize(const poly& val, const poly& d, int32_t P_param) {
    poly b(params::N);
    if constexpr (params::Q_MODE == params::QUANT_SCALAR) {
        // 标量量化
        const double scale_pq = static_cast<double>(P_param) / params::Q;
        // 如果开启了 AVX2 (-mavx2)，编译器通常会自动向量化这个循环
        for(int i = 0; i < params::N; ++i) {
            int32_t added_val = positive_mod(static_cast<int64_t>(val[i]) + d[i], params::Q);
            double scaled_val = scale_pq * added_val;
            int32_t floored_val = static_cast<int32_t>(std::floor(scaled_val));
            b[i] = positive_mod(floored_val, P_param);
        }
    } else if constexpr (params::Q_MODE == params::QUANT_D8) {
        for (int i = 0; i < params::N; i += 8) {
            std::array<int32_t, 8> val_block, d_block, b_block;
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
    poly res(params::N);
    const double scale_qp = static_cast<double>(params::Q) / P_param;
    for(int i = 0; i < params::N; ++i) {
        double scaled_val = scale_qp * b[i];
        int64_t rounded_val = static_cast<int64_t>(std::round(scaled_val));
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
    for (int i = 0; i < params::N; ++i) res[i] = positive_mod(static_cast<int64_t>(m[i]) * scale, params::Q);
    return res;
}

poly poly_message_decode(const poly& val) {
    poly res(params::N);
    const double scale = static_cast<double>(params::MSG_MODULUS) / params::Q;
    const int32_t q_half = params::Q / 2;
    for (int i = 0; i < params::N; ++i) {
        int32_t centered_val = positive_mod(val[i], params::Q);
        if (centered_val > q_half) centered_val -= params::Q;
        double scaled_val = static_cast<double>(centered_val) * scale;
        int64_t rounded_val = static_cast<int64_t>(std::round(scaled_val));
        res[i] = positive_mod(rounded_val, params::MSG_MODULUS);
    }
    return res;
}

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