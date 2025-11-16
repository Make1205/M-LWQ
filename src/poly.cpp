#include "poly.hpp"
#include <iostream>
#include <cmath> // for std::floor, std::round, std::fabs
#include <stdexcept>
#include <numeric> // for std::accumulate
#include <array>

// === 基础模运算 ===

int32_t positive_mod(int64_t val, int32_t q) {
    int64_t res = val % q;
    return (res < 0) ? (res + q) : static_cast<int32_t>(res);
}

// === 多项式运算 (mod Q) ===
// (无变化)
poly poly_add(const poly& a, const poly& b) {
    poly res(params::N);
    for (size_t i = 0; i < params::N; ++i) {
        res[i] = positive_mod(static_cast<int64_t>(a[i]) + b[i], params::Q);
    }
    return res;
}
poly poly_sub(const poly& a, const poly& b) {
    poly res(params::N);
    for (size_t i = 0; i < params::N; ++i) {
        res[i] = positive_mod(static_cast<int64_t>(a[i]) - b[i], params::Q);
    }
    return res;
}
poly poly_mul_mod(const poly& a, const poly& b) {
    poly res(params::N, 0);
    for (int i = 0; i < params::N; ++i) {
        for (int j = 0; j < params::N; ++j) {
            int k = (i + j) % params::N;
            int sign = (i + j) >= params::N ? -1 : 1;
            int64_t term = static_cast<int64_t>(a[i]) * b[j];
            int64_t val = res[k] + sign * term;
            res[k] = positive_mod(val, params::Q);
        }
    }
    return res;
}

// poly poly_mul_mod(const poly& a, const poly& b) {
//     // 1. 将 a 和 b 转换到 NTT 域
//     poly a_ntt = ntt::ntt_forward(a);
//     poly b_ntt = ntt::ntt_forward(b);

//     // 2. 在 NTT 域中进行 O(N) 按元素乘法
//     poly c_ntt = ntt::ntt_pointwise_mul(a_ntt, b_ntt);

//     // 3. 将结果 c 转换回多项式系数域
//     poly c = ntt::ntt_inverse(c_ntt);
    
//     return c;
// }

// === 向量/矩阵运算 (mod Q) ===
// (无变化)
poly_vec poly_vec_add(const poly_vec& a, const poly_vec& b) {
    if (a.size() != b.size()) throw std::runtime_error("poly_vec_add size mismatch");
    poly_vec res(a.size());
    for (size_t i = 0; i < a.size(); ++i) {
        res[i] = poly_add(a[i], b[i]);
    }
    return res;
}
poly_vec poly_vec_sub(const poly_vec& a, const poly_vec& b) {
    if (a.size() != b.size()) throw std::runtime_error("poly_vec_sub size mismatch");
    poly_vec res(a.size());
    for (size_t i = 0; i < a.size(); ++i) {
        res[i] = poly_sub(a[i], b[i]);
    }
    return res;
}
poly_vec poly_matrix_vec_mul(const poly_matrix& A, const poly_vec& s) {
    if (A.size() != params::K || s.size() != params::K) throw std::runtime_error("Matrix/vec size mismatch");
    poly_vec res(params::K);
    for(int i = 0; i < params::K; ++i) {
        if (A[i].size() != params::K) throw std::runtime_error("Matrix row size mismatch");
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
    if (a_t.size() != params::K || b.size() != params::K) throw std::runtime_error("poly_vec_transpose_mul size mismatch");
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
        for (int j = 0; j < params::K; ++j) {
            A_t[j][i] = A[i][j];
        }
    }
    return A_t;
}


// === D8 晶格量化辅助函数 ===

/**
 * @brief 对 8 个系数的块执行 D8 量化 (CVP)
 * D8 晶格 = { v in Z^8 | sum(v_i) 是偶数 }
 */
void quantize_d8_block(std::array<int32_t, 8>& b_block,
                       const std::array<int32_t, 8>& val_block,
                       const std::array<int32_t, 8>& d_block,
                       int32_t P_param) 
{
    const double scale_pq = static_cast<double>(P_param) / params::Q;

    std::array<double, 8> x_scaled;
    std::array<int32_t, 8> z_rounded;
    int32_t sum = 0;

    for (int i = 0; i < 8; ++i) {
        // 1. (val + d) mod Q
        int32_t added_val = positive_mod(static_cast<int64_t>(val_block[i]) + d_block[i], params::Q);
        // 2. 缩放到 R_p 空间
        x_scaled[i] = static_cast<double>(added_val) * scale_pq;
        // 3. 找到最近的整数点 z (Z^8)
        z_rounded[i] = static_cast<int32_t>(std::round(x_scaled[i]));
        sum += z_rounded[i];
    }

    // 4. 检查 z 是否在 D8 中 (sum 是偶数)
    if (sum % 2 == 0) {
        // 已经在 D8 中，z 就是 CVP
        for (int i = 0; i < 8; ++i) {
            b_block[i] = positive_mod(z_rounded[i], P_param);
        }
        return;
    }

    // 5. 如果 sum 是奇数，z 不在 D8 中。
    //    CVP 点 z' 是 z 中与 x_scaled[j] 距离最大的分量
    //    (即 |x_scaled[j] - z_rounded[j]| 最大) 移动 1 个单位。
    
    int j_max_dist = 0;
    double max_dist = -1.0;

    for (int i = 0; i < 8; ++i) {
        double dist = std::fabs(x_scaled[i] - z_rounded[i]);
        if (dist > max_dist) {
            max_dist = dist;
            j_max_dist = i;
        }
    }

    // 6. 修正 z[j_max_dist] 使 sum 变为偶数
    if (x_scaled[j_max_dist] > z_rounded[j_max_dist]) {
        z_rounded[j_max_dist] += 1;
    } else {
        z_rounded[j_max_dist] -= 1;
    }

    // 7. z' (现在是 z_rounded) 在 D8 中
    for (int i = 0; i < 8; ++i) {
        b_block[i] = positive_mod(z_rounded[i], P_param);
    }
}


// === 量化 & 反量化 (已修改) ===

poly poly_quantize(const poly& val, const poly& d, int32_t P_param) {
    poly b(params::N);

    // 根据 params.hpp 中的 Q_MODE 选择实现
    if constexpr (params::Q_MODE == params::QUANT_SCALAR) {
        // --- 标量 (Z 晶格) 量化 ---
        const double scale_pq = static_cast<double>(P_param) / params::Q; // p/q
        for(int i = 0; i < params::N; ++i) {
            int32_t added_val = positive_mod(static_cast<int64_t>(val[i]) + d[i], params::Q);
            double scaled_val = scale_pq * added_val;
            int32_t floored_val = static_cast<int32_t>(std::floor(scaled_val));
            b[i] = positive_mod(floored_val, P_param);
        }
    } 
    else if constexpr (params::Q_MODE == params::QUANT_D8) {
        // --- D8 晶格量化 (8 个一组) ---
        // (N 必须是 8 的倍数, 已在 params.hpp 中检查)
        for (int i = 0; i < params::N; i += 8) {
            // 准备 8 个系数的块
            std::array<int32_t, 8> val_block;
            std::array<int32_t, 8> d_block;
            std::array<int32_t, 8> b_block;
            
            for (int j = 0; j < 8; ++j) {
                val_block[j] = val[i + j];
                d_block[j] = d[i + j];
            }
            
            // 调用 D8 CVP 算法
            quantize_d8_block(b_block, val_block, d_block, P_param);

            // 将结果写回多项式
            for (int j = 0; j < 8; ++j) {
                b[i + j] = b_block[j];
            }
        }
    }
    return b;
}

poly_vec poly_vec_quantize(const poly_vec& val, const poly_vec& d, int32_t P_param) {
    if (val.size() != d.size()) throw std::runtime_error("poly_vec_quantize size mismatch");
    poly_vec res(val.size());
    for(size_t i = 0; i < val.size(); ++i) {
        res[i] = poly_quantize(val[i], d[i], P_param);
    }
    return res;
}

// 逻辑: round(q/p * b) mod q
// (此函数无变化。D8 的反量化与 Z 的反量化相同。)
poly poly_dequantize(const poly& b, int32_t P_param) {
    poly res(params::N);
    const double scale_qp = static_cast<double>(params::Q) / P_param; // q/p

    for(int i = 0; i < params::N; ++i) {
        double scaled_val = scale_qp * b[i];
        int64_t rounded_val = static_cast<int64_t>(std::round(scaled_val));
        res[i] = positive_mod(rounded_val, params::Q);
    }
    return res;
}

poly_vec poly_vec_dequantize(const poly_vec& b, int32_t P_param) {
    poly_vec res(b.size());
    for(size_t i = 0; i < b.size(); ++i) {
        res[i] = poly_dequantize(b[i], P_param);
    }
    return res;
}

// === 消息编码/解码 ===
// (无变化)
poly poly_message_encode(const poly& m) {
    poly res(params::N);
    const int32_t scale = params::Q / params::MSG_MODULUS; // q/2
    for (int i = 0; i < params::N; ++i) {
        res[i] = positive_mod(static_cast<int64_t>(m[i]) * scale, params::Q);
    }
    return res;
}
poly poly_message_decode(const poly& val) {
    poly res(params::N);
    const double scale = static_cast<double>(params::MSG_MODULUS) / params::Q; // 2/q
    const int32_t q_half = params::Q / 2;
    for (int i = 0; i < params::N; ++i) {
        int32_t centered_val = positive_mod(val[i], params::Q);
        if (centered_val > q_half) {
            centered_val -= params::Q;
        }
        double scaled_val = static_cast<double>(centered_val) * scale;
        int64_t rounded_val = static_cast<int64_t>(std::round(scaled_val));
        res[i] = positive_mod(rounded_val, params::MSG_MODULUS);
    }
    return res;
}


// === 辅助函数 ===
// (无变化)
void print_poly(const std::string& name, const poly& p, size_t count) {
    std::cout << "  " << name << " (first " << count << " coeffs) = [";
    size_t n = std::min(count, p.size());
    for(size_t i = 0; i < n; ++i) {
        std::cout << p[i] << (i == n - 1 ? "" : ", ");
    }
    if (p.size() > n) {
        std::cout << ", ...";
    }
    std::cout << "]" << std::endl;
}
void print_poly_vec(const std::string& name, const poly_vec& pv, size_t count) {
    std::cout << "  " << name << " (poly_vec[" << pv.size() << "]):" << std::endl;
    for (size_t i = 0; i < pv.size(); ++i) {
        print_poly("    [" + std::to_string(i) + "]", pv[i], count);
    }
}
bool check_poly_eq(const poly& a, const poly& b) {
    if (a.size() != b.size()) return false;
    for (size_t i = 0; i < a.size(); ++i) {
        if (a[i] != b[i]) return false;
    }
    return true;
}