#pragma once
#include "poly.hpp"
#include <vector>
#include <cstdint>
#include <utility>

// --- 性能分析统计结构 ---
struct MlwqProfiling {
    // KeyGen Components
    unsigned long long kg_gen_A = 0;      // 生成矩阵 A
    unsigned long long kg_sample_s = 0;   // 采样私钥 s
    unsigned long long kg_gen_d = 0;      // 生成抖动 d
    // [修正] 统一名称为 kg_arith_as
    unsigned long long kg_arith_as = 0;   // 算术: A * s 
    unsigned long long kg_quant = 0;      // 量化

    // Encrypt Components
    unsigned long long enc_gen_A = 0;     // 生成矩阵 A
    unsigned long long enc_sample_r = 0;  // 采样随机数 r
    unsigned long long enc_gen_d = 0;     // 生成抖动 d
    // [修正] 统一名称为 enc_arith_u, enc_arith_v
    unsigned long long enc_arith_u = 0;   // 算术: u = A^T * r
    unsigned long long enc_arith_v = 0;   // 算术: v = b^T * r + m
    unsigned long long enc_quant = 0;     // 量化 u, v

    // Decrypt Components
    unsigned long long dec_dequant = 0;   // 反量化
    unsigned long long dec_mul_sub = 0;   // 算术: v - s^T * u
    unsigned long long dec_decode = 0;    // 解码

    void reset() { *this = MlwqProfiling(); }
};

// 公钥
struct mlwq_pk {
    std::vector<uint8_t> seed_A;
    std::vector<uint8_t> seed_d_pk;
    poly_vec b_q;
};

// 私钥
struct mlwq_sk {
    poly_vec s;
};

// 密文
struct mlwq_ciphertext {
    poly_vec u;
    poly v;
};

// --- 函数接口 (增加 stats 参数) ---

std::pair<mlwq_pk, mlwq_sk> mlwq_keygen(const std::vector<uint8_t>& seed_A,
                                      const std::vector<uint8_t>& seed_d_pk,
                                      MlwqProfiling* stats = nullptr);

mlwq_ciphertext mlwq_encrypt(const mlwq_pk& pk, 
                             const poly& m_poly, 
                             const std::vector<uint8_t>& seed_ct,
                             MlwqProfiling* stats = nullptr);

poly mlwq_decrypt(const mlwq_sk& sk, const mlwq_ciphertext& ct,
                  MlwqProfiling* stats = nullptr);