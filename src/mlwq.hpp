#pragma once
#include "poly.hpp"
#include <vector>
#include <cstdint>
#include <utility>

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

// --- 函数接口 ---

std::pair<mlwq_pk, mlwq_sk> mlwq_keygen(const std::vector<uint8_t>& seed_A,
                                      const std::vector<uint8_t>& seed_d_pk);

mlwq_ciphertext mlwq_encrypt(const mlwq_pk& pk, 
                             const poly& m_poly, 
                             const std::vector<uint8_t>& seed_ct);

poly mlwq_decrypt(const mlwq_sk& sk, const mlwq_ciphertext& ct);