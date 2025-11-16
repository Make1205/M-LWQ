#pragma once
#include "poly.hpp"
#include <vector>
#include <cstdint>

// 公钥 (Algorithm 1, samplepaper.pdf)
struct mlwq_pk {
    std::vector<uint8_t> seed_A;    // (seed_A)
    std::vector<uint8_t> seed_d_pk; // (seed_d)
    poly_vec b_q;                   // (b_q)  <- 在 R_p^k 中, 已量化
};

// 私钥 (Algorithm 1, samplepaper.pdf)
struct mlwq_sk {
    poly_vec s; // (s) <- 在 R_eta^k 中, "small"
};

// 密文 (Algorithm 2, samplepaper.pdf)
struct mlwq_ciphertext {
    poly_vec u; // (c1) <- 在 R_p^k 中, 已量化
    poly v;     // (c2) <- 在 R_p 中, 已量化
};

/**
 * @brief M-LWQ 密钥生成 (Algorithm 1, samplepaper.pdf)
 * @param seed_A (输入) 生成 A 的种子
 * @param seed_d_pk (输入) 生成公钥 dither 的种子
 * @return 公私钥对 {pk, sk}
 */
std::pair<mlwq_pk, mlwq_sk> mlwq_keygen(const std::vector<uint8_t>& seed_A,
                                      const std::vector<uint8_t>& seed_d_pk);

/**
 * @brief M-LWQ 加密 (Algorithm 2, samplepaper.pdf)
 * @param pk (输入) 公钥
 * @param m_poly (输入) 消息多项式 (系数为 0 或 1)
 * @param seed_ct (输入) 生成密文 dithers (d_u, d_v) 的种子
 * @return 密文
 */
mlwq_ciphertext mlwq_encrypt(const mlwq_pk& pk, 
                             const poly& m_poly, 
                             const std::vector<uint8_t>& seed_ct);

/**
 * @brief M-LWQ 解密 (Algorithm 3, samplepaper.pdf, 修正版)
 * @param sk (输入) 私钥
 * @param ct (输入) 密文
 * @return 解密后的消息多项式 (系数为 0 或 1)
 */
poly mlwq_decrypt(const mlwq_sk& sk, const mlwq_ciphertext& ct);