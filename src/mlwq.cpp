#include "mlwq.hpp"
#include "xof.hpp"
#include "random.hpp"
#include "params.hpp"
#include <iostream> // for debug prints

/**
 * @brief M-LWQ 密钥生成 (Algorithm 1, samplepaper.pdf)
 * b_q = Quantize(As, d_pk)
 */
std::pair<mlwq_pk, mlwq_sk> mlwq_keygen(const std::vector<uint8_t>& seed_A,
                                      const std::vector<uint8_t>& seed_d_pk) {
    
    // std::cout << "  [KeyGen] 正在从 seed_A 生成矩阵 A...\n";
    poly_matrix A;
    xof_expand_matrix(A, seed_A, params::Q);

    // std::cout << "  [KeyGen] 正在生成秘密 s (B_eta)...\n";
    mlwq_sk sk;
    sk.s = random_poly_vec_eta(params::K, params::ETA);

    // std::cout << "  [KeyGen] 正在从 seed_d_pk 生成 d_pk (mod " << params::Q_OVER_P_PK_FLOOR << ")...\n";
    poly_vec d_pk;
    xof_expand_poly_vec(d_pk, seed_d_pk, params::K, params::Q_OVER_P_PK_FLOOR);
    
    // std::cout << "  [KeyGen] 正在计算 As...\n";
    poly_vec As = poly_matrix_vec_mul(A, sk.s);

    // std::cout << "  [KeyGen] 正在计算 b_q = Q(As, d_pk) (P_PK=" << params::P_PK << ")...\n";
    mlwq_pk pk;
    pk.seed_A = seed_A;
    pk.seed_d_pk = seed_d_pk;
    pk.b_q = poly_vec_quantize(As, d_pk, params::P_PK); // b_q 在 R_p_pk^k

    return {pk, sk};
}

/**
 * @brief M-LWQ 加密 (Algorithm 2, samplepaper.pdf)
 * c1 = Q(A^T r, d_u)
 * c2 = Q(Decompress(b_q)^T r + m_enc, d_v)
 */
mlwq_ciphertext mlwq_encrypt(const mlwq_pk& pk, 
                             const poly& m_poly, 
                             const std::vector<uint8_t>& seed_ct) {
    
    // std::cout << "  [Encrypt] 正在从 pk.seed_A 重新生成 A...\n";
    poly_matrix A;
    xof_expand_matrix(A, pk.seed_A, params::Q);

    // std::cout << "  [Encrypt] 正在生成秘密 r (B_eta)...\n";
    poly_vec r = random_poly_vec_eta(params::K, params::ETA);

    // std::cout << "  [Encrypt] 正在从 seed_ct 生成 d_u 和 d_v...\n";
    std::vector<uint8_t> seed_d_u = seed_ct;
    seed_d_u.push_back(0x00);
    std::vector<uint8_t> seed_d_v = seed_ct;
    seed_d_v.push_back(0x01);

    poly_vec d_u;
    xof_expand_poly_vec(d_u, seed_d_u, params::K, params::Q_OVER_P_U_FLOOR);
    
    poly_vec d_v_vec;
    xof_expand_poly_vec(d_v_vec, seed_d_v, 1, params::Q_OVER_P_V_FLOOR);
    poly d_v = d_v_vec[0];

    mlwq_ciphertext ct;
    
    // --- 计算 u = c1 = Q(A^T r, d_u) ---
    // std::cout << "  [Encrypt] 正在计算 A_t...\n";
    poly_matrix A_t = poly_matrix_transpose(A);
    // std::cout << "  [Encrypt] 正在计算 A^T r...\n";
    poly_vec A_t_r = poly_matrix_vec_mul(A_t, r);
    // std::cout << "  [Encrypt] 正在计算 c1 = Q(A^T r, d_u) (P_U=" << params::P_U << ")...\n";
    ct.u = poly_vec_quantize(A_t_r, d_u, params::P_U);

    // --- 计算 v = c2 = Q(b_q_dequant^T r + m_enc, d_v) ---
    
    // std::cout << "  [Encrypt] B 正在反量化 pk.b_q (Decompress, P_PK=" << params::P_PK << ")...\n";
    poly_vec b_q_dequant = poly_vec_dequantize(pk.b_q, params::P_PK);
    
    // std::cout << "  [Encrypt] 正在计算 b_q_dequant^T r...\n";
    poly b_t_r = poly_vec_transpose_mul(b_q_dequant, r);

    // std::cout << "  [Encrypt] 正在编码消息 m...\n";
    poly m_encoded = poly_message_encode(m_poly);

    // std::cout << "  [Encrypt] 正在计算 v_val = b_q_dequant^T r + m_enc...\n";
    poly v_val = poly_add(b_t_r, m_encoded);

    // std::cout << "  [Encrypt] 正在计算 c2 = Q(v_val, d_v) (P_V=" << params::P_V << ")...\n";
    ct.v = poly_quantize(v_val, d_v, params::P_V);

    return ct;
}

/**
 * @brief M-LWQ 解密 (Algorithm 3, samplepaper.pdf, 修正版)
 * m' = Decompress(c2) - s^T * Decompress(c1)
 */
poly mlwq_decrypt(const mlwq_sk& sk, const mlwq_ciphertext& ct) {
    
    // std::cout << "  [Decrypt] 正在反量化 c1 (Decompress, P_U=" << params::P_U << ")...\n";
    poly_vec c1_dequant = poly_vec_dequantize(ct.u, params::P_U);

    // std::cout << "  [Decrypt] 正在计算 s^T * c1_dequant...\n";
    poly s_t_u = poly_vec_transpose_mul(sk.s, c1_dequant);

    // std::cout << "  [Decrypt] 正在反量化 c2 (Decompress, P_V=" << params::P_V << ")...\n";
    poly c2_dequant = poly_dequantize(ct.v, params::P_V);

    // std::cout << "  [Decrypt] 正在计算 m' = c2_dequant - s^T * c1_dequant...\n";
    poly m_prime = poly_sub(c2_dequant, s_t_u);

    // std::cout << "  [Decrypt] 正在解码 m'...\n";
    poly m_decrypted = poly_message_decode(m_prime);

    return m_decrypted;
}