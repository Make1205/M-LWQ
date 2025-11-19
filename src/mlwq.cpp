#include "mlwq.hpp"
#include "xof.hpp"
#include "random.hpp"
#include "params.hpp"

// 密钥生成
std::pair<mlwq_pk, mlwq_sk> mlwq_keygen(const std::vector<uint8_t>& seed_A,
                                      const std::vector<uint8_t>& seed_d_pk) {
    poly_matrix A;
    
    // [修改] 根据模式选择生成方式
    if (params::USE_AVX2 && params::K == 2) {
        xof_expand_matrix_x4(A, seed_A, params::Q);
    } else {
        xof_expand_matrix(A, seed_A, params::Q);
    }

    mlwq_sk sk;
    sk.s = random_poly_vec_eta(params::K, params::ETA);

    poly_vec d_pk;
    xof_expand_poly_vec(d_pk, seed_d_pk, params::K, params::Q_OVER_P_PK_FLOOR);
    
    poly_vec As = poly_matrix_vec_mul(A, sk.s);

    mlwq_pk pk;
    pk.seed_A = seed_A;
    pk.seed_d_pk = seed_d_pk;
    pk.b_q = poly_vec_quantize(As, d_pk, params::P_PK);

    return {pk, sk};
}

// 加密
mlwq_ciphertext mlwq_encrypt(const mlwq_pk& pk, 
                             const poly& m_poly, 
                             const std::vector<uint8_t>& seed_ct) {
    poly_matrix A;
    
    // [修改] 必须与 KeyGen 保持一致的生成逻辑
    if (params::USE_AVX2 && params::K == 2) {
        xof_expand_matrix_x4(A, pk.seed_A, params::Q);
    } else {
        xof_expand_matrix(A, pk.seed_A, params::Q);
    }

    poly_vec r = random_poly_vec_eta(params::K, params::ETA);

    std::vector<uint8_t> seed_d_u = seed_ct; seed_d_u.push_back(0x00);
    std::vector<uint8_t> seed_d_v = seed_ct; seed_d_v.push_back(0x01);
    
    poly_vec d_u;
    xof_expand_poly_vec(d_u, seed_d_u, params::K, params::Q_OVER_P_U_FLOOR);
    
    poly_vec d_v_vec;
    xof_expand_poly_vec(d_v_vec, seed_d_v, 1, params::Q_OVER_P_V_FLOOR);
    poly d_v = d_v_vec[0];

    mlwq_ciphertext ct;
    poly_matrix A_t = poly_matrix_transpose(A);
    poly_vec A_t_r = poly_matrix_vec_mul(A_t, r);
    ct.u = poly_vec_quantize(A_t_r, d_u, params::P_U);

    poly_vec b_q_dequant = poly_vec_dequantize(pk.b_q, params::P_PK);
    poly b_t_r = poly_vec_transpose_mul(b_q_dequant, r);
    poly m_encoded = poly_message_encode(m_poly);
    poly v_val = poly_add(b_t_r, m_encoded);
    ct.v = poly_quantize(v_val, d_v, params::P_V);

    return ct;
}

// 解密 (保持不变)
poly mlwq_decrypt(const mlwq_sk& sk, const mlwq_ciphertext& ct) {
    poly_vec c1_dequant = poly_vec_dequantize(ct.u, params::P_U);
    poly s_t_u = poly_vec_transpose_mul(sk.s, c1_dequant);
    poly c2_dequant = poly_dequantize(ct.v, params::P_V);
    poly m_prime = poly_sub(c2_dequant, s_t_u);
    return poly_message_decode(m_prime);
}