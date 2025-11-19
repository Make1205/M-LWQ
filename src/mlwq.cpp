#include "mlwq.hpp"
#include "xof.hpp"
#include "random.hpp"
#include "params.hpp"
#include "cycles.hpp"

// 宏: 简化计时代码
#define PROFILE_START(t) uint64_t t = start_cycles();
#define PROFILE_END(stats, field, t) if (stats) { stats->field += stop_cycles() - t; }

// ==========================================
// 密钥生成 (KeyGen)
// ==========================================
std::pair<mlwq_pk, mlwq_sk> mlwq_keygen(const std::vector<uint8_t>& seed_A,
                                      const std::vector<uint8_t>& seed_d_pk,
                                      MlwqProfiling* stats) {
    // 1. Gen Matrix A
    PROFILE_START(t1);
    poly_matrix A;
    if (params::USE_AVX2 && params::K == 2) {
        xof_expand_matrix_x4(A, seed_A, params::Q);
    } else {
        xof_expand_matrix(A, seed_A, params::Q);
    }
    PROFILE_END(stats, kg_gen_A, t1);

    // 2. Sample s
    PROFILE_START(t2);
    mlwq_sk sk;
    sk.s = random_poly_vec_eta(params::K, params::ETA);
    PROFILE_END(stats, kg_sample_s, t2);

    // 3. Gen Dither
    PROFILE_START(t3);
    poly_vec d_pk;
    xof_expand_poly_vec(d_pk, seed_d_pk, params::K, params::Q_OVER_P_PK_FLOOR);
    PROFILE_END(stats, kg_gen_d, t3);
    
    // 4. Arithmetic: A * s
    PROFILE_START(t4);
    poly_vec As = poly_matrix_vec_mul(A, sk.s);
    PROFILE_END(stats, kg_arith_as, t4);

    // 5. Quantize
    PROFILE_START(t5);
    mlwq_pk pk;
    pk.seed_A = seed_A;
    pk.seed_d_pk = seed_d_pk;
    pk.b_q = poly_vec_quantize(As, d_pk, params::P_PK);
    PROFILE_END(stats, kg_quant, t5);

    return {pk, sk};
}

// ==========================================
// 加密 (Encrypt)
// ==========================================
mlwq_ciphertext mlwq_encrypt(const mlwq_pk& pk, 
                             const poly& m_poly, 
                             const std::vector<uint8_t>& seed_ct,
                             MlwqProfiling* stats) {
    
    // 1. Gen Matrix A
    PROFILE_START(t1);
    poly_matrix A;
    if (params::USE_AVX2 && params::K == 2) {
        xof_expand_matrix_x4(A, pk.seed_A, params::Q);
    } else {
        xof_expand_matrix(A, pk.seed_A, params::Q);
    }
    PROFILE_END(stats, enc_gen_A, t1);

    // 2. Sample r
    PROFILE_START(t2);
    poly_vec r = random_poly_vec_eta(params::K, params::ETA);
    PROFILE_END(stats, enc_sample_r, t2);

    // 3. Gen Dithers
    PROFILE_START(t3);
    std::vector<uint8_t> seed_d_u = seed_ct; seed_d_u.push_back(0x00);
    std::vector<uint8_t> seed_d_v = seed_ct; seed_d_v.push_back(0x01);
    poly_vec d_u;
    xof_expand_poly_vec(d_u, seed_d_u, params::K, params::Q_OVER_P_U_FLOOR);
    poly_vec d_v_vec;
    xof_expand_poly_vec(d_v_vec, seed_d_v, 1, params::Q_OVER_P_V_FLOOR);
    poly d_v = d_v_vec[0];
    PROFILE_END(stats, enc_gen_d, t3);

    mlwq_ciphertext ct;
    
    // 4. Arithmetic: u = A^T * r
    PROFILE_START(t4);
    poly_matrix A_t = poly_matrix_transpose(A);
    poly_vec A_t_r = poly_matrix_vec_mul(A_t, r);
    PROFILE_END(stats, enc_arith_u, t4);

    // 5. Arithmetic: v = b^T * r + m
    PROFILE_START(t5);
    poly_vec b_q_dequant = poly_vec_dequantize(pk.b_q, params::P_PK);
    poly b_t_r = poly_vec_transpose_mul(b_q_dequant, r);
    poly m_encoded = poly_message_encode(m_poly);
    poly v_val = poly_add(b_t_r, m_encoded);
    PROFILE_END(stats, enc_arith_v, t5);

    // 6. Quantize
    PROFILE_START(t6);
    ct.u = poly_vec_quantize(A_t_r, d_u, params::P_U);
    ct.v = poly_quantize(v_val, d_v, params::P_V);
    PROFILE_END(stats, enc_quant, t6);

    return ct;
}

// ==========================================
// 解密 (Decrypt)
// ==========================================
poly mlwq_decrypt(const mlwq_sk& sk, const mlwq_ciphertext& ct, MlwqProfiling* stats) {
    
    // 1. Dequantize
    PROFILE_START(t1);
    poly_vec c1_dequant = poly_vec_dequantize(ct.u, params::P_U);
    poly c2_dequant = poly_dequantize(ct.v, params::P_V);
    PROFILE_END(stats, dec_dequant, t1);

    // 2. Arithmetic: v - s^T * u
    PROFILE_START(t2);
    poly s_t_u = poly_vec_transpose_mul(sk.s, c1_dequant);
    poly m_prime = poly_sub(c2_dequant, s_t_u);
    PROFILE_END(stats, dec_mul_sub, t2);

    // 3. Decode
    PROFILE_START(t3);
    poly res = poly_message_decode(m_prime);
    PROFILE_END(stats, dec_decode, t3);

    return res;
}