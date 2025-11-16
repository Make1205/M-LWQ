#include <iostream>
#include <vector>
#include <string>
#include <cmath>     // for ceil
#include <numeric>   // for std::accumulate
#include <iomanip>   // for std::setw, std::fixed

#include "cycles.hpp" // <-- 包含 CPU 周期计数器
#include "params.hpp"
#include "poly.hpp"
#include "mlwq.hpp"
#include "random.hpp" // for random_poly_uniform, random_poly_vec_eta
#include "xof.hpp"    // <-- 新增: 需要 XOF 来生成抖动

// 辅助函数，用于创建随机种子
std::vector<uint8_t> string_to_seed(const std::string& s) {
    return std::vector<uint8_t>(s.begin(), s.end());
}

// 辅助函数，用于创建随机消息
poly create_test_message() {
    // 创建一个系数为 0 或 1 的随机多项式
    return random_poly_uniform(params::MSG_MODULUS);
}

// 辅助函数，用于计算平均值
double calculate_average(const std::vector<unsigned long long>& timings) {
    if (timings.empty()) return 0.0;
    unsigned long long sum = std::accumulate(timings.begin(), timings.end(), 0ULL);
    return static_cast<double>(sum) / timings.size();
}

int main() {
    std::cout << "=== C++ M-LWQ PKE 完整验证与性能基准 ===\n";
    std::cout << "参数集: " << params::PARAM_SET_NAME << "\n";
    std::cout << "参数: K=" << params::K
              << ", N=" << params::N 
              << ", Q=" << params::Q 
              << ", ETA=" << params::ETA << std::endl;
    std::cout << "消息模数: " << params::MSG_MODULUS << std::endl;
    
    // 打印量化模式
    if constexpr (params::Q_MODE == params::QUANT_SCALAR) {
        std::cout << "量化模式: SCALAR (Z 晶格)\n";
    } else if constexpr (params::Q_MODE == params::QUANT_D8) {
        std::cout << "量化模式: D8 (类 E8 晶格)\n";
    }

    // 打印独立的 P 参数 (缩放系数)
    std::cout << "\n--- 缩放系数 (M-LWQ-" << params::K * params::N << ") ---\n";
    std::cout << "  d_pk=" << params::D_PK_BITS << " (P_PK=" << params::P_PK 
              << ", dither mod " << params::Q_OVER_P_PK_FLOOR << ")\n";
    std::cout << "  d_u=" << params::D_U_BITS << " (P_U=" << params::P_U 
              << ", dither mod " << params::Q_OVER_P_U_FLOOR << ")\n";
    std::cout << "  d_v=" << params::D_V_BITS << " (P_V=" << params::P_V 
              << ", dither mod " << params::Q_OVER_P_V_FLOOR << ")\n";


    // --- 理论尺寸计算 ---
    std::cout << "\n--- 理论尺寸 (M-LWQ-" << params::K * params::N << ") ---\n";
    
    const size_t s_bits_per_coeff = 3; 
    const size_t seed_bytes = 32; 

    size_t b_q_bits = params::K * params::N * params::D_PK_BITS;
    size_t pk_size = seed_bytes + seed_bytes + (size_t)std::ceil(b_q_bits / 8.0);
    std::cout << "  公钥 (PK) 尺寸: " << pk_size << " 字节"
              << " (seedA: " << seed_bytes << ", seed_d_pk: " << seed_bytes 
              << ", b_q: " << (size_t)std::ceil(b_q_bits / 8.0) << ")\n";
    
    size_t sk_bits = params::K * params::N * s_bits_per_coeff;
    size_t sk_size = (size_t)std::ceil(sk_bits / 8.0);
    std::cout << "  私钥 (SK) 尺寸: " << sk_size << " 字节\n";
    
    size_t u_bits = params::K * params::N * params::D_U_BITS;
    size_t v_bits = params::N * params::D_V_BITS;
    size_t ct_size = (size_t)std::ceil((u_bits + v_bits) / 8.0);
    std::cout << "  密文 (CT) 尺寸: " << ct_size << " 字节"
              << " (u: " << (size_t)std::ceil(u_bits / 8.0) 
              << ", v: " << (size_t)std::ceil(v_bits / 8.0) << ")\n";
    
    size_t pt_bits = params::N * 1;
    size_t pt_size = (size_t)std::ceil(pt_bits / 8.0);
    std::cout << "  明文 (PT) 尺寸: " << pt_size << " 字节\n";


    // ===================================================================
    // ===           新增: (量化 vs 采样) 独立基准测试           ===
    // ===================================================================
    std::cout << "\n[MAIN] 正在运行 (量化 vs 采样) 基准测试...\n";
    
    const int N_BENCH_RUNS = 100; // 运行 100 轮

    std::vector<unsigned long long> quant_pk_timings, quant_u_timings, quant_v_timings;
    std::vector<unsigned long long> sample_e_pk_timings, sample_e_u_timings, sample_e_v_timings;

    // --- 1. 设置测试所需的通用输入 ---
    // (我们只需要一次 KeyGen 来获取有效的 pk.b_q 以计算 v_val)
    std::vector<uint8_t> seed_A = string_to_seed("seed_for_A_matrix");
    std::vector<uint8_t> seed_d_pk = string_to_seed("seed_for_d_pk_dither");
    std::vector<uint8_t> seed_ct = string_to_seed("seed_for_ciphertext_dithers");
    poly m_original = create_test_message();

    auto [pk, sk] = mlwq_keygen(seed_A, seed_d_pk); // sk.s

    // --- 2. 准备量化的 *输入值* (val) ---
    // (这些是 Q(val, d) 中的 'val' )
    poly_matrix A; 
    xof_expand_matrix(A, seed_A, params::Q);
    poly_vec r = random_poly_vec_eta(params::K, params::ETA);

    // PK: val = As
    poly_vec As = poly_matrix_vec_mul(A, sk.s);
    // CT (u): val = A^T r
    poly_matrix A_t = poly_matrix_transpose(A);
    poly_vec A_t_r = poly_matrix_vec_mul(A_t, r);
    // CT (v): val = b_q^T r + m_enc
    poly_vec b_q_dequant = poly_vec_dequantize(pk.b_q, params::P_PK);
    poly b_t_r = poly_vec_transpose_mul(b_q_dequant, r);
    poly m_encoded = poly_message_encode(m_original);
    poly v_val = poly_add(b_t_r, m_encoded);
    
    // --- 3. 准备量化的 *抖动值* (d) ---
    poly_vec d_pk; 
    xof_expand_poly_vec(d_pk, seed_d_pk, params::K, params::Q_OVER_P_PK_FLOOR);
    
    std::vector<uint8_t> seed_d_u = seed_ct; seed_d_u.push_back(0x00);
    std::vector<uint8_t> seed_d_v = seed_ct; seed_d_v.push_back(0x01);
    poly_vec d_u; 
    xof_expand_poly_vec(d_u, seed_d_u, params::K, params::Q_OVER_P_U_FLOOR);
    poly_vec d_v_vec; 
    xof_expand_poly_vec(d_v_vec, seed_d_v, 1, params::Q_OVER_P_V_FLOOR);
    poly d_v = d_v_vec[0];
    
    // --- 4. 运行对比基准测试循环 ---
    for (int i = 0; i < N_BENCH_RUNS; ++i) {
        // --- A. 测试 M-LWQ 量化 (Quantize) ---
        
        // A1. 公钥量化 (k x k)
        unsigned long long t_s_q_pk = start_cycles();
        volatile poly_vec b_q_bench = poly_vec_quantize(As, d_pk, params::P_PK);
        unsigned long long t_e_q_pk = stop_cycles();
        quant_pk_timings.push_back(t_e_q_pk - t_s_q_pk);
        
        // A2. 密文 u 量化 (k x 1)
        unsigned long long t_s_q_u = start_cycles();
        volatile poly_vec u_bench = poly_vec_quantize(A_t_r, d_u, params::P_U);
        unsigned long long t_e_q_u = stop_cycles();
        quant_u_timings.push_back(t_e_q_u - t_s_q_u);

        // A3. 密文 v 量化 (1 x 1)
        unsigned long long t_s_q_v = start_cycles();
        volatile poly v_bench = poly_quantize(v_val, d_v, params::P_V);
        unsigned long long t_e_q_v = stop_cycles();
        quant_v_timings.push_back(t_e_q_v - t_s_q_v);

        // --- B. 测试 LWE 错误采样 (Sample e) ---
        // (这是传统 LWE/Kyber 中等效的操作)
        
        // B1. 公钥错误 e_pk (k x 1)
        unsigned long long t_s_e_pk = start_cycles();
        volatile poly_vec e_pk = random_poly_vec_eta(params::K, params::ETA);
        unsigned long long t_e_e_pk = stop_cycles();
        sample_e_pk_timings.push_back(t_e_e_pk - t_s_e_pk);

        // B2. 密文错误 e_u (k x 1)
        unsigned long long t_s_e_u = start_cycles();
        volatile poly_vec e_u = random_poly_vec_eta(params::K, params::ETA);
        unsigned long long t_e_e_u = stop_cycles();
        sample_e_u_timings.push_back(t_e_e_u - t_s_e_u);

        // B3. 密文错误 e_v (1 x 1)
        unsigned long long t_s_e_v = start_cycles();
        volatile poly e_v = random_poly_eta(params::ETA);
        unsigned long long t_e_e_v = stop_cycles();
        sample_e_v_timings.push_back(t_e_e_v - t_s_e_v);
    }
    
    // --- 5. 打印对比结果 ---
    std::cout << "\n--- (量化 vs 采样) 基准 (平均 " << N_BENCH_RUNS << " 轮) ---\n";
    std::cout << std::fixed << std::setprecision(0);
    std::cout << "  组件       | M-LWQ 量化 (Quantize) | LWE 采样 (Sample e) \n";
    std::cout << "  ----------------|-----------------------|----------------------\n";
    std::cout << "  PK (b_q)      | " << std::setw(12) << calculate_average(quant_pk_timings) << " cycles | " 
                                   << std::setw(12) << calculate_average(sample_e_pk_timings) << " cycles\n";
    std::cout << "  CT (u)        | " << std::setw(12) << calculate_average(quant_u_timings) << " cycles | "
                                   << std::setw(12) << calculate_average(sample_e_u_timings) << " cycles\n";
    std::cout << "  CT (v)        | " << std::setw(12) << calculate_average(quant_v_timings) << " cycles | "
                                   << std::setw(12) << calculate_average(sample_e_v_timings) << " cycles\n";
    std::cout << "====================================================\n";


    // ===================================================================
    // ===           原有: 完整流程基准测试与验证           ===
    // ===================================================================
    std::cout << "\n[MAIN] 正在运行完整流程基准测试...\n";
    
    std::vector<unsigned long long> keygen_timings;
    std::vector<unsigned long long> encrypt_timings;
    std::vector<unsigned long long> decrypt_timings;
    bool all_success = true;

    // (使用上面已经生成的固定种子和消息)

    for (int i = 0; i < N_BENCH_RUNS; ++i) {
        // --- 1. 计时 KeyGen ---
        unsigned long long t_start_kg = start_cycles();
        // (注意: KeyGen 内部会执行 "PK (b_q)" 的量化)
        auto [pk_loop, sk_loop] = mlwq_keygen(seed_A, seed_d_pk);
        unsigned long long t_end_kg = stop_cycles();
        keygen_timings.push_back(t_end_kg - t_start_kg);

        // --- 2. 计时 Encrypt ---
        unsigned long long t_start_enc = start_cycles();
        // (注意: Encrypt 内部会执行 "CT (u)" 和 "CT (v)" 的量化)
        mlwq_ciphertext ct = mlwq_encrypt(pk_loop, m_original, seed_ct);
        unsigned long long t_end_enc = stop_cycles();
        encrypt_timings.push_back(t_end_enc - t_start_enc);

        // --- 3. 计时 Decrypt ---
        unsigned long long t_start_dec = start_cycles();
        poly m_decrypted = mlwq_decrypt(sk_loop, ct);
        unsigned long long t_end_dec = stop_cycles();
        decrypt_timings.push_back(t_end_dec - t_start_dec);

        // --- 4. 验证 ---
        if (!check_poly_eq(m_original, m_decrypted)) {
            all_success = false;
        }
    }

    // --- 打印验证结果 ---
    std::cout << "\n[MAIN] 正在验证 " << N_BENCH_RUNS << " 轮运行的正确性...\n";
    if (all_success) {
        std::cout << "\n=========================\n";
        std::cout << "✅ 验证成功! 所有 " << N_BENCH_RUNS << " 轮解密均正确\n";
        std::cout << "=========================\n";

        // --- 打印性能结果 ---
        std::cout << "\n--- 完整流程性能基准 (平均 " << N_BENCH_RUNS << " 轮) ---\n";
        std::cout << std::fixed << std::setprecision(0);
        std::cout << "  KeyGen:  " << std::setw(12) << calculate_average(keygen_timings) << " cycles\n";
        std::cout << "  Encrypt: " << std::setw(12) << calculate_average(encrypt_timings) << " cycles\n";
        std::cout << "  Decrypt: " << std::setw(12) << calculate_average(decrypt_timings) << " cycles\n";
        std::cout << "=========================\n";

    } else {
        std::cout << "\n=========================\n";
        std::cout << "❌ 验证失败! 存在解密错误\n";
        std::cout << "=========================\n";
    }
    
    return 0;
}