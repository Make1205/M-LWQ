#include <iostream>
#include <vector>
#include <string>
#include <cmath>     // for ceil
// #include <chrono>    // 已移除 chrono
#include <numeric>   // for std::accumulate
#include <iomanip>   // for std::setw, std::fixed

#include "cycles.hpp" // <-- 新增: 包含 CPU 周期计数器
#include "params.hpp"
#include "poly.hpp"
#include "mlwq.hpp"
#include "random.hpp" // for random_poly_uniform

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
double calculate_average(const std::vector<unsigned long long>& timings) { // <-- 修改: 接受 unsigned long long
    if (timings.empty()) return 0.0;
    unsigned long long sum = std::accumulate(timings.begin(), timings.end(), 0ULL); // <-- 修改: 接受 unsigned long long
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


    // --- 性能基准测试 ---
    std::cout << "\n[MAIN] 正在运行性能基准测试...\n";
    const int N_BENCH_RUNS = 100; // 运行 100 轮
    
    std::vector<unsigned long long> keygen_timings;  // <-- 修改: 存储周期
    std::vector<unsigned long long> encrypt_timings; // <-- 修改: 存储周期
    std::vector<unsigned long long> decrypt_timings; // <-- 修改: 存储周期
    bool all_success = true;

    // 固定的种子
    std::vector<uint8_t> seed_A = string_to_seed("seed_for_A_matrix");
    std::vector<uint8_t> seed_d_pk = string_to_seed("seed_for_d_pk_dither");
    std::vector<uint8_t> seed_ct = string_to_seed("seed_for_ciphertext_dithers");
    poly m_original = create_test_message();

    for (int i = 0; i < N_BENCH_RUNS; ++i) {
        // --- 1. 计时 KeyGen ---
        unsigned long long t_start_kg = start_cycles(); // <-- 修改
        auto [pk, sk] = mlwq_keygen(seed_A, seed_d_pk);
        unsigned long long t_end_kg = stop_cycles();    // <-- 修改
        keygen_timings.push_back(t_end_kg - t_start_kg);   // <-- 修改

        // --- 2. 计时 Encrypt ---
        unsigned long long t_start_enc = start_cycles(); // <-- 修改
        mlwq_ciphertext ct = mlwq_encrypt(pk, m_original, seed_ct);
        unsigned long long t_end_enc = stop_cycles();    // <-- 修改
        encrypt_timings.push_back(t_end_enc - t_start_enc);   // <-- 修改

        // --- 3. 计时 Decrypt ---
        unsigned long long t_start_dec = start_cycles(); // <-- 修改
        poly m_decrypted = mlwq_decrypt(sk, ct);
        unsigned long long t_end_dec = stop_cycles();    // <-- 修改
        decrypt_timings.push_back(t_end_dec - t_start_dec);   // <-- 修改

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
        std::cout << "\n--- 性能基准 (平均 " << N_BENCH_RUNS << " 轮) ---\n";
        std::cout << std::fixed << std::setprecision(0); // <-- 修改: 周期不需要小数
        std::cout << "  KeyGen:  " << std::setw(12) << calculate_average(keygen_timings) << " cycles\n"; // <-- 修改
        std::cout << "  Encrypt: " << std::setw(12) << calculate_average(encrypt_timings) << " cycles\n"; // <-- 修改
        std::cout << "  Decrypt: " << std::setw(12) << calculate_average(decrypt_timings) << " cycles\n"; // <-- 修改
        std::cout << "=========================\n";

    } else {
        std::cout << "\n=========================\n";
        std::cout << "❌ 验证失败! 存在解密错误\n";
        std::cout << "=========================\n";
    }
    
    return 0;
}