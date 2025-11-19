#include <iostream>
#include <vector>
#include <string>
#include <cmath>     // for ceil, round
#include <numeric>   // for std::accumulate
#include <iomanip>   // for std::setw, std::fixed

#include "cycles.hpp" 
#include "params.hpp"
#include "poly.hpp"
#include "mlwq.hpp"
#include "random.hpp" 
#include "xof.hpp"    
#include "ntt.hpp" // 引入 NTT 初始化

// 定义全局 AVX2 开关
namespace params {
    bool USE_AVX2 = false;
}

// --- 辅助函数 ---
std::vector<uint8_t> string_to_seed(const std::string& s) {
    return std::vector<uint8_t>(s.begin(), s.end());
}

poly create_test_message() {
    return random_poly_uniform(params::MSG_MODULUS);
}

double calculate_average(const std::vector<unsigned long long>& timings) {
    if (timings.empty()) return 0.0;
    unsigned long long sum = std::accumulate(timings.begin(), timings.end(), 0ULL);
    return static_cast<double>(sum) / timings.size();
}

// --- 存储单次测试结果的结构体 ---
struct BenchResult {
    std::string mode_name;
    bool all_passed = false;
    double t_keygen = 0;
    double t_encrypt = 0;
    double t_decrypt = 0;
    double t_quant_pk = 0;
    double t_quant_u = 0;
    double t_quant_v = 0;
    double t_sample_pk = 0; // 对比基准
};

/**
 * @brief 运行完整的测试套件（包含正确性验证和两类基准测试）
 */
BenchResult run_full_benchmark(bool use_avx, int rounds) {
    // 1. 设置模式
    params::USE_AVX2 = use_avx;
    BenchResult res;
    res.mode_name = use_avx ? "AVX2 Mode" : "Scalar Mode";

    std::cout << "\n>>> 正在运行: " << res.mode_name << " (" << rounds << " 轮) <<<\n";

    // 2. 准备通用种子和数据
    std::vector<uint8_t> seed_A = string_to_seed("seed_A_BENCH");
    std::vector<uint8_t> seed_d = string_to_seed("seed_d_BENCH");
    std::vector<uint8_t> seed_ct = string_to_seed("seed_ct_BENCH");
    poly m_original = create_test_message();

    // =================================================================
    // Part A: 完整流程 (KeyGen -> Enc -> Dec) 与 正确性验证
    // =================================================================
    std::vector<unsigned long long> timings_kg, timings_enc, timings_dec;
    bool check_success = true;

    // 预热 (Warmup)
    mlwq_keygen(seed_A, seed_d);

    for (int i = 0; i < rounds; ++i) {
        // KeyGen
        unsigned long long t1 = start_cycles();
        auto keys = mlwq_keygen(seed_A, seed_d);
        timings_kg.push_back(stop_cycles() - t1);

        // Encrypt
        unsigned long long t2 = start_cycles();
        auto ct = mlwq_encrypt(keys.first, m_original, seed_ct);
        timings_enc.push_back(stop_cycles() - t2);

        // Decrypt
        unsigned long long t3 = start_cycles();
        poly m_dec = mlwq_decrypt(keys.second, ct);
        timings_dec.push_back(stop_cycles() - t3);

        // Verify Correctness (每轮都检查!)
        if (!check_poly_eq(m_original, m_dec)) {
            check_success = false;
            // 如果出错，打印一次错误信息
            if (i == 0) std::cerr << "   [ERROR] Decryption mismatch in " << res.mode_name << "!\n";
        }
    }

    res.all_passed = check_success;
    res.t_keygen = calculate_average(timings_kg);
    res.t_encrypt = calculate_average(timings_enc);
    res.t_decrypt = calculate_average(timings_dec);

    if (res.all_passed) {
        std::cout << "   [PASS] 正确性验证通过 (所有 " << rounds << " 轮解密均正确)\n";
    } else {
        std::cout << "   [FAIL] 正确性验证失败!\n";
    }

    // =================================================================
    // Part B: 组件基准 (Quantize vs Sample) - 保留原来的详细对比
    // =================================================================
    // 准备输入数据
    auto [pk, sk] = mlwq_keygen(seed_A, seed_d);
    poly_matrix A; xof_expand_matrix(A, seed_A, params::Q);
    poly_vec r = random_poly_vec_eta(params::K, params::ETA);
    
    // 构造量化输入
    poly_vec As = poly_matrix_vec_mul(A, sk.s); // PK input
    poly_matrix A_t = poly_matrix_transpose(A);
    poly_vec A_t_r = poly_matrix_vec_mul(A_t, r); // CT(u) input
    // 构造 CT(v) input (略去部分细节，只测 Quantize 函数本身性能)
    poly v_val = random_poly_uniform(params::Q); 

    // 构造 Dithers
    poly_vec d_pk; xof_expand_poly_vec(d_pk, seed_d, params::K, params::Q_OVER_P_PK_FLOOR);
    poly_vec d_u = d_pk; // 复用维度相同的向量用于测试
    poly d_v = d_pk[0];

    std::vector<unsigned long long> t_q_pk, t_q_u, t_q_v, t_samp;

    for(int i=0; i<rounds; i++) {
        // Quantize PK (b_q)
        auto s1 = start_cycles();
        volatile auto q1 = poly_vec_quantize(As, d_pk, params::P_PK);
        t_q_pk.push_back(stop_cycles() - s1);

        // Quantize CT (u)
        auto s2 = start_cycles();
        volatile auto q2 = poly_vec_quantize(A_t_r, d_u, params::P_U);
        t_q_u.push_back(stop_cycles() - s2);

        // Quantize CT (v)
        auto s3 = start_cycles();
        volatile auto q3 = poly_quantize(v_val, d_v, params::P_V);
        t_q_v.push_back(stop_cycles() - s3);

        // Sample Error (Benchmark Baseline)
        auto s4 = start_cycles();
        volatile auto samp = random_poly_vec_eta(params::K, params::ETA);
        t_samp.push_back(stop_cycles() - s4);
    }

    res.t_quant_pk = calculate_average(t_q_pk);
    res.t_quant_u = calculate_average(t_q_u);
    res.t_quant_v = calculate_average(t_q_v);
    res.t_sample_pk = calculate_average(t_samp);

    return res;
}

int main() {
    ntt::init_tables(); // 必须初始化
    
    std::cout << "==========================================================\n";
    std::cout << "       M-LWQ PKE: 完整验证与 AVX2 性能对比基准\n";
    std::cout << "==========================================================\n";
    
    // --- 1. 打印参数信息 (保留原测试的内容) ---
    std::cout << "参数集: " << params::PARAM_SET_NAME << "\n";
    std::cout << "维度: N=" << params::N << ", K=" << params::K 
              << ", Q=" << params::Q << ", ETA=" << params::ETA << "\n";
    
    if constexpr (params::Q_MODE == params::QUANT_SCALAR) {
        std::cout << "量化模式: SCALAR (Z 晶格)\n";
    }

    // 打印理论尺寸
    size_t pk_size = 32 + 32 + (size_t)std::ceil((params::K * params::N * params::D_PK_BITS) / 8.0);
    size_t ct_size = (size_t)std::ceil((params::K * params::N * params::D_U_BITS + params::N * params::D_V_BITS) / 8.0);
    std::cout << "\n[理论尺寸]\n";
    std::cout << "  Public Key: " << pk_size << " bytes\n";
    std::cout << "  Ciphertext: " << ct_size << " bytes\n";

    // --- 2. 运行测试 ---
    int rounds = 100;
    
    // 运行 Scalar 模式
    BenchResult res_scalar = run_full_benchmark(false, rounds);

    // 运行 AVX2 模式
    BenchResult res_avx = run_full_benchmark(true, rounds);

    // --- 3. 最终对比整合报表 ---
    std::cout << "\n\n";
    std::cout << "==========================================================================\n";
    std::cout << "                 性能对比总结 (CPU Cycles) | 平均 " << rounds << " 轮\n";
    std::cout << "==========================================================================\n";
    
    // 表头
    std::cout << std::left << std::setw(20) << "Operation" 
              << std::setw(18) << "Scalar Mode" 
              << std::setw(18) << "AVX2 Mode" 
              << std::setw(15) << "Speedup" 
              << "Status" << std::endl;
    std::cout << "--------------------------------------------------------------------------\n";

    auto print_row = [&](std::string name, double t_s, double t_a, bool check = true) {
        double speedup = t_s / t_a;
        std::cout << std::left << std::setw(20) << name 
                  << std::setw(18) << (long long)t_s 
                  << std::setw(18) << (long long)t_a 
                  << std::fixed << std::setprecision(2) << speedup << "x" 
                  << std::setw(5) << " " 
                  << (check ? "✅" : "") << std::endl;
    };

    // 完整流程数据
    print_row("KeyGen", res_scalar.t_keygen, res_avx.t_keygen);
    print_row("Encrypt", res_scalar.t_encrypt, res_avx.t_encrypt);
    print_row("Decrypt", res_scalar.t_decrypt, res_avx.t_decrypt);

    std::cout << "--------------------------------------------------------------------------\n";
    std::cout << "Component: Quantization vs Sampling (LWE Baseline)\n";
    std::cout << "--------------------------------------------------------------------------\n";

    print_row("Quantize(PK)", res_scalar.t_quant_pk, res_avx.t_quant_pk, false);
    print_row("Quantize(u)", res_scalar.t_quant_u, res_avx.t_quant_u, false);
    
    std::cout << "--------------------------------------------------------------------------\n";
    // 打印 Sample 作为参考
    std::cout << std::left << std::setw(20) << "Sample(e) Ref" 
              << std::setw(18) << (long long)res_scalar.t_sample_pk 
              << std::setw(18) << "-" 
              << "Baseline" << std::endl;

    std::cout << "==========================================================================\n";
    
    if (res_scalar.all_passed && res_avx.all_passed) {
        std::cout << "\n[FINAL RESULT] 所有模式下的加解密验证均通过！测试完成。\n";
    } else {
        std::cout << "\n[FINAL RESULT] 警告：存在验证失败的模式，请检查实现！\n";
    }

    return 0;
}