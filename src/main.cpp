#include <iostream>
#include <vector>
#include <string>
#include <cmath>     
#include <numeric>   
#include <iomanip>   

#include "cycles.hpp" 
#include "params.hpp"
#include "poly.hpp"
#include "mlwq.hpp"
#include "random.hpp" 
#include "xof.hpp"    
#include "ntt.hpp" 

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

// --- 结果结构体 ---
struct BenchResult {
    std::string mode_name;
    bool all_passed = false;
    
    // 完整流程
    double t_keygen = 0;
    double t_encrypt = 0;
    double t_decrypt = 0;
    
    // 量化组件 (Quantization)
    double t_quant_vec = 0;  // PK / u (K x N)
    double t_quant_poly = 0; // v (1 x N)
    
    // 采样组件 (LWE Sampling)
    double t_sample_vec = 0; // PK / u 对应的错误向量
    double t_sample_poly = 0; // v 对应的错误多项式
};

/**
 * @brief 运行完整的测试套件
 */
BenchResult run_full_benchmark(bool use_avx, int rounds) {
    params::USE_AVX2 = use_avx;
    BenchResult res;
    res.mode_name = use_avx ? "AVX2 Mode" : "Scalar Mode";

    std::cout << "\n>>> 正在运行: " << res.mode_name << " (" << rounds << " 轮) <<<\n";

    std::vector<uint8_t> seed_A = string_to_seed("seed_A_BENCH");
    std::vector<uint8_t> seed_d = string_to_seed("seed_d_BENCH");
    std::vector<uint8_t> seed_ct = string_to_seed("seed_ct_BENCH");
    poly m_original = create_test_message();

    // 1. 完整流程 (KeyGen -> Enc -> Dec)
    std::vector<unsigned long long> t_kg, t_enc, t_dec;
    bool check_success = true;

    // Warmup
    mlwq_keygen(seed_A, seed_d);

    for (int i = 0; i < rounds; ++i) {
        auto s1 = start_cycles();
        auto keys = mlwq_keygen(seed_A, seed_d);
        t_kg.push_back(stop_cycles() - s1);

        auto s2 = start_cycles();
        auto ct = mlwq_encrypt(keys.first, m_original, seed_ct);
        t_enc.push_back(stop_cycles() - s2);

        auto s3 = start_cycles();
        poly m_dec = mlwq_decrypt(keys.second, ct);
        t_dec.push_back(stop_cycles() - s3);

        if (!check_poly_eq(m_original, m_dec)) check_success = false;
    }

    res.all_passed = check_success;
    res.t_keygen = calculate_average(t_kg);
    res.t_encrypt = calculate_average(t_enc);
    res.t_decrypt = calculate_average(t_dec);

    if (res.all_passed) std::cout << "   [PASS] 正确性验证通过\n";
    else std::cout << "   [FAIL] 正确性验证失败!\n";

    // 2. 核心组件对比: 量化 vs 采样
    auto [pk, sk] = mlwq_keygen(seed_A, seed_d);
    poly_matrix A; xof_expand_matrix(A, seed_A, params::Q);
    poly_vec r = random_poly_vec_eta(params::K, params::ETA);
    
    poly_vec vec_input = poly_matrix_vec_mul(A, sk.s); // K x N
    poly poly_input = random_poly_uniform(params::Q);  // 1 x N
    
    poly_vec d_vec; xof_expand_poly_vec(d_vec, seed_d, params::K, params::Q_OVER_P_PK_FLOOR);
    poly d_poly = d_vec[0];

    std::vector<unsigned long long> t_qv, t_qp, t_sv, t_sp;

    for(int i=0; i<rounds; i++) {
        // --- 量化 (M-LWQ) ---
        auto c1 = start_cycles();
        volatile auto res_qv = poly_vec_quantize(vec_input, d_vec, params::P_PK);
        t_qv.push_back(stop_cycles() - c1);

        auto c2 = start_cycles();
        volatile auto res_qp = poly_quantize(poly_input, d_poly, params::P_V);
        t_qp.push_back(stop_cycles() - c2);

        // --- 采样 (LWE) ---
        auto c3 = start_cycles();
        volatile auto res_sv = random_poly_vec_eta(params::K, params::ETA);
        t_sv.push_back(stop_cycles() - c3);

        auto c4 = start_cycles();
        volatile auto res_sp = random_poly_eta(params::ETA);
        t_sp.push_back(stop_cycles() - c4);
    }

    res.t_quant_vec = calculate_average(t_qv);
    res.t_quant_poly = calculate_average(t_qp);
    res.t_sample_vec = calculate_average(t_sv);
    res.t_sample_poly = calculate_average(t_sp);

    return res;
}

int main() {
    ntt::init_tables();
    
    std::cout << "==========================================================\n";
    std::cout << "       M-LWQ PKE: 全面性能对比 (Quantize vs Sample)\n";
    std::cout << "==========================================================\n";
    std::cout << "参数集: " << params::PARAM_SET_NAME << " (N=" << params::N << ", K=" << params::K << ")\n";

    int rounds = 100;
    BenchResult scalar = run_full_benchmark(false, rounds);
    BenchResult avx = run_full_benchmark(true, rounds);

    // --- 最终对比报表 ---
    std::cout << "\n\n";
    std::cout << "==============================================================================================\n";
    std::cout << "                       核心组件性能对比 (Cycles) | Lower is Better\n";
    std::cout << "==============================================================================================\n";
    
    // 打印表头
    std::cout << std::left << std::setw(12) << "Component" 
              << std::setw(12) << "Mode"
              << std::setw(16) << "Quantize(M-LWQ)" 
              << std::setw(16) << "Sample(LWE)" 
              << std::setw(22) << "Alg. Efficiency"
              << std::setw(20) << "AVX Improvement"
              << std::endl;
    
    std::cout << std::left << std::setw(12) << "" 
              << std::setw(12) << ""
              << std::setw(16) << "(Cycles)" 
              << std::setw(16) << "(Cycles)" 
              << std::setw(22) << "(Sample / Quant)"
              << std::setw(20) << "(Scalar / AVX)"
              << std::endl;
    std::cout << "----------------------------------------------------------------------------------------------\n";

    // Lambda: 打印一行数据
    auto print_line = [&](std::string item, std::string mode, double t_quant, double t_sample, double ref_quant_scalar) {
        double alg_eff = (t_quant > 0) ? (t_sample / t_quant) : 0.0;
        
        std::stringstream ss_avx;
        if (ref_quant_scalar < 0) {
            ss_avx << "1.00x (Ref)"; // Scalar 模式下自己是基准
        } else {
            double speedup = (t_quant > 0) ? (ref_quant_scalar / t_quant) : 0.0;
            ss_avx << std::fixed << std::setprecision(2) << speedup << "x";
        }

        std::cout << std::left << std::setw(12) << item 
                  << std::setw(12) << mode
                  << std::setw(16) << (long long)t_quant 
                  << std::setw(16) << (long long)t_sample 
                  << std::setw(22) << (std::to_string((int)alg_eff) + "." + std::to_string((int)((alg_eff-(int)alg_eff)*100)).substr(0,2) + "x")
                  << std::setw(20) << ss_avx.str()
                  << std::endl;
    };

    // 1. PK/u 对比 (向量 K x N)
    print_line("PK / u", "Scalar", scalar.t_quant_vec, scalar.t_sample_vec, -1.0);
    print_line("PK / u", "AVX2",   avx.t_quant_vec,    avx.t_sample_vec,    scalar.t_quant_vec);

    std::cout << "----------------------------------------------------------------------------------------------\n";

    // 2. v 对比 (标量 1 x N)
    print_line("v (Poly)", "Scalar", scalar.t_quant_poly, scalar.t_sample_poly, -1.0);
    print_line("v (Poly)", "AVX2",   avx.t_quant_poly,    avx.t_sample_poly,    scalar.t_quant_poly);

    std::cout << "==============================================================================================\n";
    std::cout << "说明:\n";
    std::cout << "1. Alg. Efficiency (Sample / Quant): 表示在当前模式下，Quantize 比 Sample 快多少倍。\n";
    std::cout << "2. AVX Improvement (Scalar / AVX):   表示 AVX2 版本的 Quantize 比 Scalar 版本快多少倍。\n";


    // --- 完整流程报表 ---
    std::cout << "\n\n";
    std::cout << "==============================================================================================\n";
    std::cout << "                       完整流程加速比 (Scalar vs AVX2)\n";
    std::cout << "==============================================================================================\n";
    std::cout << std::left << std::setw(20) << "Operation" 
              << std::setw(18) << "Scalar Cycles" 
              << std::setw(18) << "AVX2 Cycles" 
              << "Speedup (Scalar / AVX2)" << std::endl;
    std::cout << "----------------------------------------------------------------------------------------------\n";
    
    auto print_flow = [&](std::string n, double s, double a) {
        std::cout << std::left << std::setw(20) << n 
                  << std::setw(18) << (long long)s 
                  << std::setw(18) << (long long)a 
                  << std::fixed << std::setprecision(2) << (s/a) << "x" << std::endl;
    };
    
    print_flow("KeyGen", scalar.t_keygen, avx.t_keygen);
    print_flow("Encrypt", scalar.t_encrypt, avx.t_encrypt);
    print_flow("Decrypt", scalar.t_decrypt, avx.t_decrypt);
    std::cout << "==============================================================================================\n";

    return 0;
}