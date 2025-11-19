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
    
    // 完整流程 (包含 SHAKE)
    double t_keygen = 0;
    double t_encrypt = 0;
    double t_decrypt = 0;
    
    // 组件 (Quantize vs Sample)
    double t_quant_vec = 0;
    double t_quant_poly = 0;
    double t_sample_vec = 0; 
    double t_sample_poly = 0;
};

BenchResult run_full_benchmark(bool use_avx, int rounds) {
    params::USE_AVX2 = use_avx;
    BenchResult res;
    res.mode_name = use_avx ? "AVX2 Mode" : "Scalar Mode";

    std::cout << "\n>>> 正在运行: " << res.mode_name << " (" << rounds << " 轮) <<<\n";

    std::vector<uint8_t> seed_A = string_to_seed("seed_A_BENCH");
    std::vector<uint8_t> seed_d = string_to_seed("seed_d_BENCH");
    std::vector<uint8_t> seed_ct = string_to_seed("seed_ct_BENCH");
    poly m_original = create_test_message();

    // =================================================================
    // 1. 完整流程 (KeyGen -> Enc -> Dec)
    // =================================================================
    std::vector<unsigned long long> t_kg, t_enc, t_dec;
    bool check_success = true;

    // Warmup
    mlwq_keygen(seed_A, seed_d);

    for (int i = 0; i < rounds; ++i) {
        // --- KeyGen (Full) ---
        auto s1 = start_cycles();
        auto keys = mlwq_keygen(seed_A, seed_d);
        t_kg.push_back(stop_cycles() - s1);

        // --- Encrypt (Full) ---
        auto s2 = start_cycles();
        auto ct = mlwq_encrypt(keys.first, m_original, seed_ct);
        t_enc.push_back(stop_cycles() - s2);

        // --- Decrypt (Full) ---
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

    // =================================================================
    // 2. 组件对比: 量化 vs 采样
    // =================================================================
    auto [pk, sk] = mlwq_keygen(seed_A, seed_d); 
    poly_matrix A; xof_expand_matrix(A, seed_A, params::Q);
    poly_vec vec_input = poly_matrix_vec_mul(A, sk.s); 
    poly poly_input = random_poly_uniform(params::Q);  
    poly_vec d_vec; xof_expand_poly_vec(d_vec, seed_d, params::K, params::Q_OVER_P_PK_FLOOR);
    poly d_poly = d_vec[0];

    std::vector<unsigned long long> t_qv, t_qp, t_sv, t_sp;

    for(int i=0; i<rounds; i++) {
        // Quantize
        auto c1 = start_cycles();
        volatile auto res_qv = poly_vec_quantize(vec_input, d_vec, params::P_PK);
        t_qv.push_back(stop_cycles() - c1);

        auto c2 = start_cycles();
        volatile auto res_qp = poly_quantize(poly_input, d_poly, params::P_V);
        t_qp.push_back(stop_cycles() - c2);

        // Sample
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
    std::cout << "       M-LWQ PKE: 完整流程与组件性能对比\n";
    std::cout << "==========================================================\n";
    std::cout << "参数集: " << params::PARAM_SET_NAME << " (N=" << params::N << ", K=" << params::K << ")\n";

    int rounds = 10000;
    BenchResult scalar = run_full_benchmark(false, rounds);
    BenchResult avx = run_full_benchmark(true, rounds);

    // ==============================================================================================
    // 1. 先打印：核心组件性能对比 (Cycles)
    // ==============================================================================================
    std::cout << "\n\n";
    std::cout << "==============================================================================================\n";
    std::cout << "                       核心组件性能对比 (Cycles)\n";
    std::cout << "==============================================================================================\n";
    
    std::cout << std::left << std::setw(12) << "Component" 
              << std::setw(12) << "Mode"
              << std::setw(16) << "Quantize" 
              << std::setw(16) << "Sample" 
              << std::setw(22) << "Alg. Efficiency"
              << std::setw(20) << "AVX Improvement"
              << std::endl;
    std::cout << "----------------------------------------------------------------------------------------------\n";

    auto print_line = [&](std::string item, std::string mode, double t_quant, double t_sample, double ref_quant_scalar) {
        double alg_eff = (t_quant > 0) ? (t_sample / t_quant) : 0.0;
        std::stringstream ss_avx;
        if (ref_quant_scalar < 0) ss_avx << "1.00x (Ref)";
        else ss_avx << std::fixed << std::setprecision(2) << (ref_quant_scalar / t_quant) << "x";

        std::cout << std::left << std::setw(12) << item 
                  << std::setw(12) << mode
                  << std::setw(16) << (long long)t_quant 
                  << std::setw(16) << (long long)t_sample 
                  << std::setw(22) << (std::to_string((int)alg_eff) + "." + std::to_string((int)((alg_eff-(int)alg_eff)*100)).substr(0,2) + "x")
                  << std::setw(20) << ss_avx.str()
                  << std::endl;
    };

    print_line("PK / u", "Scalar", scalar.t_quant_vec, scalar.t_sample_vec, -1.0);
    print_line("PK / u", "AVX2",   avx.t_quant_vec,    avx.t_sample_vec,    scalar.t_quant_vec);
    std::cout << "----------------------------------------------------------------------------------------------\n";
    print_line("v (Poly)", "Scalar", scalar.t_quant_poly, scalar.t_sample_poly, -1.0);
    print_line("v (Poly)", "AVX2",   avx.t_quant_poly,    avx.t_sample_poly,    scalar.t_quant_poly);

    // ==============================================================================================
    // 2. 后打印：完整流程加速比 (Scalar vs AVX2)
    // ==============================================================================================
    std::cout << "\n\n";
    std::cout << "==============================================================================================\n";
    std::cout << "                       完整流程加速比 (Include SHAKE GenA)\n";
    std::cout << "==============================================================================================\n";
    std::cout << std::left << std::setw(20) << "Operation" 
              << std::setw(18) << "Scalar Cycles" 
              << std::setw(18) << "AVX2 Cycles" 
              << "Speedup" << std::endl;
    std::cout << "----------------------------------------------------------------------------------------------\n";
    
    auto print_flow = [&](std::string n, double s, double a) {
        std::cout << std::left << std::setw(20) << n 
                  << std::setw(18) << (long long)s 
                  << std::setw(18) << (long long)a 
                  << std::fixed << std::setprecision(2) << (s/a) << "x" << std::endl;
    };
    
    print_flow("KeyGen (Full)", scalar.t_keygen, avx.t_keygen);
    print_flow("Encrypt (Full)", scalar.t_encrypt, avx.t_encrypt);
    print_flow("Decrypt (Full)", scalar.t_decrypt, avx.t_decrypt);
    std::cout << "==============================================================================================\n";

    return 0;
}