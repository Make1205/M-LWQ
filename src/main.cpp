#include <iostream>
#include <vector>
#include <string>
#include <cmath>     
#include <numeric>   
#include <iomanip>   
#include <sstream>

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

// --- 综合结果结构体 ---
struct BenchResult {
    // 0. 正确性状态
    bool all_passed = true;

    // 1. 内部耗时拆解 (Breakdown)
    MlwqProfiling stats; 
    
    // 2. 完整流程总耗时 (Full Flow)
    double t_total_kg = 0;
    double t_total_enc = 0;
    double t_total_dec = 0;
    
    // 3. 核心组件独立测试 (Component)
    double t_quant_vec = 0;   // PK / u
    double t_quant_poly = 0;  // v
    double t_sample_vec = 0; 
    double t_sample_poly = 0;
};

BenchResult run_benchmark(bool use_avx, int rounds) {
    params::USE_AVX2 = use_avx;
    BenchResult res;
    res.stats.reset();
    res.all_passed = true;

    std::cout << ">>> Running: " << (use_avx ? "AVX2 Mode" : "Scalar Mode") << " (" << rounds << " rounds)...\n";

    std::vector<uint8_t> seed_A = string_to_seed("seed_A");
    std::vector<uint8_t> seed_d = string_to_seed("seed_d");
    std::vector<uint8_t> seed_ct = string_to_seed("seed_ct");
    poly m_original = create_test_message();

    // Warmup
    mlwq_keygen(seed_A, seed_d);

    // ==========================================
    // Part A: 完整流程 & 内部拆解 & 正确性检查
    // ==========================================
    uint64_t sum_kg = 0, sum_enc = 0, sum_dec = 0;

    for (int i = 0; i < rounds; ++i) {
        // KeyGen
        auto s1 = start_cycles();
        auto keys = mlwq_keygen(seed_A, seed_d, &res.stats);
        sum_kg += (stop_cycles() - s1);

        // Encrypt
        auto s2 = start_cycles();
        auto ct = mlwq_encrypt(keys.first, m_original, seed_ct, &res.stats);
        sum_enc += (stop_cycles() - s2);

        // Decrypt
        auto s3 = start_cycles();
        // [修复] 必须接收返回值 m_dec
        poly m_dec = mlwq_decrypt(keys.second, ct, &res.stats);
        sum_dec += (stop_cycles() - s3);

        // [修复] 立即检查正确性
        if (!check_poly_eq(m_original, m_dec)) {
            res.all_passed = false;
            // 可选：出错时打印一次警告
            if (i == 0) std::cerr << "   [ERROR] Decryption mismatch detected!\n";
        }
    }

    // 打印正确性结果
    if (res.all_passed) std::cout << "   [PASS] Correctness verified.\n";
    else std::cout << "   [FAIL] Correctness check FAILED!\n";

    // 计算均值
    res.t_total_kg = (double)sum_kg / rounds;
    res.t_total_enc = (double)sum_enc / rounds;
    res.t_total_dec = (double)sum_dec / rounds;
    
    // Stats 均值化
    auto div = [&](unsigned long long& val) { val /= rounds; };
    div(res.stats.kg_gen_A); div(res.stats.kg_sample_s); div(res.stats.kg_gen_d); div(res.stats.kg_arith_as); div(res.stats.kg_quant);
    div(res.stats.enc_gen_A); div(res.stats.enc_sample_r); div(res.stats.enc_gen_d); div(res.stats.enc_arith_u); div(res.stats.enc_arith_v); div(res.stats.enc_quant);
    div(res.stats.dec_dequant); div(res.stats.dec_mul_sub); div(res.stats.dec_decode);


    // ==========================================
    // Part B: 核心组件独立测试 (Quantize vs Sample)
    // ==========================================
    auto [pk, sk] = mlwq_keygen(seed_A, seed_d); 
    poly_matrix A; xof_expand_matrix(A, seed_A, params::Q);
    poly_vec vec_input = poly_matrix_vec_mul(A, sk.s); 
    poly poly_input = random_poly_uniform(params::Q);  
    poly_vec d_vec; xof_expand_poly_vec(d_vec, seed_d, params::K, params::Q_OVER_P_PK_FLOOR);
    poly d_poly = d_vec[0];

    std::vector<unsigned long long> v_qv, v_qp, v_sv, v_sp;

    for(int i=0; i<rounds; i++) {
        // Quantize Vector
        auto c1 = start_cycles();
        volatile auto r1 = poly_vec_quantize(vec_input, d_vec, params::P_PK);
        v_qv.push_back(stop_cycles() - c1);

        // Quantize Poly
        auto c2 = start_cycles();
        volatile auto r2 = poly_quantize(poly_input, d_poly, params::P_V);
        v_qp.push_back(stop_cycles() - c2);

        // Sample Vector
        auto c3 = start_cycles();
        volatile auto r3 = random_poly_vec_eta(params::K, params::ETA);
        v_sv.push_back(stop_cycles() - c3);

        // Sample Poly
        auto c4 = start_cycles();
        volatile auto r4 = random_poly_eta(params::ETA);
        v_sp.push_back(stop_cycles() - c4);
    }

    res.t_quant_vec = calculate_average(v_qv);
    res.t_quant_poly = calculate_average(v_qp);
    res.t_sample_vec = calculate_average(v_sv);
    res.t_sample_poly = calculate_average(v_sp);

    return res;
}

// 打印拆解表的辅助函数
void print_breakdown(const std::string& stage, 
                     const std::vector<std::pair<std::string, uint64_t>>& parts_scalar,
                     const std::vector<std::pair<std::string, uint64_t>>& parts_avx) {
    std::cout << "--------------------------------------------------------------------------------------\n";
    std::cout << " " << stage << " Breakdown (Detailed)\n";
    std::cout << "--------------------------------------------------------------------------------------\n";
    std::cout << std::left << std::setw(20) << "Sub-Component" 
              << std::setw(15) << "Scalar (cyc)" 
              << std::setw(15) << "AVX2 (cyc)" 
              << std::setw(15) << "Speedup" 
              << "Scalar %" << std::endl;
    
    uint64_t tot_s = 0; for(auto& p : parts_scalar) tot_s += p.second;
    
    for (size_t i = 0; i < parts_scalar.size(); ++i) {
        double s = (double)parts_scalar[i].second;
        double a = (double)parts_avx[i].second;
        double ratio = (a > 0) ? s / a : 0.0;
        double pct = (tot_s > 0) ? (s / tot_s) * 100.0 : 0.0;
        
        std::cout << std::left << std::setw(20) << parts_scalar[i].first 
                  << std::setw(15) << (long long)s
                  << std::setw(15) << (long long)a
                  << std::fixed << std::setprecision(2) << ratio << "x"
                  << std::setw(10) << " " << std::setprecision(1) << pct << "%"
                  << std::endl;
    }
    std::cout << std::endl;
}

int main() {
    ntt::init_tables();
    std::cout << "=== M-LWQ Comprehensive Performance Report ===\n";
    std::cout << "N=" << params::N << ", K=" << params::K << "\n\n";

    int rounds = 1000;
    // 运行测试
    BenchResult scalar = run_benchmark(false, rounds);
    BenchResult avx = run_benchmark(true, rounds);

    // =================================================================
    // 1. 内部耗时拆解 (Breakdown)
    // =================================================================
    std::cout << "\n>>> PART 1: Internal Breakdown (Where is time spent?)\n\n";

    print_breakdown("KeyGen", 
        {{"GenMatrix (A)", scalar.stats.kg_gen_A}, {"Sample (s)", scalar.stats.kg_sample_s}, 
         {"GenDither", scalar.stats.kg_gen_d}, {"Arith (A*s)", scalar.stats.kg_arith_as}, {"Quantize", scalar.stats.kg_quant}},
        {{"GenMatrix (A)", avx.stats.kg_gen_A}, {"Sample (s)", avx.stats.kg_sample_s}, 
         {"GenDither", avx.stats.kg_gen_d}, {"Arith (A*s)", avx.stats.kg_arith_as}, {"Quantize", avx.stats.kg_quant}}
    );

    print_breakdown("Encrypt", 
        {{"GenMatrix (A)", scalar.stats.enc_gen_A}, {"Sample (r)", scalar.stats.enc_sample_r}, 
         {"GenDither", scalar.stats.enc_gen_d}, {"Arith (u)", scalar.stats.enc_arith_u}, 
         {"Arith (v)", scalar.stats.enc_arith_v}, {"Quantize", scalar.stats.enc_quant}},
        {{"GenMatrix (A)", avx.stats.enc_gen_A}, {"Sample (r)", avx.stats.enc_sample_r}, 
         {"GenDither", avx.stats.enc_gen_d}, {"Arith (u)", avx.stats.enc_arith_u}, 
         {"Arith (v)", avx.stats.enc_arith_v}, {"Quantize", avx.stats.enc_quant}}
    );

    print_breakdown("Decrypt", 
        {{"DeQuantize", scalar.stats.dec_dequant}, {"Arith (v-su)", scalar.stats.dec_mul_sub}, {"Decode", scalar.stats.dec_decode}},
        {{"DeQuantize", avx.stats.dec_dequant}, {"Arith (v-su)", avx.stats.dec_mul_sub}, {"Decode", avx.stats.dec_decode}}
    );

    // =================================================================
    // 2. 核心组件对比 (Quantize vs Sample)
    // =================================================================
    std::cout << "\n>>> PART 2: Core Component Comparison (Quantize vs Sample)\n";
    std::cout << "----------------------------------------------------------------------------------------------\n";
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

    // =================================================================
    // 3. 完整流程总结
    // =================================================================
    std::cout << "\n\n>>> PART 3: Full Flow Summary (Total Time)\n";
    std::cout << "----------------------------------------------------------------------------------------------\n";
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
    
    print_flow("KeyGen", scalar.t_total_kg, avx.t_total_kg);
    print_flow("Encrypt", scalar.t_total_enc, avx.t_total_enc);
    print_flow("Decrypt", scalar.t_total_dec, avx.t_total_dec);
    std::cout << "----------------------------------------------------------------------------------------------\n";

    if (scalar.all_passed && avx.all_passed) {
        std::cout << "\n[FINAL] All checks passed! Implementation is correct.\n";
    } else {
        std::cout << "\n[FINAL] ERROR: Some correctness checks failed.\n";
    }

    return 0;
}