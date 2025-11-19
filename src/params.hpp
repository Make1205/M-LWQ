#pragma once
#include <cstdint>

// --- 安全级别选择 ---
#define NIST_LEVEL 1 

namespace params {

    // --- 量化模式选择 ---
    enum QuantizationType { QUANT_SCALAR, QUANT_D8 };
    // 默认使用标量量化 (Z)，因为 D8 量化较难向量化
    // constexpr QuantizationType Q_MODE = QUANT_D8;
    constexpr QuantizationType Q_MODE = QUANT_SCALAR;

    // --- 基础参数 ---
    constexpr int32_t N = 256;
    constexpr int32_t Q = 3329;
    constexpr int32_t ETA = 2;
    constexpr int32_t MSG_MODULUS = 2;

    // --- [新增] 全局 AVX2 开关 ---
    // 在 main.cpp 中定义，用于运行时切换模式
    extern bool USE_AVX2;

    // --- 级别特定的参数 ---
#if NIST_LEVEL == 1
    constexpr char PARAM_SET_NAME[] = "M-LWQ-512 (NIST L1)";
    constexpr int32_t K = 2;
    constexpr int32_t D_PK_BITS = 9;
    constexpr int32_t D_U_BITS = 9;
    constexpr int32_t D_V_BITS = 5;

#elif NIST_LEVEL == 3
    constexpr char PARAM_SET_NAME[] = "M-LWQ-768 (NIST L3)";
    constexpr int32_t K = 3;
    constexpr int32_t D_PK_BITS = 10;
    constexpr int32_t D_U_BITS = 9;
    constexpr int32_t D_V_BITS = 5;

#elif NIST_LEVEL == 5
    constexpr char PARAM_SET_NAME[] = "M-LWQ-1024 (NIST L5)";
    constexpr int32_t K = 4;
    constexpr int32_t D_PK_BITS = 11;
    constexpr int32_t D_U_BITS = 9;
    constexpr int32_t D_V_BITS = 6;
#else
    #error "无效的 NIST_LEVEL"
#endif

    // --- 自动派生的参数 ---
    constexpr int32_t P_PK = (1 << D_PK_BITS);
    constexpr int32_t Q_OVER_P_PK_FLOOR = Q / P_PK;
    constexpr int32_t P_U = (1 << D_U_BITS);
    constexpr int32_t Q_OVER_P_U_FLOOR = Q / P_U;
    constexpr int32_t P_V = (1 << D_V_BITS);
    constexpr int32_t Q_OVER_P_V_FLOOR = Q / P_V;

    static_assert(N % 8 == 0, "N 必须是 8 的倍数");
    static_assert((K * N) % 8 == 0, "K*N 必须是 8 的倍数");
}