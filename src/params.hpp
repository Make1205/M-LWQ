#pragma once
#include <cstdint>

/**
 * @brief 定义 M-LWQ 的全局密码学参数
 * 基于 'samplepaper.pdf' (Table 1, Page 10)
 * n = 256
 * q = 3329
 * eta = 2
 */

// --- 安全级别选择 ---
// *** 在此处切换您想要的NIST安全级别 (1, 3, 或 5) ***
#define NIST_LEVEL 1 // <-- 移至 namespace 外部并改为 #define
// **********************************

namespace params {

    // --- 量化模式选择 ---
    enum QuantizationType { QUANT_SCALAR, QUANT_D8 };
    
    // *** 在此处切换您想要的量化模式 ***
    constexpr QuantizationType Q_MODE = QUANT_D8;
    // constexpr QuantizationType Q_MODE = QUANT_SCALAR;
    // **********************************


    // --- 基础参数 (跨级别通用) ---
    constexpr int32_t N = 256;
    constexpr int32_t Q = 3329;
    constexpr int32_t ETA = 2;
    constexpr int32_t MSG_MODULUS = 2;


    // --- 级别特定的参数 (根据 NIST_LEVEL 自动设置) ---
#if NIST_LEVEL == 1
    // M-LWQ-512 (NIST L1)
    constexpr char PARAM_SET_NAME[] = "M-LWQ-512 (NIST L1)";
    constexpr int32_t K = 2;
    constexpr int32_t D_PK_BITS = 9;  // d_pk = 9
    constexpr int32_t D_U_BITS = 9;   // d_u = 9
    constexpr int32_t D_V_BITS = 5;   // d_v = 5

#elif NIST_LEVEL == 3
    // M-LWQ-768 (NIST L3)
    constexpr char PARAM_SET_NAME[] = "M-LWQ-768 (NIST L3)";
    constexpr int32_t K = 3;
    constexpr int32_t D_PK_BITS = 10; // d_pk = 10
    constexpr int32_t D_U_BITS = 9;   // d_u = 9
    constexpr int32_t D_V_BITS = 5;   // d_v = 5

#elif NIST_LEVEL == 5
    // M-LWQ-1024 (NIST L5)
    constexpr char PARAM_SET_NAME[] = "M-LWQ-1024 (NIST L5)";
    constexpr int32_t K = 4;
    constexpr int32_t D_PK_BITS = 11; // d_pk = 11
    constexpr int32_t D_U_BITS = 9;   // d_u = 9
    constexpr int32_t D_V_BITS = 6;   // d_v = 6

#else
    #error "无效的 NIST_LEVEL (必须是 1, 3, 或 5)"
#endif


    // --- 自动派生的参数 ---
    // 公钥 b_q
    constexpr int32_t P_PK = (1 << D_PK_BITS);
    constexpr int32_t Q_OVER_P_PK_FLOOR = Q / P_PK;

    // 密文 u
    constexpr int32_t P_U = (1 << D_U_BITS);
    constexpr int32_t Q_OVER_P_U_FLOOR = Q / P_U;

    // 密文 v
    constexpr int32_t P_V = (1 << D_V_BITS);
    constexpr int32_t Q_OVER_P_V_FLOOR = Q / P_V;


    // --- 编译时检查 ---
    static_assert(N % 8 == 0, "N 必须是 8 的倍数才能使用 D8 量化");
    // 检查 K*N 是否是 8 的倍数
    static_assert((K * N) % 8 == 0, "K*N (向量总系数) 必须是 8 的倍数");
}