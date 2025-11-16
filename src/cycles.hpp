#pragma once
#include <cstdint>

// C++ 封装 RDTSC (Read Time-Stamp Counter)
// 这是一个 x86/x86_64 特定的指令
// 注意: 为获得最准确的结果，建议在编译时启用优化 (例如 -O2 或 -O3)
// 并在相对空闲的系统上运行。

#if defined(_MSC_VER)
    // Microsoft Visual C++
    #include <intrin.h>
    #define rdtsc_intrinsic __rdtsc
#elif defined(__GNUC__) || defined(__clang__)
    // GCC and Clang
    #include <x86intrin.h>
    #define rdtsc_intrinsic __rdtsc
#else
    #error "RDTSC intrinsic not supported on this compiler. Please use <chrono> instead."
#endif

/**
 * @brief 标记周期计数的开始
 * @return 当前 CPU 周期数
 */
static inline unsigned long long start_cycles() {
    // _mm_lfence() (Load Fence) 确保所有之前的加载指令都已完成
    // (防止乱序执行影响计时器启动)
    _mm_lfence();
    return rdtsc_intrinsic();
}

/**
 * @brief 标记周期计数的结束
 * @return 当前 CPU 周期数
 */
static inline unsigned long long stop_cycles() {
    // _mm_mfence() (Memory Fence) 确保所有之前的加载和存储指令都已完成
    // (防止乱序执行影响计时器停止)
    _mm_mfence();
    return rdtsc_intrinsic();
}