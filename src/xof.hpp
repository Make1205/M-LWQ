#pragma once
#include <vector>
#include <cstdint>
#include <memory>
#include "poly.hpp"

// 前向声明，隐藏具体实现细节 (Pimpl)
struct sha3_context;
struct Keccak4x_State; 

/**
 * @brief 单路 SHAKE-128 包装器 (Scalar)
 */
class Shake128 {
public:
    Shake128();
    ~Shake128();
    
    Shake128(const Shake128&) = delete;
    Shake128& operator=(const Shake128&) = delete;

    void update(const std::vector<uint8_t>& data);
    void finalize(); 
    void digest(std::vector<uint8_t>& output, size_t len); 

private:
    std::unique_ptr<sha3_context> ctx;
};

/**
 * @brief 4路并行 SHAKE-128 包装器 (AVX2)
 * 用于加速矩阵 A 的生成
 */
class Shake128x4 {
public:
    Shake128x4();
    ~Shake128x4();

    // 同时 Update 4 个不同的种子流
    void update4(const std::vector<uint8_t>& d0,
                 const std::vector<uint8_t>& d1,
                 const std::vector<uint8_t>& d2,
                 const std::vector<uint8_t>& d3);

    // 准备进入挤压阶段 (Absorb)
    void finalize4();

    // 并行挤出数据到 4 个输出缓冲区
    void digest4(std::vector<uint8_t>& out0,
                 std::vector<uint8_t>& out1,
                 std::vector<uint8_t>& out2,
                 std::vector<uint8_t>& out3, 
                 size_t len);

private:
    std::unique_ptr<Keccak4x_State> state;
    // 临时缓存种子，以便在 finalize 时一次性 absorb
    std::vector<uint8_t> seeds[4];
};

// --- M-LWQ 辅助函数 ---

// 标量版: 扩展矩阵 A
void xof_expand_matrix(poly_matrix& A, 
                       const std::vector<uint8_t>& seed, 
                       int32_t modulus);

// 标量版: 扩展向量 (用于 dither)
void xof_expand_poly_vec(poly_vec& v, 
                         const std::vector<uint8_t>& seed, 
                         int32_t k,
                         int32_t modulus);

// [新增] AVX2并行版: 扩展矩阵 A (4路并行)
void xof_expand_matrix_x4(poly_matrix& A, 
                          const std::vector<uint8_t>& seed, 
                          int32_t modulus);