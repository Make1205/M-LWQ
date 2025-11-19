#pragma once
#include <vector>
#include <cstdint>
#include <memory>
#include <array>
#include "poly.hpp"

struct sha3_context;

// 单路 SHAKE (保持不变)
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

// [新增] 4路并行 SHAKE
// 用于同时生成矩阵 A 的 4 个多项式 (K=2 时正好填满)
class Shake128x4 {
public:
    Shake128x4();
    ~Shake128x4();

    // 并行 Update: 同时吸收 4 个种子
    void update4(const std::vector<uint8_t>& d0,
                 const std::vector<uint8_t>& d1,
                 const std::vector<uint8_t>& d2,
                 const std::vector<uint8_t>& d3);

    // 并行 Finalize
    void finalize4();

    // 并行 Digest: 同时输出到 4 个缓冲区
    void digest4(std::vector<uint8_t>& out0,
                 std::vector<uint8_t>& out1,
                 std::vector<uint8_t>& out2,
                 std::vector<uint8_t>& out3, 
                 size_t len);

private:
    // 内部维护 4 个上下文 (在真正的 AVX2 实现中，这将是一个 __m256i state[25])
    std::unique_ptr<sha3_context> ctx[4];
};

// --- 辅助函数 ---
void xof_expand_matrix(poly_matrix& A, 
                       const std::vector<uint8_t>& seed, 
                       int32_t modulus);

void xof_expand_poly_vec(poly_vec& v, 
                         const std::vector<uint8_t>& seed, 
                         int32_t k,
                         int32_t modulus);

// [新增] 并行矩阵扩展函数
void xof_expand_matrix_x4(poly_matrix& A, 
                          const std::vector<uint8_t>& seed, 
                          int32_t modulus);