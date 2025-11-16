#pragma once
#include <vector>
#include <cstdint>
#include <memory> // for std::unique_ptr
#include "poly.hpp" // for 'poly_vec', 'poly_matrix'

//
// === 修改: C++ 风格的 struct 前向声明 ===
//
struct sha3_context;

/**
 * @brief C++ RAII 包装器
 */
class Shake128 {
public:
    Shake128();
    ~Shake128();
    
    // Non-copyable
    Shake128(const Shake128&) = delete;
    Shake128& operator=(const Shake128&) = delete;

    void update(const std::vector<uint8_t>& data);
    void finalize(); // Enters squeezing mode
    void digest(std::vector<uint8_t>& output, size_t len); // Squeeze

private:
    std::unique_ptr<sha3_context> ctx;
};

// --- M-LWQ 辅助函数 ---

/**
 * @brief 从 XOF (SHAKE-128) 扩展一个 k x k 的多项式矩阵 A
 */
void xof_expand_matrix(poly_matrix& A, 
                       const std::vector<uint8_t>& seed, 
                       int32_t modulus);

/**
 * @brief 从 XOF (SHAKE-128) 扩展一个 k x 1 的多项式向量 d
 */
void xof_expand_poly_vec(poly_vec& v, 
                         const std::vector<uint8_t>& seed, 
                         int32_t k,
                         int32_t modulus);