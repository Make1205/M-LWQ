#include "xof.hpp"
#include "sha3.hpp" // C-API
#include <stdexcept>

// --- C++ Wrapper Impl ---

Shake128::Shake128() {
    ctx = std::make_unique<sha3_context>();
    shake128_init(ctx.get());
}

Shake128::~Shake128() = default;

void Shake128::update(const std::vector<uint8_t>& data) {
    shake128_update(ctx.get(), data.data(), data.size());
}

void Shake128::finalize() {
    shake128_xof(ctx.get());
}

void Shake128::digest(std::vector<uint8_t>& output, size_t len) {
    output.resize(len);
    shake128_out(ctx.get(), output.data(), len);
}

// --- M-LWQ 辅助函数实现 ---

// 内部帮助函数
static void xof_expand_poly(poly& p, Shake128& shake, int32_t modulus) {
    p.resize(params::N);
    size_t num_bytes = 2 * params::N;
    std::vector<uint8_t> raw_bytes(num_bytes);
    shake.digest(raw_bytes, num_bytes);
    
    for (size_t i = 0; i < params::N; ++i) {
        uint16_t val_raw = static_cast<uint16_t>(raw_bytes[2*i]) | 
                           (static_cast<uint16_t>(raw_bytes[2*i + 1]) << 8);
        p[i] = val_raw % modulus;
    }
}

// 扩展一个 k x k 矩阵 A
void xof_expand_matrix(poly_matrix& A, 
                       const std::vector<uint8_t>& seed, 
                       int32_t modulus) 
{
    A.resize(params::K, poly_vec(params::K));
    Shake128 shake;
    shake.update(seed);
    shake.finalize();
    
    for (int i = 0; i < params::K; ++i) {
        for (int j = 0; j < params::K; ++j) {
            // 注意：为了可区分性，通常会为 A[i][j] 加入域分隔符
            // 为简单起见，我们按顺序生成
            xof_expand_poly(A[i][j], shake, modulus);
        }
    }
}

// 扩展一个 k x 1 向量 v
void xof_expand_poly_vec(poly_vec& v, 
                         const std::vector<uint8_t>& seed, 
                         int32_t k,
                         int32_t modulus) 
{
    v.resize(k);
    Shake128 shake;
    shake.update(seed);
    shake.finalize();
    
    for (int i = 0; i < k; ++i) {
        xof_expand_poly(v[i], shake, modulus);
    }
}