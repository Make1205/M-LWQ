#include "xof.hpp"
#include "sha3.hpp"     // 标量实现
#include "keccak4x.hpp" // AVX2 并行实现
#include <stdexcept>

// ==========================================
// Shake128 Implementation (Scalar)
// ==========================================

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

// ==========================================
// Shake128x4 Implementation (AVX2)
// ==========================================

Shake128x4::Shake128x4() {
    state = std::make_unique<Keccak4x_State>();
}

Shake128x4::~Shake128x4() = default;

void Shake128x4::update4(const std::vector<uint8_t>& d0, const std::vector<uint8_t>& d1,
                         const std::vector<uint8_t>& d2, const std::vector<uint8_t>& d3) {
    // 简单缓存，直到 finalize 时一次性并行 Absorb
    seeds[0] = d0; 
    seeds[1] = d1; 
    seeds[2] = d2; 
    seeds[3] = d3;
}

void Shake128x4::finalize4() {
    // 调用 AVX2 优化的 Absorb 函数
    // 假设所有种子长度相同 (在 KeyGen 中成立)
    shake128x4_absorb_once(state.get(), 
                           seeds[0].data(), seeds[1].data(), 
                           seeds[2].data(), seeds[3].data(), 
                           seeds[0].size());
}

void Shake128x4::digest4(std::vector<uint8_t>& out0, std::vector<uint8_t>& out1,
                         std::vector<uint8_t>& out2, std::vector<uint8_t>& out3, size_t len) {
    // 计算需要的 block 数量 (SHAKE128 Rate = 168 bytes)
    constexpr size_t rate = 168;
    size_t nblocks = (len + rate - 1) / rate;
    
    // 预分配足够容纳 nblocks 的空间
    std::vector<uint8_t> buf0(nblocks * rate);
    std::vector<uint8_t> buf1(nblocks * rate);
    std::vector<uint8_t> buf2(nblocks * rate);
    std::vector<uint8_t> buf3(nblocks * rate);
    
    // 调用 AVX2 优化的 Squeeze 函数
    shake128x4_squeezeblocks(buf0.data(), buf1.data(), buf2.data(), buf3.data(), 
                             nblocks, state.get());
                             
    // 截取所需的长度并输出
    out0.assign(buf0.begin(), buf0.begin() + len);
    out1.assign(buf1.begin(), buf1.begin() + len);
    out2.assign(buf2.begin(), buf2.begin() + len);
    out3.assign(buf3.begin(), buf3.begin() + len);
}

// ==========================================
// 辅助函数 (Helpers)
// ==========================================

// 将字节流解析为多项式系数 (int16)
static void bytes_to_poly(poly& p, const std::vector<uint8_t>& bytes, int32_t modulus) {
    p.resize(params::N);
    for (size_t i = 0; i < params::N; ++i) {
        // Little-endian load 16-bit
        uint16_t val = static_cast<uint16_t>(bytes[2*i]) | 
                       (static_cast<uint16_t>(bytes[2*i + 1]) << 8);
        
        // 简单的模约简 (注意：这里为了通用性用了取模，Kyber通常用 rejection sampling)
        // 在 M-LWQ 演示中，为了保持确定性和简单，我们直接取模
        p[i] = static_cast<int16_t>(val % modulus);
    }
}

// 标量版: 扩展向量 (用于 dither)
void xof_expand_poly_vec(poly_vec& v, const std::vector<uint8_t>& seed, int32_t k, int32_t modulus) {
    v.resize(k);
    Shake128 shake;
    shake.update(seed);
    shake.finalize();
    
    size_t bytes_needed = 2 * params::N;
    std::vector<uint8_t> raw_bytes;
    
    for (int i = 0; i < k; ++i) {
        shake.digest(raw_bytes, bytes_needed);
        bytes_to_poly(v[i], raw_bytes, modulus);
    }
}

// 标量版: 扩展矩阵 A (串行生成)
void xof_expand_matrix(poly_matrix& A, const std::vector<uint8_t>& seed, int32_t modulus) {
    A.resize(params::K, poly_vec(params::K));
    Shake128 shake;
    shake.update(seed);
    shake.finalize();
    
    size_t bytes_needed = 2 * params::N;
    std::vector<uint8_t> raw_bytes;
    
    for (int i = 0; i < params::K; ++i) {
        for (int j = 0; j < params::K; ++j) {
            shake.digest(raw_bytes, bytes_needed);
            bytes_to_poly(A[i][j], raw_bytes, modulus);
        }
    }
}

// [AVX2 加速] 并行扩展矩阵 A
// 利用 Keccak-x4 同时生成 4 个多项式
void xof_expand_matrix_x4(poly_matrix& A, const std::vector<uint8_t>& seed, int32_t modulus) {
    A.resize(params::K, poly_vec(params::K));
    
    // 注意：此函数假设 K=2 (总共 2x2=4 个多项式)，正好填满 4-way AVX2。
    // 如果 K > 2，需要分批次调用。为了 M-LWQ-512 (K=2) 演示，我们只处理一次。
    if (params::K != 2) {
        // 回退到标量 (或者在这里实现循环分块)
        xof_expand_matrix(A, seed, modulus);
        return;
    }

    // 构造 4 个独立的种子 (模拟 Kyber 方式: seed || nonce)
    // A[0][0] -> seed || 0
    // A[0][1] -> seed || 1
    // A[1][0] -> seed || 2
    // A[1][1] -> seed || 3
    std::vector<uint8_t> s0 = seed; s0.push_back(0);
    std::vector<uint8_t> s1 = seed; s1.push_back(1);
    std::vector<uint8_t> s2 = seed; s2.push_back(2);
    std::vector<uint8_t> s3 = seed; s3.push_back(3);

    Shake128x4 shake4;
    shake4.update4(s0, s1, s2, s3);
    shake4.finalize4();

    size_t bytes_needed = 2 * params::N;
    std::vector<uint8_t> b0, b1, b2, b3;
    
    // 并行生成随机字节流
    shake4.digest4(b0, b1, b2, b3, bytes_needed);

    // 解析为多项式
    // A 矩阵布局:
    // [ A00  A01 ]
    // [ A10  A11 ]
    bytes_to_poly(A[0][0], b0, modulus);
    bytes_to_poly(A[0][1], b1, modulus);
    bytes_to_poly(A[1][0], b2, modulus);
    bytes_to_poly(A[1][1], b3, modulus);
}