#pragma once
#include <stdint.h>
#include <immintrin.h>

// 4路并行 Keccak 状态
// 我们用 25 个 __m256i 寄存器来存储 4 个 1600-bit 的状态
// state[i] 的第 k 个 64-bit lane 存储第 k 个 SHAKE 实例的第 i 个 64-bit 字
struct Keccak4x_State {
    __m256i s[25];
};

void KeccakP1600_times4_AVX2(__m256i *s);

void shake128x4_absorb_once(Keccak4x_State *state,
                            const uint8_t *in0,
                            const uint8_t *in1,
                            const uint8_t *in2,
                            const uint8_t *in3,
                            size_t inlen);

void shake128x4_squeezeblocks(uint8_t *out0,
                              uint8_t *out1,
                              uint8_t *out2,
                              uint8_t *out3,
                              size_t nblocks,
                              Keccak4x_State *state);