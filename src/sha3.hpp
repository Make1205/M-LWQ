/*
 * Copyright (c) 2017, M. Guenther
 * ... (版权信息省略) ...
 */
 
#pragma once
#include <cstdint>
#include <cstddef>

#define SHA3_STATE_SIZE 200 // 1600 bits / 8 bits/byte

// Keccak state
//
// === 修改: 给予 struct 一个名称 ===
//
typedef struct sha3_context {
    uint8_t state[SHA3_STATE_SIZE];
    uint32_t pos;
    uint32_t rate; // Rate in bytes
} sha3_context;

#ifdef __cplusplus
extern "C" {
#endif

// SHAKE-128
void shake128_init(sha3_context *ctx);
void shake128_update(sha3_context *ctx, const void *data, size_t len);
void shake128_xof(sha3_context *ctx); // Finalize for XOF
void shake128_out(sha3_context *ctx, void *out, size_t len); // Read output

#ifdef __cplusplus
}
#endif