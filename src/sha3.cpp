/*
 * Copyright (c) 2017, M. Guenther
 * This is a minimal C implementation of SHAKE-128 based on the reference
 * implementation by the Keccak team.
 */
 
#include "sha3.hpp"
#include <string.h> // for memset, memcpy

// Keccak-f[1600]
static const uint64_t keccak_round_constants[24] = {
    0x0000000000000001ULL, 0x0000000000008082ULL, 0x800000000000808aULL,
    0x8000000080008000ULL, 0x000000000000808bULL, 0x0000000080000001ULL,
    0x8000000080008081ULL, 0x8000000000008009ULL, 0x000000000000008aULL,
    0x0000000000000088ULL, 0x0000000080008009ULL, 0x000000008000000aULL,
    0x000000008000808bULL, 0x800000000000008bULL, 0x8000000000008089ULL,
    0x8000000000008003ULL, 0x8000000000008002ULL, 0x8000000000000080ULL,
    0x000000000000800aULL, 0x800000008000000aULL, 0x8000000080008081ULL,
    0x8000000000008080ULL, 0x0000000080000001ULL, 0x8000000080008008ULL
};

static uint64_t load64(const uint8_t *x) {
    uint64_t r = 0;
    for (int i = 0; i < 8; ++i) {
        r |= (uint64_t)x[i] << (8 * i);
    }
    return r;
}

static void store64(uint8_t *x, uint64_t u) {
    for (int i = 0; i < 8; ++i) {
        x[i] = (uint8_t)(u >> (8 * i));
    }
}

// Keccak-f[1600] permutation
static void keccak_f(uint8_t *state_bytes) {
    uint64_t A[5][5];
    uint64_t C[5], D[5];
    int i, j, round;

    for (i = 0; i < 5; ++i) {
        for (j = 0; j < 5; ++j) {
            A[i][j] = load64(state_bytes + 8 * (5 * j + i));
        }
    }

    auto ROL64 = [](uint64_t x, int s) { return (x << s) | (x >> (64 - s)); };

    for (round = 0; round < 24; ++round) {
        // Theta
        for (i = 0; i < 5; ++i)
            C[i] = A[i][0] ^ A[i][1] ^ A[i][2] ^ A[i][3] ^ A[i][4];
        for (i = 0; i < 5; ++i)
            D[i] = C[(i + 4) % 5] ^ ROL64(C[(i + 1) % 5], 1);
        for (i = 0; i < 5; ++i)
            for (j = 0; j < 5; ++j)
                A[i][j] ^= D[i];

        // Rho and Pi
        uint64_t B[5][5];
        
        // === 修正: 已修复的 Rho/Pi 硬编码 ===
        B[0][0] = A[0][0];
        B[3][0] = ROL64(A[1][0], 43);
        B[1][2] = ROL64(A[2][0], 56);
        B[4][2] = ROL64(A[3][0], 21);
        B[2][1] = ROL64(A[4][0], 14);
        B[0][1] = ROL64(A[0][1], 1);
        B[3][1] = ROL64(A[1][1], 62);
        B[1][3] = ROL64(A[2][1], 28);
        B[4][3] = ROL64(A[3][1], 55);
        B[2][2] = ROL64(A[4][1], 20);
        B[0][2] = ROL64(A[0][2], 41);
        B[3][2] = ROL64(A[1][2], 20);
        B[1][4] = ROL64(A[2][2], 3);
        B[4][4] = ROL64(A[3][2], 45);
        B[2][3] = ROL64(A[4][2], 36);
        B[0][3] = ROL64(A[0][3], 27);
        B[3][3] = ROL64(A[1][3], 36);
        B[1][0] = ROL64(A[2][3], 44);
        B[4][0] = ROL64(A[3][3], 6);
        B[2][4] = ROL64(A[4][3], 61);
        B[0][4] = ROL64(A[0][4], 55);
        B[3][4] = ROL64(A[1][4], 6);
        B[1][1] = ROL64(A[2][4], 14);
        B[4][1] = ROL64(A[3][4], 39);
        B[2][0] = ROL64(A[4][4], 18);

        for (i = 0; i < 5; ++i) for (j = 0; j < 5; ++j) A[i][j] = B[i][j];

        // Chi
        for (j = 0; j < 5; ++j) {
            for (i = 0; i < 5; ++i)
                B[i][j] = A[i][j];
            for (i = 0; i < 5; ++i)
                A[i][j] = B[i][j] ^ ((~B[(i + 1) % 5][j]) & B[(i + 2) % 5][j]);
        }

        // Iota
        A[0][0] ^= keccak_round_constants[round];
    }
    
    for (i = 0; i < 5; ++i) {
        for (j = 0; j < 5; ++j) {
            store64(state_bytes + 8 * (5 * j + i), A[i][j]);
        }
    }
}


// --- Sponge functions ---

static void sha3_init(sha3_context *ctx, uint32_t rate_bytes) {
    memset(ctx->state, 0, SHA3_STATE_SIZE);
    ctx->rate = rate_bytes;
    ctx->pos = 0;
}

static void sha3_absorb(sha3_context *ctx, const uint8_t *data, size_t len) {
    size_t i;
    for (i = 0; i < len; ++i) {
        ctx->state[ctx->pos++] ^= data[i];
        if (ctx->pos == ctx->rate) {
            keccak_f(ctx->state);
            ctx->pos = 0;
        }
    }
}

static void sha3_finalize(sha3_context *ctx, uint8_t pad) {
    ctx->state[ctx->pos] ^= pad;
    ctx->state[ctx->rate - 1] ^= 0x80;
    keccak_f(ctx->state);
    ctx->pos = 0;
}

static void sha3_squeeze(sha3_context *ctx, uint8_t *out, size_t len) {
    size_t i;
    for (i = 0; i < len; ++i) {
        if (ctx->pos == ctx->rate) {
            keccak_f(ctx->state);
            ctx->pos = 0;
        }
        out[i] = ctx->state[ctx->pos++];
    }
}


// --- Public API ---

void shake128_init(sha3_context *ctx) {
    // 1600 - 2 * 128 = 1344 bits = 168 bytes
    sha3_init(ctx, 168);
}

void shake128_update(sha3_context *ctx, const void *data, size_t len) {
    sha3_absorb(ctx, (const uint8_t*)data, len);
}

void shake128_xof(sha3_context *ctx) {
    sha3_finalize(ctx, 0x1F); // 0x1F for SHAKE
}

void shake128_out(sha3_context *ctx, void *out, size_t len) {
    sha3_squeeze(ctx, (uint8_t*)out, len);
}