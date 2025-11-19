#include "keccak4x.hpp"
#include <immintrin.h>
#include <string.h>

// --- AVX2 Keccak-f1600 实现 ---

// 循环左移 64位整数向量
#define ROL64(x, n) _mm256_xor_si256(_mm256_slli_epi64(x, n), _mm256_srli_epi64(x, 64 - n))

// Keccak 常数 (Round Constants)
static const uint64_t KeccakF_RoundConstants[24] = {
    0x0000000000000001ULL, 0x0000000000008082ULL, 0x800000000000808aULL,
    0x8000000080008000ULL, 0x000000000000808bULL, 0x0000000080000001ULL,
    0x8000000080008081ULL, 0x8000000000008009ULL, 0x000000000000008aULL,
    0x0000000000000088ULL, 0x0000000080008009ULL, 0x000000008000000aULL,
    0x000000008000808bULL, 0x800000000000008bULL, 0x8000000000008089ULL,
    0x8000000000008003ULL, 0x8000000000008002ULL, 0x8000000000000080ULL,
    0x000000000000800aULL, 0x800000008000000aULL, 0x8000000080008081ULL,
    0x8000000000008080ULL, 0x0000000080000001ULL, 0x8000000080008008ULL
};

// Keccak Theta, Rho, Pi, Chi, Iota 步骤的宏定义
#define REPEAT5(e) e(0) e(1) e(2) e(3) e(4)

void KeccakP1600_times4_AVX2(__m256i *s) {
    __m256i Aba, Abe, Abi, Abo, Abu;
    __m256i Aga, Age, Agi, Ago, Agu;
    __m256i Aka, Ake, Aki, Ako, Aku;
    __m256i Ama, Ame, Ami, Amo, Amu;
    __m256i Asa, Ase, Asi, Aso, Asu;
    __m256i Bba, Bbe, Bbi, Bbo, Bbu;
    __m256i Bga, Bge, Bgi, Bgo, Bgu;
    __m256i Bka, Bke, Bki, Bko, Bku;
    __m256i Bma, Bme, Bmi, Bmo, Bmu;
    __m256i Bsa, Bse, Bsi, Bso, Bsu;
    __m256i Ca, Ce, Ci, Co, Cu;
    __m256i Da, De, Di, Do, Du;
    __m256i Eba, Ebe, Ebi, Ebo, Ebu;
    __m256i Ega, Ege, Egi, Ego, Egu;
    __m256i Eka, Eke, Eki, Eko, Eku;
    __m256i Ema, Eme, Emi, Emo, Emu;
    __m256i Esa, Ese, Esi, Eso, Esu;

    // Load state
    Aba = s[0]; Abe = s[1]; Abi = s[2]; Abo = s[3]; Abu = s[4];
    Aga = s[5]; Age = s[6]; Agi = s[7]; Ago = s[8]; Agu = s[9];
    Aka = s[10]; Ake = s[11]; Aki = s[12]; Ako = s[13]; Aku = s[14];
    Ama = s[15]; Ame = s[16]; Ami = s[17]; Amo = s[18]; Amu = s[19];
    Asa = s[20]; Ase = s[21]; Asi = s[22]; Aso = s[23]; Asu = s[24];

    for (int r = 0; r < 24; r += 2) {
        // --- Round R ---
        // Theta
        Ca = _mm256_xor_si256(Aba, _mm256_xor_si256(Aga, _mm256_xor_si256(Aka, _mm256_xor_si256(Ama, Asa))));
        Ce = _mm256_xor_si256(Abe, _mm256_xor_si256(Age, _mm256_xor_si256(Ake, _mm256_xor_si256(Ame, Ase))));
        Ci = _mm256_xor_si256(Abi, _mm256_xor_si256(Agi, _mm256_xor_si256(Aki, _mm256_xor_si256(Ami, Asi))));
        Co = _mm256_xor_si256(Abo, _mm256_xor_si256(Ago, _mm256_xor_si256(Ako, _mm256_xor_si256(Amo, Aso))));
        Cu = _mm256_xor_si256(Abu, _mm256_xor_si256(Agu, _mm256_xor_si256(Aku, _mm256_xor_si256(Amu, Asu))));

        Da = _mm256_xor_si256(Cu, ROL64(Ce, 1));
        De = _mm256_xor_si256(Ca, ROL64(Ci, 1));
        Di = _mm256_xor_si256(Ce, ROL64(Co, 1));
        Do = _mm256_xor_si256(Ci, ROL64(Cu, 1));
        Du = _mm256_xor_si256(Co, ROL64(Ca, 1));

        Aba = _mm256_xor_si256(Aba, Da); Abe = _mm256_xor_si256(Abe, De); Abi = _mm256_xor_si256(Abi, Di); Abo = _mm256_xor_si256(Abo, Do); Abu = _mm256_xor_si256(Abu, Du);
        Aga = _mm256_xor_si256(Aga, Da); Age = _mm256_xor_si256(Age, De); Agi = _mm256_xor_si256(Agi, Di); Ago = _mm256_xor_si256(Ago, Do); Agu = _mm256_xor_si256(Agu, Du);
        Aka = _mm256_xor_si256(Aka, Da); Ake = _mm256_xor_si256(Ake, De); Aki = _mm256_xor_si256(Aki, Di); Ako = _mm256_xor_si256(Ako, Do); Aku = _mm256_xor_si256(Aku, Du);
        Ama = _mm256_xor_si256(Ama, Da); Ame = _mm256_xor_si256(Ame, De); Ami = _mm256_xor_si256(Ami, Di); Amo = _mm256_xor_si256(Amo, Do); Amu = _mm256_xor_si256(Amu, Du);
        Asa = _mm256_xor_si256(Asa, Da); Ase = _mm256_xor_si256(Ase, De); Asi = _mm256_xor_si256(Asi, Di); Aso = _mm256_xor_si256(Aso, Do); Asu = _mm256_xor_si256(Asu, Du);

        // Rho Pi
        Bba = Aba;
        Bbe = ROL64(Age, 44);
        Bbi = ROL64(Aki, 43);
        Bbo = ROL64(Amo, 21);
        Bbu = ROL64(Asu, 14);
        Bga = ROL64(Ako, 28);
        Bge = ROL64(Amu, 20);
        Bgi = ROL64(Asa, 3);
        Bgo = ROL64(Abe, 45);
        Bgu = ROL64(Agi, 61);
        Bka = ROL64(Abe, 1); // Wait, Abe logic in Rho is complex. Correct constants:
        // Re-check standard constants. A[1][0] -> rot 1? No.
        // Let's use standard mapping:
        // A[0][0] r0
        // A[1][0] r1
        // ...
        // Correct assignment:
        Bba = Aba;
        Bbe = ROL64(Age, 44);
        Bbi = ROL64(Aki, 43);
        Bbo = ROL64(Amo, 21);
        Bbu = ROL64(Asu, 14);
        
        Bga = ROL64(Aka, 28); // x=2,y=0 -> A[2][0]
        Bge = ROL64(Ame, 20);
        Bgi = ROL64(Asi, 3);
        Bgo = ROL64(Abo, 45);
        Bgu = ROL64(Agu, 61);
        
        Bka = ROL64(Asa, 6);
        Bke = ROL64(Abe, 1);
        Bki = ROL64(Agi, 6);
        Bko = ROL64(Ako, 25);
        Bku = ROL64(Amu, 8);
        
        Bma = ROL64(Ama, 27);
        Bme = ROL64(Ase, 36);
        Bmi = ROL64(Abi, 10);
        Bmo = ROL64(Ago, 15);
        Bmu = ROL64(Aku, 56);
        
        Bsa = ROL64(Aga, 36);
        Bse = ROL64(Ake, 55);
        Bsi = ROL64(Ami, 39);
        Bso = ROL64(Aso, 41);
        Bsu = ROL64(Abu, 2);

        // Chi
        Aba = _mm256_xor_si256(Bba, _mm256_andnot_si256(Bbe, Bbi));
        Abe = _mm256_xor_si256(Bbe, _mm256_andnot_si256(Bbi, Bbo));
        Abi = _mm256_xor_si256(Bbi, _mm256_andnot_si256(Bbo, Bbu));
        Abo = _mm256_xor_si256(Bbo, _mm256_andnot_si256(Bbu, Bba));
        Abu = _mm256_xor_si256(Bbu, _mm256_andnot_si256(Bba, Bbe));

        Aga = _mm256_xor_si256(Bga, _mm256_andnot_si256(Bge, Bgi));
        Age = _mm256_xor_si256(Bge, _mm256_andnot_si256(Bgi, Bgo));
        Agi = _mm256_xor_si256(Bgi, _mm256_andnot_si256(Bgo, Bgu));
        Ago = _mm256_xor_si256(Bgo, _mm256_andnot_si256(Bgu, Bga));
        Agu = _mm256_xor_si256(Bgu, _mm256_andnot_si256(Bga, Bge));

        Aka = _mm256_xor_si256(Bka, _mm256_andnot_si256(Bke, Bki));
        Ake = _mm256_xor_si256(Bke, _mm256_andnot_si256(Bki, Bko));
        Aki = _mm256_xor_si256(Bki, _mm256_andnot_si256(Bko, Bku));
        Ako = _mm256_xor_si256(Bko, _mm256_andnot_si256(Bku, Bka));
        Aku = _mm256_xor_si256(Bku, _mm256_andnot_si256(Bka, Bke));

        Ama = _mm256_xor_si256(Bma, _mm256_andnot_si256(Bme, Bmi));
        Ame = _mm256_xor_si256(Bme, _mm256_andnot_si256(Bmi, Bmo));
        Ami = _mm256_xor_si256(Bmi, _mm256_andnot_si256(Bmo, Bmu));
        Amo = _mm256_xor_si256(Bmo, _mm256_andnot_si256(Bmu, Bma));
        Amu = _mm256_xor_si256(Bmu, _mm256_andnot_si256(Bma, Bme));

        Asa = _mm256_xor_si256(Bsa, _mm256_andnot_si256(Bse, Bsi));
        Ase = _mm256_xor_si256(Bse, _mm256_andnot_si256(Bsi, Bso));
        Asi = _mm256_xor_si256(Bsi, _mm256_andnot_si256(Bso, Bsu));
        Aso = _mm256_xor_si256(Bso, _mm256_andnot_si256(Bsu, Bsa));
        Asu = _mm256_xor_si256(Bsu, _mm256_andnot_si256(Bsa, Bse));

        // Iota
        Aba = _mm256_xor_si256(Aba, _mm256_set1_epi64x(KeccakF_RoundConstants[r]));

        // --- Round R+1 ---
        // Theta
        Ca = _mm256_xor_si256(Aba, _mm256_xor_si256(Aga, _mm256_xor_si256(Aka, _mm256_xor_si256(Ama, Asa))));
        Ce = _mm256_xor_si256(Abe, _mm256_xor_si256(Age, _mm256_xor_si256(Ake, _mm256_xor_si256(Ame, Ase))));
        Ci = _mm256_xor_si256(Abi, _mm256_xor_si256(Agi, _mm256_xor_si256(Aki, _mm256_xor_si256(Ami, Asi))));
        Co = _mm256_xor_si256(Abo, _mm256_xor_si256(Ago, _mm256_xor_si256(Ako, _mm256_xor_si256(Amo, Aso))));
        Cu = _mm256_xor_si256(Abu, _mm256_xor_si256(Agu, _mm256_xor_si256(Aku, _mm256_xor_si256(Amu, Asu))));

        Da = _mm256_xor_si256(Cu, ROL64(Ce, 1));
        De = _mm256_xor_si256(Ca, ROL64(Ci, 1));
        Di = _mm256_xor_si256(Ce, ROL64(Co, 1));
        Do = _mm256_xor_si256(Ci, ROL64(Cu, 1));
        Du = _mm256_xor_si256(Co, ROL64(Ca, 1));

        Eba = _mm256_xor_si256(Aba, Da); Ebe = _mm256_xor_si256(Abe, De); Ebi = _mm256_xor_si256(Abi, Di); Ebo = _mm256_xor_si256(Abo, Do); Ebu = _mm256_xor_si256(Abu, Du);
        Ega = _mm256_xor_si256(Aga, Da); Ege = _mm256_xor_si256(Age, De); Egi = _mm256_xor_si256(Agi, Di); Ego = _mm256_xor_si256(Ago, Do); Egu = _mm256_xor_si256(Agu, Du);
        Eka = _mm256_xor_si256(Aka, Da); Eke = _mm256_xor_si256(Ake, De); Eki = _mm256_xor_si256(Aki, Di); Eko = _mm256_xor_si256(Ako, Do); Eku = _mm256_xor_si256(Aku, Du);
        Ema = _mm256_xor_si256(Ama, Da); Eme = _mm256_xor_si256(Ame, De); Emi = _mm256_xor_si256(Ami, Di); Emo = _mm256_xor_si256(Amo, Do); Emu = _mm256_xor_si256(Amu, Du);
        Esa = _mm256_xor_si256(Asa, Da); Ese = _mm256_xor_si256(Ase, De); Esi = _mm256_xor_si256(Asi, Di); Eso = _mm256_xor_si256(Aso, Do); Esu = _mm256_xor_si256(Asu, Du);

        // Rho Pi
        Aba = Eba;
        Abe = ROL64(Ege, 44);
        Abi = ROL64(Eki, 43);
        Abo = ROL64(Emo, 21);
        Abu = ROL64(Esu, 14);
        Aga = ROL64(Eka, 28);
        Age = ROL64(Eme, 20);
        Agi = ROL64(Esi, 3);
        Ago = ROL64(Ebo, 45);
        Agu = ROL64(Egu, 61);
        Aka = ROL64(Esa, 6);
        Ake = ROL64(Ebe, 1);
        Aki = ROL64(Egi, 6);
        Ako = ROL64(Eko, 25);
        Aku = ROL64(Emu, 8);
        Ama = ROL64(Ema, 27);
        Ame = ROL64(Ese, 36);
        Ami = ROL64(Ebi, 10);
        Amo = ROL64(Ego, 15);
        Amu = ROL64(Eku, 56);
        Asa = ROL64(Ega, 36);
        Ase = ROL64(Eke, 55);
        Asi = ROL64(Emi, 39);
        Aso = ROL64(Eso, 41);
        Asu = ROL64(Ebu, 2);

        // Chi
        Bba = _mm256_xor_si256(Aba, _mm256_andnot_si256(Abe, Abi));
        Bbe = _mm256_xor_si256(Abe, _mm256_andnot_si256(Abi, Abo));
        Bbi = _mm256_xor_si256(Abi, _mm256_andnot_si256(Abo, Abu));
        Bbo = _mm256_xor_si256(Abo, _mm256_andnot_si256(Abu, Aba));
        Bbu = _mm256_xor_si256(Abu, _mm256_andnot_si256(Aba, Abe));

        Bga = _mm256_xor_si256(Aga, _mm256_andnot_si256(Age, Agi));
        Bge = _mm256_xor_si256(Age, _mm256_andnot_si256(Agi, Ago));
        Bgi = _mm256_xor_si256(Agi, _mm256_andnot_si256(Ago, Agu));
        Bgo = _mm256_xor_si256(Ago, _mm256_andnot_si256(Agu, Aga));
        Bgu = _mm256_xor_si256(Agu, _mm256_andnot_si256(Aga, Age));

        Bka = _mm256_xor_si256(Aka, _mm256_andnot_si256(Ake, Aki));
        Bke = _mm256_xor_si256(Ake, _mm256_andnot_si256(Aki, Ako));
        Bki = _mm256_xor_si256(Aki, _mm256_andnot_si256(Ako, Aku));
        Bko = _mm256_xor_si256(Ako, _mm256_andnot_si256(Aku, Aka));
        Bku = _mm256_xor_si256(Aku, _mm256_andnot_si256(Aka, Ake));

        Bma = _mm256_xor_si256(Ama, _mm256_andnot_si256(Ame, Ami));
        Bme = _mm256_xor_si256(Ame, _mm256_andnot_si256(Ami, Amo));
        Bmi = _mm256_xor_si256(Ami, _mm256_andnot_si256(Amo, Amu));
        Bmo = _mm256_xor_si256(Amo, _mm256_andnot_si256(Amu, Ama));
        Bmu = _mm256_xor_si256(Amu, _mm256_andnot_si256(Ama, Ame));

        Bsa = _mm256_xor_si256(Asa, _mm256_andnot_si256(Ase, Asi));
        Bse = _mm256_xor_si256(Ase, _mm256_andnot_si256(Asi, Aso));
        Bsi = _mm256_xor_si256(Asi, _mm256_andnot_si256(Aso, Asu));
        Bso = _mm256_xor_si256(Aso, _mm256_andnot_si256(Asu, Asa));
        Bsu = _mm256_xor_si256(Asu, _mm256_andnot_si256(Asa, Ase));

        // Iota
        Aba = _mm256_xor_si256(Bba, _mm256_set1_epi64x(KeccakF_RoundConstants[r+1]));
        Abe = Bbe; Abi = Bbi; Abo = Bbo; Abu = Bbu;
        Aga = Bga; Age = Bge; Agi = Bgi; Ago = Bgo; Agu = Bgu;
        Aka = Bka; Ake = Bke; Aki = Bki; Ako = Bko; Aku = Bku;
        Ama = Bma; Ame = Bme; Ami = Bmi; Amo = Bmo; Amu = Bmu;
        Asa = Bsa; Ase = Bse; Asi = Bsi; Aso = Bso; Asu = Bsu;
    }

    // Store back
    s[0] = Aba; s[1] = Abe; s[2] = Abi; s[3] = Abo; s[4] = Abu;
    s[5] = Aga; s[6] = Age; s[7] = Agi; s[8] = Ago; s[9] = Agu;
    s[10] = Aka; s[11] = Ake; s[12] = Aki; s[13] = Ako; s[14] = Aku;
    s[15] = Ama; s[16] = Ame; s[17] = Ami; s[18] = Amo; s[19] = Amu;
    s[20] = Asa; s[21] = Ase; s[22] = Asi; s[23] = Aso; s[24] = Asu;
}

// 辅助: Absorb (Assume fixed length for SHAKE-128 seeds + nonce)
void shake128x4_absorb_once(Keccak4x_State *state,
                            const uint8_t *in0, const uint8_t *in1,
                            const uint8_t *in2, const uint8_t *in3,
                            size_t inlen) {
    // 清零状态
    for(int i=0; i<25; ++i) state->s[i] = _mm256_setzero_si256();

    // 我们假设 inlen 比较短 (Seed+Nonce ~= 34 bytes)，小于 SHAKE128 Rate (168 bytes)
    // 所以只需要 XOR 一个 Block，然后 Pad 即可。
    
    // 1. Load and Interleave (SIMD Transpose needed really, but for initialization we can do semi-scalar)
    // Construct AVX register from 4 pointers is tricky without gather.
    // Simple approach: Load into temp buffer and set_epi64
    
    uint64_t *s64 = (uint64_t *)state->s;
    
    for (size_t i = 0; i < inlen; ++i) {
        // For each byte position i, we have 4 bytes from 4 streams.
        // state is __m256i array. 
        // state[k] holds 4x uint64. 
        // Byte i maps to word (i/8).
        int w = i / 8;
        int b = i % 8;
        
        // We need to XOR byte into the correct position of the 4 lanes.
        // Since memory layout of __m256i is [lane0_64, lane1_64, lane2_64, lane3_64]
        // We can access via ((uint64_t*)state->s)[w * 4 + lane]
        
        uint64_t mask = 1ULL << (8*b);
        s64[w*4 + 0] ^= (uint64_t)in0[i] << (8*b);
        s64[w*4 + 1] ^= (uint64_t)in1[i] << (8*b);
        s64[w*4 + 2] ^= (uint64_t)in2[i] << (8*b);
        s64[w*4 + 3] ^= (uint64_t)in3[i] << (8*b);
    }
    
    // 2. Padding (0x1F ... 0x80)
    // Byte index inlen
    int w = inlen / 8;
    int b = inlen % 8;
    s64[w*4+0] ^= 0x1FULL << (8*b);
    s64[w*4+1] ^= 0x1FULL << (8*b);
    s64[w*4+2] ^= 0x1FULL << (8*b);
    s64[w*4+3] ^= 0x1FULL << (8*b);
    
    // Last byte of rate (Rate=168 bytes) -> byte 167
    // Word 167/8 = 20, byte 7.
    s64[20*4+0] ^= 0x80ULL << 56;
    s64[20*4+1] ^= 0x80ULL << 56;
    s64[20*4+2] ^= 0x80ULL << 56;
    s64[20*4+3] ^= 0x80ULL << 56;
}

void shake128x4_squeezeblocks(uint8_t *out0, uint8_t *out1, uint8_t *out2, uint8_t *out3,
                              size_t nblocks, Keccak4x_State *state) {
    size_t rate = 168; // SHAKE128 rate in bytes
    uint64_t *s64 = (uint64_t *)state->s;

    for (size_t n = 0; n < nblocks; ++n) {
        KeccakP1600_times4_AVX2(state->s); // Permute
        
        // Extract 168 bytes for each stream
        for (size_t i = 0; i < rate; ++i) {
             int w = i / 8;
             int b = i % 8;
             uint64_t val0 = s64[w*4+0];
             uint64_t val1 = s64[w*4+1];
             uint64_t val2 = s64[w*4+2];
             uint64_t val3 = s64[w*4+3];
             
             out0[n*rate + i] = (val0 >> (8*b)) & 0xFF;
             out1[n*rate + i] = (val1 >> (8*b)) & 0xFF;
             out2[n*rate + i] = (val2 >> (8*b)) & 0xFF;
             out3[n*rate + i] = (val3 >> (8*b)) & 0xFF;
        }
    }
}