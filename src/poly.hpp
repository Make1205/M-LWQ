#pragma once
#include <vector>
#include <cstdint>
#include <string>
#include "params.hpp"

// [核心定义] 使用 int16_t
using poly = std::vector<int16_t>;
using poly_vec = std::vector<poly>;
using poly_matrix = std::vector<poly_vec>;

// --- 基础模运算 ---
int16_t positive_mod(int64_t val, int16_t q);

// --- 多项式运算 (mod Q) ---
poly poly_add(const poly& a, const poly& b);
poly poly_sub(const poly& a, const poly& b);
poly poly_mul_mod(const poly& a, const poly& b);

// --- 向量/矩阵运算 (mod Q) ---
poly_vec poly_vec_add(const poly_vec& a, const poly_vec& b);
poly_vec poly_vec_sub(const poly_vec& a, const poly_vec& b);
poly_vec poly_matrix_vec_mul(const poly_matrix& A, const poly_vec& s);
poly poly_vec_transpose_mul(const poly_vec& a_t, const poly_vec& b);
poly_matrix poly_matrix_transpose(const poly_matrix& A);

// --- 量化 & 反量化 ---
poly poly_quantize(const poly& val, const poly& d, int32_t P_param);
poly_vec poly_vec_quantize(const poly_vec& val, const poly_vec& d, int32_t P_param);
poly poly_dequantize(const poly& b, int32_t P_param);
poly_vec poly_vec_dequantize(const poly_vec& b, int32_t P_param);

// --- 消息编码/解码 ---
poly poly_message_encode(const poly& m);
poly poly_message_decode(const poly& val);

// --- 辅助函数 ---
void print_poly(const std::string& name, const poly& p, size_t count = 8);
void print_poly_vec(const std::string& name, const poly_vec& pv, size_t count = 8);
bool check_poly_eq(const poly& a, const poly& b);