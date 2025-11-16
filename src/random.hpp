#pragma once
#include "poly.hpp"

/**
 * @brief 生成一个 [0, modulus-1] 范围的随机多项式
 */
poly random_poly_uniform(int32_t modulus);

/**
 * @brief 从中心二项分布 B_eta 采样一个多项式 (近似)
 * 系数在 [-eta, +eta] 
 */
poly random_poly_eta(int32_t eta);

/**
 * @brief 从 B_eta 采样一个 k x 1 的多项式向量
 */
poly_vec random_poly_vec_eta(int32_t k, int32_t eta);