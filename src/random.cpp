#include "random.hpp"
#include <random>

namespace {
    std::mt19937& get_rng() {
        static std::mt19937 rng{std::random_device{}()};
        return rng;
    }
}

poly random_poly_uniform(int32_t modulus) {
    std::uniform_int_distribution<int32_t> dist(0, modulus - 1);
    poly p(params::N);
    for (int i = 0; i < params::N; ++i) {
        p[i] = dist(get_rng());
    }
    return p;
}

poly random_poly_eta(int32_t eta) {
    // 近似 B_eta: 均匀分布 U(-eta, +eta)
    std::uniform_int_distribution<int32_t> dist(-eta, eta);
    poly p(params::N);
    for (int i = 0; i < params::N; ++i) {
        p[i] = dist(get_rng());
    }
    return p;
}

poly_vec random_poly_vec_eta(int32_t k, int32_t eta) {
    poly_vec pv(k);
    for (int i = 0; i < k; ++i) {
        pv[i] = random_poly_eta(eta);
    }
    return pv;
}