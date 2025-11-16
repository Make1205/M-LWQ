#include "rlwqz.hpp"
#include "xof.hpp"
#include "random.hpp"
#include "params.hpp"
#include <algorithm> // for std::copy

// 对应 'main_show_lwqz.py' 中的密钥生成
std::pair<rlwqz_pk, rlwqz_sk> rlwqz_keygen(const std::vector<uint8_t>& seed_ad) {
    
    rlwqz_pk pk;
    
    // 1. 从种子生成 'a' 和 'd'
    // Python: ad_vector = xof_expand(seed_ad, 2*n, q)
    poly ad_vector;
    xof_expand(ad_vector, seed_ad, 2 * params::N, params::Q); 
    
    pk.a.resize(params::N);
    pk.d.resize(params::N);
    
    // Python: a = ad_vector[:n] % q (xof_expand 已处理 % q)
    std::copy(ad_vector.begin(), 
              ad_vector.begin() + params::N, 
              pk.a.begin());
    
    // Python: d = ad_vector[n:] % (q//p)
    const int32_t mod_d = params::Q_OVER_P_FLOOR; // q/p
    for(int i = 0; i < params::N; ++i) {
        pk.d[i] = ad_vector[params::N + i] % mod_d;
    }
    
    // 2. 生成秘密 s
    // Python: s = np.random.randint(0, q, size=n)
    rlwqz_sk sk = random_poly(params::Q);
    
    // 3. 计算 b
    // Python: c = poly_mul_mod(a, s, q)
    poly c = poly_mul_mod(pk.a, sk);
    
    // Python: val = (c + d) % q
    poly val = poly_add_mod_q(c, pk.d);
    
    // Python: b = np.floor((p / q) * val) % p
    pk.b = poly_quantize(val);
    
    return {pk, sk};
}