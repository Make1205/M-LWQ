# M-LWQ-PKE: High-Performance C++ Implementation with AVX2 Acceleration

[![C++17](https://img.shields.io/badge/C%2B%2B-17-blue.svg)](https://isocpp.org/std/the-standard)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![AVX2](https://img.shields.io/badge/Arch-AVX2-red.svg)](https://en.wikipedia.org/wiki/Advanced_Vector_Extensions)

This repository contains an optimized C++17 implementation of the **M-LWQ (Module-Learning With Quantization)** Public Key Encryption (PKE) scheme.

It serves as a reference implementation and high-precision benchmarking tool, featuring **AVX2 SIMD acceleration** and **Number Theoretic Transform (NTT)** for polynomial arithmetic.

## 1. Core Concept (M-LWQ)

M-LWQ is a novel post-quantum cryptosystem built on the **Learning With Quantization (LWQ)** problem. It replaces the additive Gaussian error sampling (used in Kyber/LWE) with a deterministic **dithered quantization process**, achieving:

1.  **Tight Security:** Security reduction to Module-LWE.
2.  **Extreme Compactness:** Eliminates the need to transmit or store large error terms.

## 2. Implementation Features

This project focuses on **performance** and **correctness verification**.

### 🚀 Optimization Highlights
* **AVX2 Acceleration:**
    * Explicit **AVX2 Intrinsics** for polynomial addition, subtraction, and component-wise operations.
    * Vectorized **Base Multiplication** within the NTT domain.
* **Fast NTT (Number Theoretic Transform):**
    * Replaces the naive $O(N^2)$ multiplication with an efficient $O(N \log N)$ **NTT** implementation compatible with Kyber parameters ($N=256, Q=3329$).
    * Includes **Barrett Reduction** for fast modular arithmetic.
* **Scalar vs. AVX2 Benchmark:**
    * A built-in benchmarking suite that runs the cryptosystem in both **Scalar (Pure C++)** and **AVX2** modes side-by-side to demonstrate speedups.

### 🛠 Algorithms
* **KeyGen / Encrypt / Decrypt:** Complete PKE flow implementation.
* **Quantization:** Efficient implementation of $\mathbb{Z}$ (Scalar) lattice quantization.
* **SHAKE-128:** Self-contained implementation (no external crypto libraries required).

## 3. Build and Run

### Dependencies
* **Compiler:** C++17 compatible (GCC, Clang, or MSVC).
* **Hardware:** CPU with **AVX2** and **FMA** instruction set support (required for the accelerated path).
* **CMake:** Version 3.10 or higher.

### Compilation

The `CMakeLists.txt` is configured to automatically enable `-mavx2`, `-mfma`, and `-O3` optimizations.

```bash
# 1. Clone the repository
git clone [https://github.com/Make1205/M-LWQ.git](https://github.com/Make1205/M-LWQ.git)
cd M-LWQ

# 2. Create build directory
mkdir build
cd build

# 3. Configure and Build
cmake ..
make
Running the Benchmark
Execute the compiled binary mlwq_demo:

Bash

./mlwq_demo

```
## 4. Expected Output


The program will run the full PKE suite in Scalar Mode followed by AVX2 Mode, verifying decryption correctness in every round, and finally producing a speedup report.

(Sample output on an Intel Core i7 CPU)

```
==========================================================
       M-LWQ PKE: 完整流程与组件性能对比
==========================================================
参数集: M-LWQ-512 (NIST L1) (N=256, K=2)

>>> 正在运行: Scalar Mode (1000 轮) <<<
   [PASS] 正确性验证通过

>>> 正在运行: AVX2 Mode (1000 轮) <<<
   [PASS] 正确性验证通过


==============================================================================================
                       核心组件性能对比 (Cycles)
==============================================================================================
Component   Mode        Quantize        Sample          Alg. Efficiency       AVX Improvement     
----------------------------------------------------------------------------------------------
PK / u      Scalar      47776           47559           0.99x                 1.00x (Ref)         
PK / u      AVX2        20751           48263           2.32x                 2.30x               
----------------------------------------------------------------------------------------------
v (Poly)    Scalar      28369           33845           1.19x                 1.00x (Ref)         
v (Poly)    AVX2        19765           41655           2.10x                 1.44x               


==============================================================================================
                       完整流程加速比 (Include SHAKE GenA)
==============================================================================================
Operation           Scalar Cycles     AVX2 Cycles       Speedup
----------------------------------------------------------------------------------------------
KeyGen (Full)       323118            264328            1.22x
Encrypt (Full)      431172            356118            1.21x
Decrypt (Full)      129783            98851             1.31x
==============================================================================================
```
Note: Speedup factors depend on your specific CPU architecture. The NTT implementation reduces complexity from quadratic to log-linear, providing significant gains even without AVX, while AVX2 further accelerates the vectorized operations.



## 5. Project Structure
```
.
├── CMakeLists.txt          # CMake config (Auto-enables AVX2)
├── src/
│   ├── main.cpp            # Dual-mode benchmark runner
│   ├── mlwq.hpp/cpp        # Core M-LWQ PKE algorithms
│   ├── poly.hpp/cpp        # Poly arithmetic (Add/Sub AVX2 intrinsics)
│   ├── ntt.hpp/cpp         # NTT implementation & AVX2 BaseMul
│   ├── params.hpp          # Global params & runtime AVX switch
│   ├── random.hpp/cpp      # Random sampling
│   ├── xof.hpp/cpp         # SHAKE-128 wrapper
│   └── cycles.hpp          # RDTSC cycle counter
└── ...
```
## 6. Academic Citation
If you use this work in your research, please cite the accompanying paper:

代码段
```
@misc{cryptoeprint:2024/714,
      author = {Shanxiang Lyu and Ling Liu and Cong Ling},
      title = {Learning With Quantization: A Ciphertext Efficient Lattice Problem with Tight Security Reduction from {LWE}},
      howpublished = {Cryptology {ePrint} Archive, Paper 2024/714},
      year = {2024},
      url = {[https://eprint.iacr.org/2024/714](https://eprint.iacr.org/2024/714)}
}
```
## 7. License
This project is licensed under the MIT License.