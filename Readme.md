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
       M-LWQ PKE: 全面性能对比 (Quantize vs Sample)
==========================================================
参数集: M-LWQ-512 (NIST L1) (N=256, K=2)

>>> 正在运行: Scalar Mode (100 轮) <<<
   [PASS] 正确性验证通过

>>> 正在运行: AVX2 Mode (100 轮) <<<
   [PASS] 正确性验证通过


==============================================================================================
                       核心组件性能对比 (Cycles) | Lower is Better
==============================================================================================
Component   Mode        Quantize(M-LWQ) Sample(LWE)     Alg. Efficiency       AVX Improvement     
                        (Cycles)        (Cycles)        (Sample / Quant)      (Scalar / AVX)      
----------------------------------------------------------------------------------------------
PK / u      Scalar      40282           53725           1.33x                 1.00x (Ref)         
PK / u      AVX2        37070           46865           1.26x                 1.09x               
----------------------------------------------------------------------------------------------
v (Poly)    Scalar      27461           32194           1.17x                 1.00x (Ref)         
v (Poly)    AVX2        27450           39171           1.42x                 1.00x               
==============================================================================================
说明:
1. Alg. Efficiency (Sample / Quant): 表示在当前模式下，Quantize 比 Sample 快多少倍。
2. AVX Improvement (Scalar / AVX):   表示 AVX2 版本的 Quantize 比 Scalar 版本快多少倍。


==============================================================================================
                       完整流程加速比 (Scalar vs AVX2)
==============================================================================================
Operation           Scalar Cycles     AVX2 Cycles       Speedup (Scalar / AVX2)
----------------------------------------------------------------------------------------------
KeyGen              322764            318362            1.01x
Encrypt             638169            425448            1.50x
Decrypt             164618            119171            1.38x
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