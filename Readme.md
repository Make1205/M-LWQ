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
=== M-LWQ Comprehensive Performance Report ===
N=256, K=2

>>> Running: Scalar Mode (1000 rounds)...
   [PASS] Correctness verified.
>>> Running: AVX2 Mode (1000 rounds)...
   [PASS] Correctness verified.

>>> PART 1: Internal Breakdown (Where is time spent?)

--------------------------------------------------------------------------------------
 KeyGen Breakdown (Detailed)
--------------------------------------------------------------------------------------
Sub-Component       Scalar (cyc)   AVX2 (cyc)     Speedup        Scalar %
GenMatrix (A)       27754          13898          2.00x          18.2%
Sample (s)          5027           4647           1.08x          3.3%
GenDither           14480          13758          1.05x          9.5%
Arith (A*s)         101448         89632          1.13x          66.3%
Quantize            4191           469            8.94x          2.7%

--------------------------------------------------------------------------------------
 Encrypt Breakdown (Detailed)
--------------------------------------------------------------------------------------
Sub-Component       Scalar (cyc)   AVX2 (cyc)     Speedup        Scalar %
GenMatrix (A)       25904          13550          1.91x          11.7%
Sample (r)          5425           4741           1.14x          2.4%
GenDither           22529          21061          1.07x          10.1%
Arith (u)           103194         91999          1.12x          46.4%
Arith (v)           58289          48334          1.21x          26.2%
Quantize            6972           940            7.42x          3.1%

--------------------------------------------------------------------------------------
 Decrypt Breakdown (Detailed)
--------------------------------------------------------------------------------------
Sub-Component       Scalar (cyc)   AVX2 (cyc)     Speedup        Scalar %
DeQuantize          8352           3390           2.46x          12.5%
Arith (v-su)        50017          45541          1.10x          74.6%
Decode              8710           139            62.66x          13.0%


>>> PART 2: Core Component Comparison (Quantize vs Sample)
----------------------------------------------------------------------------------------------
Component   Mode        Quantize        Sample          Alg. Efficiency       AVX Improvement     
----------------------------------------------------------------------------------------------
PK / u      Scalar      3691            4201            1.13x                 1.00x (Ref)         
PK / u      AVX2        246             4265            17.32x                15.00x              
----------------------------------------------------------------------------------------------
v (Poly)    Scalar      1618            2661            1.64x                 1.00x (Ref)         
v (Poly)    AVX2        158             1900            11.97x                10.20x              


>>> PART 3: Full Flow Summary (Total Time)
----------------------------------------------------------------------------------------------
Operation           Scalar Cycles     AVX2 Cycles       Speedup
----------------------------------------------------------------------------------------------
KeyGen              154790            124479            1.24x
Encrypt             223969            182193            1.23x
Decrypt             67719             49405             1.37x
----------------------------------------------------------------------------------------------

[FINAL] All checks passed! Implementation is correct.
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
<!-- ## 6. Academic Citation
If you use this work in your research, please cite the accompanying paper:
```
@misc{cryptoeprint:2024/714,
      author = {Shanxiang Lyu and Ling Liu and Cong Ling},
      title = {Learning With Quantization: A Ciphertext Efficient Lattice Problem with Tight Security Reduction from {LWE}},
      howpublished = {Cryptology {ePrint} Archive, Paper 2024/714},
      year = {2024},
      url = {[https://eprint.iacr.org/2024/714](https://eprint.iacr.org/2024/714)}
}
``` -->
## 7. License
This project is licensed under the MIT License.