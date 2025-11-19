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
GenMatrix (A)       30035          13286          2.26x          18.3%
Sample (s)          5476           4589           1.19x          3.3%
GenDither           16074          14512          1.11x          9.8%
Arith (A*s)         108576         92251          1.18x          66.1%
Quantize            4203           521            8.07x          2.6%

--------------------------------------------------------------------------------------
 Encrypt Breakdown (Detailed)
--------------------------------------------------------------------------------------
Sub-Component       Scalar (cyc)   AVX2 (cyc)     Speedup        Scalar %
GenMatrix (A)       28644          13659          2.10x          12.3%
Sample (r)          5477           4739           1.16x          2.4%
GenDither           22848          21984          1.04x          9.8%
Arith (u)           107954         93025          1.16x          46.4%
Arith (v)           61608          51038          1.21x          26.5%
Quantize            6230           1011           6.16x          2.7%

--------------------------------------------------------------------------------------
 Decrypt Breakdown (Detailed)
--------------------------------------------------------------------------------------
Sub-Component       Scalar (cyc)   AVX2 (cyc)     Speedup        Scalar %
DeQuantize          9277           3265           2.84x          12.7%
Arith (v-su)        54629          47873          1.14x          74.9%
Decode              8983           150            59.89x          12.3%


>>> PART 2: Core Component Comparison (Quantize vs Sample)
----------------------------------------------------------------------------------------------
Component   Mode        Quantize        Sample          Alg. Efficiency       AVX Improvement     
----------------------------------------------------------------------------------------------
PK / u      Scalar      3401            4109            1.20x                 1.00x (Ref)         
PK / u      AVX2        248             4698            18.88x                13.67x              
----------------------------------------------------------------------------------------------
v (Poly)    Scalar      2142            2071            0.96x                 1.00x (Ref)         
v (Poly)    AVX2        156             2500            16.2x                 13.73x              


>>> PART 3: Full Flow Summary (Total Time)
----------------------------------------------------------------------------------------------
Operation           Scalar Cycles     AVX2 Cycles       Speedup
----------------------------------------------------------------------------------------------
KeyGen              166956            127495            1.31x
Encrypt             234873            187327            1.25x
Decrypt             73299             51650             1.42x
----------------------------------------------------------------------------------------------
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