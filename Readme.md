# M-LWQ-PKE: A C++ Implementation of Module-Learning With Quantization

[![C++17](https://img.shields.io/badge/C%2B%2B-17-blue.svg)](https://isocpp.org/std/the-standard)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

This repository contains a C++17 implementation of the **M-LWQ (Module-Learning With Quantization)** Public Key Encryption (PKE) scheme.

It serves as a reference implementation and high-precision benchmarking tool for the accompanying research paper:

<!-- > **"M-LWQ: A Module-Lattice-Based KEM with Hybrid Quantization for Extreme Compactness"** 
>
> **Author**: Ma Ke (Jinan University) -->

## 1. Core Concept (M-LWQ)

M-LWQ is a novel post-quantum cryptosystem built on the **Learning With Quantization (LWQ)** problem.

The central innovation of LWQ is to replace the additive Gaussian error `e` used in traditional LWE-based schemes (like CRYSTALS-Kyber ) with a deterministic, dithered **quantization process**. This approach is designed to achieve the "best of both worlds":

1.  **Tight Security:** The quantization error can be statistically indistinguishable from a Gaussian error, allowing for a tight security reduction to the standard Module-LWE problem.
2.  **Extreme Compactness:** By eliminating the need to sample and store large error terms, the scheme can achieve state-of-the-art compactness, as shown in the paper's comparative analysis against Kyber and Saber.

<!-- The M-LWQ paper proposes a **Hybrid Quantization Strategy**:
* **Public Key:** Uses **Polar Lattices** to generate quantization error that is statistically close to a Gaussian, enabling a tight security proof.
* **Ciphertext:** Uses the **$E_8$ Lattice** for an optimal balance of compression and computational efficiency. -->

## 2. About This Implementation

This C++ implementation focuses on **correctness validation** and **high-precision performance benchmarking** of the M-LWQ PKE scheme.

### Key Features:

* **PKE Algorithms:** Provides a complete implementation of the core PKE algorithms from the paper:
    * `M-LWQ.PKE.KeyGen` 
    * `M-LWQ.PKE.Encrypt` 
    * `M-LWQ.PKE.Decrypt` 
* **$\mathbb{Z}$ (Scalar) Quantization:** This implementation features an efficient, in-place $\mathbb{Z}$ (Scalar) lattice quantizer.
    <!-- * **Note:** Per `params.hpp`, the current configuration (`Q_MODE = QUANT_D8`) applies the $E_8$ quantizer to *all* components (public key `b_q` and ciphertext `u`, `v`). -->
* **High-Precision Benchmarking:** The `main.cpp` entry point is a comprehensive benchmark tool that uses the `RDTSC` (Read Time-Stamp Counter) instruction for highly accurate CPU cycle counting.
* **Parameter Set:** The code is pre-configured for **NIST Security Level 1 (M-LWQ-512)**, using `K=2`, `N=256`, `Q=3329`, and `ETA=2`.
* **Self-Contained:** Includes a minimal C++ implementation of SHAKE-128 (`xof.cpp`, `sha3.cpp`) for seed expansion, requiring no external cryptographic libraries.
* **NTT-Ready:** An implementation of the Number Theoretic Transform (NTT) is included (`ntt.cpp`), though the default polynomial multiplication in `poly.cpp` uses a simple $O(N^2)$ implementation.

## 3. How to Build and Run

This project uses CMake (minimum version 3.10).

### Dependencies
* A C++17 compliant compiler (e.g., `g++`, `clang++`, or MSVC)
* `CMake` (>= 3.10)

### Compilation

```bash
# 1. Clone the repository
git clone [https://github.com/Make1205/M-LWQ.git](https://github.com/Make1205/M-LWQ.git)
cd M-LWQ

# 2. Create and enter a build directory
mkdir build
cd build

# 3. Configure the project with CMake
cmake ..

# 4. Build the executable
make
# Or, using the cross-platform CMake command
# cmake --build .
```

This will create an executable named `mlwq_demo` in the `build/` directory.

### Running the Benchmark

Simply execute the compiled binary:

```bash
./mlwq_demo
```

### Expected Output

The program will first print the current parameter set and theoretical component sizes. It will then run two benchmarks:

1.  A "component benchmark" comparing the CPU cycles for M-LWQ's `Quantize` vs. LWE's `Sample e`.
2.  A "full flow benchmark" measuring the total cycles for `KeyGen`, `Encrypt`, and `Decrypt`, followed by a correctness verification.

```
=== C++ M-LWQ PKE 完整验证与性能基准 ===
参数集: M-LWQ-512 (NIST L1)
参数: K=2, N=256, Q=3329, ETA=2
量化模式: SCALAR (Z)

--- 缩放系数 (M-LWQ-512) ---
  d_pk=9 (P_PK=512, dither mod 6)
  d_u=9 (P_U=512, dither mod 6)
  d_v=5 (P_V=32, dither mod 104)

--- 理论尺寸 (M-LWQ-512) ---
  公钥 (PK) 尺寸: 640 字节 (seedA: 32, seed_d_pk: 32, b_q: 576)
  私钥 (SK) 尺寸: 192 字节
  密文 (CT) 尺寸: 736 字节 (u: 576, v: 160)
  明文 (PT) 尺寸: 32 字节

[MAIN] 正在运行 (量化 vs 采样) 基准测试...

--- (量化 vs 采样) 基准 (平均 100 轮) ---
  组件       | M-LWQ 量化 (Quantize) | LWE 采样 (Sample e) 
  ----------------|-----------------------|----------------------
  PK (b_q)      |        18825 cycles |        35972 cycles
  CT (u)        |        19862 cycles |        37564 cycles
  CT (v)        |         7542 cycles |        16615 cycles
====================================================

[MAIN] 正在运行完整流程基准测试...

[MAIN] 正在验证 100 轮运行的正确性...

=========================
✅ 验证成功! 所有 100 轮解密均正确
=========================

--- 完整流程性能基准 (平均 100 轮) ---
  KeyGen:       7481690 cycles
  Encrypt:     11070301 cycles
  Decrypt:      3336777 cycles
=========================
```

## 4. Project Structure

```
.
├── CMakeLists.txt          # The build script
├── src/
│   ├── main.cpp            # Main entry point and benchmark runner
│   ├── mlwq.hpp/cpp        # Core M-LWQ PKE algorithms
│   ├── poly.hpp/cpp        # Polynomial, vector, and matrix arithmetic
│   │                         (Includes the D8/E8 quantizer)
│   ├── params.hpp          # Cryptographic parameters (N, Q, K, etc.)
│   ├── ntt.hpp/cpp         # Number Theoretic Transform (NTT)
│   ├── xof.hpp/cpp         # C++ wrapper for SHAKE-128 XOF
│   ├── sha3.hpp/cpp        # C implementation of SHAKE-128
│   ├── random.hpp/cpp      # Random polynomial sampling
│   └── cycles.hpp          # RDTSC header for CPU cycle counting
│
└── ... (Other files)
```

## 5. Academic Citation

If you use this work in your research, please cite the accompanying paper:

```bibtex
@misc{cryptoeprint:2024/714,
      author = {Shanxiang Lyu and Ling Liu and Cong Ling},
      title = {Learning With Quantization: A Ciphertext Efficient Lattice Problem with Tight Security Reduction from {LWE}},
      howpublished = {Cryptology {ePrint} Archive, Paper 2024/714},
      year = {2024},
      url = {https://eprint.iacr.org/2024/714}
}
```

## 6. License

This project is licensed under the MIT License.