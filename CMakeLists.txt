cmake_minimum_required(VERSION 3.10)
project(M_LWQ_PKE CXX)

# 设置 C++17 标准
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED True)

# --- 关键修改：强制开启 AVX2, FMA 和 O3 优化 ---
if(MSVC)
    add_compile_options(/arch:AVX2 /O2)
else()
    # Linux/macOS (GCC/Clang) 必须添加 -mavx2
    # -march=native 会自动检测你本机 CPU 支持的所有指令集
    add_compile_options(-mavx2 -mfma -O3 -march=native)
endif()

# 添加可执行文件
add_executable(mlwq_demo
    src/main.cpp
    src/poly.cpp
    src/random.cpp
    src/mlwq.cpp
    src/xof.cpp
    src/sha3.cpp
    src/ntt.cpp
)

# 包含 src 目录作为头文件路径
target_include_directories(mlwq_demo PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/src)

# 在 Linux/macOS 上链接数学库 (libm)
if (NOT WIN32)
    target_link_libraries(mlwq_demo PUBLIC m)
endif()

message(STATUS "Project configured with AVX2 support.")