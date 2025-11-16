cmake_minimum_required(VERSION 3.10)
project(M_LWQ_PKE CXX)

# 设置 C++17 标准
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED True)

# 添加可执行文件
add_executable(mlwq_demo
    src/main.cpp
    src/poly.cpp
    src/random.cpp
    src/mlwq.cpp
    src/xof.cpp
    src/sha3.cpp
    src/ntt.cpp   # <-- 新增 NTT 实现文件
)

# 包含 src 目录作为头文件路径
target_include_directories(mlwq_demo PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/src)

# 在 Linux/macOS 上链接数学库 (libm) 以使用 std::floor/round
if (NOT WIN32)
    target_link_libraries(mlwq_demo PUBLIC m)
endif()

# (可选) 添加安装规则
install(TARGETS mlwq_demo DESTINATION bin)

# 打印消息
message(STATUS "Project configured. Build with 'cmake --build .'")