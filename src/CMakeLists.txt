# Copyright (c) Huawei Technologies Co., Ltd. 2019. All rights reserved.

cmake_minimum_required(VERSION 3.5.1)

project(sampleYOLOV8)

add_compile_options(-std=c++11)

add_definitions(-DENABLE_DVPP_INTERFACE)
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY  "../out")
set(CMAKE_CXX_FLAGS_DEBUG "-fPIC -O0 -g -Wall")
set(CMAKE_CXX_FLAGS_RELEASE "-fPIC -O2 -Wall")

# 1. 设置 CANN 路径
set(INC_PATH "/usr/local/Ascend/ascend-toolkit/latest")

# 2. 设置库路径
set(LIB_PATH "${INC_PATH}/aarch64-linux/lib64")

# 3. 设置 ACLLite 路径
set(THIRDPART "/usr/local/Ascend/thirdpart/aarch64")

# ==========================================
# 4. 【核心修改】自动查找 OpenCV
# ==========================================
find_package(OpenCV REQUIRED)
message(STATUS "OpenCV found: ${OpenCV_INCLUDE_DIRS}")

# 头文件路径
include_directories(
    ${INC_PATH}/runtime/include/
    ${INC_PATH}/aarch64-linux/include/
    ${OpenCV_INCLUDE_DIRS}       # <--- 这里会自动填入刚才安装的路径
    ${THIRDPART}/include/
    ${THIRDPART}/include/acllite
)

# 库文件路径
link_directories(
    ${THIRDPART}/lib/
    ${LIB_PATH}
    /usr/lib/aarch64-linux-gnu/
)

add_executable(main
        sampleYOLOV8.cpp)

# 链接库
target_link_libraries(main ascendcl acl_dvpp stdc++ acllite opencv_core opencv_imgproc opencv_imgcodecs dl rt)

install(TARGETS main DESTINATION ${CMAKE_RUNTIME_OUTPUT_DIRECTORY})