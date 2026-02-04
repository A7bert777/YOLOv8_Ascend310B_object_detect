# Copyright (c) Huawei Technologies Co., Ltd. 2019. All rights reserved.
cmake_minimum_required(VERSION 3.5.1)
project(sampleYOLOV8)

add_compile_options(-std=c++11)
add_definitions(-DENABLE_DVPP_INTERFACE -D_GLIBCXX_USE_CXX11_ABI=0)
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY "${CMAKE_SOURCE_DIR}/build")
set(CMAKE_CXX_FLAGS_DEBUG "-fPIC -O0 -g -Wall")
set(CMAKE_CXX_FLAGS_RELEASE "-fPIC -O2 -Wall")

# 1. 设置 CANN 路径
set(INC_PATH "/usr/local/Ascend/ascend-toolkit/latest")
# 2. 设置库路径
set(LIB_PATH "${INC_PATH}/aarch64-linux/lib64")
# 3. 设置 ACLLite 路径 (刚才 make install 的位置)
set(THIRDPART "/usr/local/Ascend/thirdpart")

# ==========================================
# 4. 【核心修复】手动指定 OpenCV 路径（避开损坏的 find_package）
# ==========================================
set(OPENCV_INC "/usr/local/Ascend/mxVision-5.0.RC3/opensource/include/opencv4")
set(OPENCV_LIB_DIR "/usr/local/Ascend/mxVision-5.0.RC3/opensource/lib")

# 头文件路径
include_directories(
    ${INC_PATH}/runtime/include/
    ${INC_PATH}/aarch64-linux/include/
    ${OPENCV_INC}
    ${THIRDPART}/include/
    ${THIRDPART}/include/acllite
    ${CMAKE_SOURCE_DIR}/include
)

# 库文件路径
link_directories(
    ${THIRDPART}/lib/
    ${LIB_PATH}
    ${OPENCV_LIB_DIR}
    /usr/lib/aarch64-linux-gnu/
)

add_executable(ascend_310b_yolov8_demo src/sampleYOLOV8.cpp)

# 链接库
target_link_libraries(ascend_310b_yolov8_demo 
    ascendcl 
    acl_dvpp 
    stdc++ 
    acllite 
    opencv_world
    dl 
    rt
)

install(TARGETS ascend_310b_yolov8_demo DESTINATION ${CMAKE_RUNTIME_OUTPUT_DIRECTORY})