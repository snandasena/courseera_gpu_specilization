cmake_minimum_required(VERSION 3.10)

# Enable CMP0104 to ensure CUDA architectures are set properly
cmake_policy(SET CMP0104 NEW)

project(Streams LANGUAGES CXX CUDA)

# Set CUDA standard version and CUDA architectures
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_ARCHITECTURES 50 60 70 75 80)

# Set the CUDA root directory (adjust if needed)
set(CUDA_ROOT /usr/local/cuda)

# Include directories
include_directories(${CUDA_ROOT}/include)
include_directories(/home/sajith/lib/cuda-samples/Common)

# Specify the source files

# Create the executable
add_executable(streams.exe stream.cu stream.h)
add_executable(mergesort.exe merge_sort.cu merge_sort.h)

# Link CUDA libraries directly
target_link_libraries(streams.exe ${CUDA_ROOT}/lib64/libcudart.so)
target_link_libraries(mergesort.exe ${CUDA_ROOT}/lib64/libcudart.so)

# Suppress deprecated GPU target warnings
add_compile_options(-Wno-deprecated-gpu-targets)

# Define custom clean and run targets
#[[add_custom_target(custom_clean
        COMMAND ${CMAKE_COMMAND} -E remove streams.exe
        COMMENT "Removing streams.exe"
)
add_custom_target(run
        COMMAND ./streams.exe
        DEPENDS streams.exe
        COMMENT "Running streams.exe"
)]]
