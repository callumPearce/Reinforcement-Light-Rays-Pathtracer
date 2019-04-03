cmake_minimum_required(VERSION 3.7)
project(NN_Learn_Q_Values LANGUAGES CXX CUDA)

# Set compiler
set(CMAKE_CXX_COMPILER "nvcc")

# Set cuda
set(CUDACXX "/usr/local/cuda/bin/nvcc")

# Compiler Flags
set(CUDA_NVCC_FLAGS "${CUDA_NVCC_FLAGS} -jN -std=c++11 -g -Xptxas –ptxas-options=-v -use_fast_math")

# Set output directory for executables
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_HOME_DIRECTORY}/Build)

# Set the output directory for library:
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${CMAKE_HOME_DIRECTORY}/Lib)

# Include project directories
include_directories(glm)
include_directories(
    Source
    Source/deep_learning
    ../NN_Builders
)

# Set the sources
set(
    SOURCES
    Source/main.cu
    ../NN_Builders/fc_layer.cu
    ../NN_Builders/dq_network.cu
)


add_executable(${PROJECT_NAME} ${SOURCES})

# Find SDL2 and include directories to project and link to the executable
include_directories(-I/usr/local/include/dynet)
target_link_libraries(${PROJECT_NAME} -L/usr/local/lib/ -ldynet)

# CUDA properties
# target_compile_features(${PROJECT_NAME} PUBLIC cxx_std_11)
set_target_properties(${PROJECT_NAME} PROPERTIES CUDA_SEPARABLE_COMPILATION ON)
set_target_properties(${PROJECT_NAME} PROPERTIES LINKER_LANGUAGE CXX)
