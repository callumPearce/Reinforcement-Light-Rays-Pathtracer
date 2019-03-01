cmake_minimum_required(VERSION 3.7)
project(Monte_Carlo_Raytracer_Cuda LANGUAGES CXX CUDA)

# Set compiler
set(CMAKE_CXX_COMPILER "nvcc")

# Compiler Flags
set(CUDA_NVCC_FLAGS "${CUDA_NVCC_FLAGS} -std=c++11 -g -Xptxas -use_fast_math")

# Set output directory for executables
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_HOME_DIRECTORY}/Build)

# Set the output directory for library:
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${CMAKE_HOME_DIRECTORY}/Lib)

# Find SDL2 and include directories to project
find_package(SDL2 REQUIRED)
include_directories(${SDL2_INCLUDE_DIRS})

# Include project directories
include_directories(glm)
include_directories(Source)
include_directories(Source/constants)
include_directories(Source/objects)
include_directories(Source/scenes)
include_directories(Source/utils)
include_directories(Source/lights)
include_directories(Source/sdl)
include_directories(Source/rays)
include_directories(Source/path_tracing)

# Set the sources
set(
    SOURCES
    Source/main.cu
    Source/camera.cu
    Source/sdl/sdl_screen.cpp
    Source/rays/ray.cu
    Source/objects/material.cu
    Source/objects/shape.cu
    Source/objects/triangle.cu
    Source/objects/surface.cu
    Source/scenes/cornell_box_scene.cu
    Source/utils/printing.cpp
    Source/utils/hemisphere_helpers.cu
    Source/lights/area_light.cu
    Source/path_tracing/default_path_tracing.cu
)

add_executable(${PROJECT_NAME} ${SOURCES})
target_link_libraries(${PROJECT_NAME} ${SDL2_LIBRARIES})

# CUDA properties
# target_compile_features(${PROJECT_NAME} PUBLIC cxx_std_11)
set_target_properties(${PROJECT_NAME} PROPERTIES CUDA_SEPARABLE_COMPILATION ON)
set_target_properties(${PROJECT_NAME} PROPERTIES LINKER_LANGUAGE CXX)
