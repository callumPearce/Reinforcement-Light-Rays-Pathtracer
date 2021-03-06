cmake_minimum_required(VERSION 3.7)
project(GPU_Rendering_Engine LANGUAGES CXX CUDA)

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
    ../NN_Builders
    Source
    Source/constants
    Source/objects
    Source/scenes
    Source/utils
    Source/lights
    Source/sdl
    Source/rays
    Source/path_tracing
    Source/radiance_volumes
    Source/deep_learning
)

# Set the sources
set(
    SOURCES
    ../NN_Builders/dq_network.cu
    ../NN_Builders/fc_layer.cu
    Source/main.cu
    Source/camera.cu
    Source/sdl/sdl_screen.cpp
    Source/rays/ray.cu
    Source/objects/material.cu
    Source/objects/shape.cu
    Source/objects/triangle.cu
    Source/objects/surface.cu
    Source/objects/object_importer.cu
    Source/radiance_volumes/radiance_volume.cu
    Source/radiance_volumes/radiance_volume_comparator.cu
    Source/radiance_volumes/radiance_tree.cu
    Source/radiance_volumes/radiance_map.cu
    Source/scenes/cornell_box_scene.cu
    Source/scenes/scene.cu
    Source/utils/printing.cpp
    Source/utils/hemisphere_helpers.cu
    Source/utils/stack.cu
    Source/utils/cuda_helpers.cu
    Source/lights/area_light.cu
    Source/path_tracing/default_path_tracing.cu
    Source/path_tracing/reinforcement_path_tracing.cu
    Source/path_tracing/voronoi_trace.cu
    Source/deep_learning/neural_q_pathtracer.cu
    Source/deep_learning/pre_trained_pathtracer.cu
    Source/deep_learning/nn_rendering_helpers.cu
    Source/deep_learning/q_value_extractor.cu
)


add_executable(${PROJECT_NAME} ${SOURCES})

# Find SDL2 and include directories to project and link to the executable
find_package(SDL2 REQUIRED)
include_directories(${SDL2_INCLUDE_DIRS})
message( ${SDL2_INCLUDE_DIRS} )
message( ${SDL2_LIBRARIES} )
include_directories(-I/usr/local/include/dynet)
target_link_libraries(${PROJECT_NAME} ${SDL2_LIBRARIES} -L/usr/local/lib/ -ldynet)


# Find Tensorflow and link to the executable
# include_directories(/usr/local/lib)
# include_directories(/usr/local/include)
# include_directories(/usr/local/include/eigen3/Eigen)
# target_link_libraries(${PROJECT_NAME} libtensorflow_all.so libprotobuf.so)

# CUDA properties
# target_compile_features(${PROJECT_NAME} PUBLIC cxx_std_11)
set_target_properties(${PROJECT_NAME} PROPERTIES CUDA_SEPARABLE_COMPILATION ON)
set_target_properties(${PROJECT_NAME} PROPERTIES LINKER_LANGUAGE CXX)
