cmake_minimum_required(VERSION 3.7)
project(Monte_Carlo_Raytracer)

# Set compiler
set(CMAKE_CXX_COMPILER "icc")

# Compiler Flags
set(CMAKE_CXX_FLAGS "-std=c++11 -pipe -Wall -Wno-switch -qopenmp -O3 -g -no-prec-div -lSDL2main -lSDL2 -static-intel") #-flto=full -lm -ftree-vectorize
# set(CMAKE_CXX_FLAGS "-std=c++11 -O3 -g -lSDL2main -lSDL2") #-flto=full -lm -ftree-vectorize

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
include_directories(Source/radiance_volumes)
include_directories(Source/path_tracing)

# Set the sources
set(
    SOURCES
    Source/main.cpp
    Source/camera.cpp
    Source/sdl/sdl_screen.cpp
    Source/rays/ray.cpp
    Source/objects/material.cpp
    Source/objects/shape.cpp
    Source/objects/triangle.cpp
    Source/objects/surface.cpp
    Source/objects/object_importer.cpp
    Source/scenes/cornell_box_scene.cpp
    Source/scenes/monte_carlo_test_scene.cpp
    Source/utils/printing.cpp
    Source/utils/hemisphere_helpers.cpp
    Source/utils/interpolation.cpp
    Source/lights/area_light.cpp
    Source/lights/area_light_plane.cpp
    Source/path_tracing/default_path_tracing.cpp
    Source/path_tracing/importance_sampling_path_tracing.cpp
    Source/path_tracing/precompute_irradiance_path_tracing.cpp
    Source/path_tracing/reinforcement_path_tracing.cpp
    Source/path_tracing/voronoi_trace.cpp
    Source/radiance_volumes/radiance_volume.cpp
    Source/radiance_volumes/radiance_map.cpp
    Source/radiance_volumes/radiance_volume_comparator.cpp
    Source/radiance_volumes/radiance_tree.cpp
)

add_executable(${PROJECT_NAME} ${SOURCES})
target_link_libraries(${PROJECT_NAME} ${SDL2_LIBRARIES})
