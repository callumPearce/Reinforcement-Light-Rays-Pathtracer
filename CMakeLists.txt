cmake_minimum_required(VERSION 3.7)
project(Monte_Carlo_Raytracer)

# Compiler Flags
set(CMAKE_CXX_FLAGS "-pipe -Wall -g -Wno-switch -fopenmp -g3 -O3 -lm -ffast-math -funroll-loops -ftree-vectorize -flto=full")

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

# Set the sources
set(
    SOURCES
    Source/main.cpp
    Source/camera.cpp
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
    Source/lights/area_light.cpp
    Source/lights/area_light_plane.cpp
    Source/rays/path_tracing.cpp
    Source/radiance_volume.cpp
)

add_executable(${PROJECT_NAME} ${SOURCES})
target_link_libraries(${PROJECT_NAME} ${SDL2_LIBRARIES})
