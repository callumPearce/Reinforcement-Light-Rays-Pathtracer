cmake_minimum_required(VERSION 3.7)
project(Monte_Carlo_Raytracer)

# Compiler Flags
set(CMAKE_CXX_FLAGS "-Wall -Wextra -O3 -g")

# Set output directory for executables
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_HOME_DIRECTORY}/build)

# Set the output directory for library:
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${CMAKE_HOME_DIRECTORY}/lib)

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

# Set the sources
set(
    SOURCES
    Source/main.cpp
    Source/camera.cpp
    Source/ray.cpp
    Source/objects/material.cpp
    Source/objects/shape.cpp
    Source/objects/triangle.cpp
    Source/scenes/cornell_box_scene.cpp
    Source/utils/printing.cpp
)

add_executable(${PROJECT_NAME} ${SOURCES})
target_link_libraries(${PROJECT_NAME} ${SDL2_LIBRARIES})