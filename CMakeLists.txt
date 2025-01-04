cmake_minimum_required(VERSION 3.14)
project(medical_vision VERSION 0.1.0)

set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

# C++ standard
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# OpenCV
set(OpenCV_DIR "D:/Packages/opencv-4.10.0/build/")
find_package(OpenCV REQUIRED)

# Define source files
set(SOURCES
    src/image_preprocessor.cpp
)

# Define header files
set(HEADERS
    include/medical_vision/image_preprocessor.hpp
)
include_directories(${OpenCV_INCLUDE_DIRS})

# Create library
add_library(${PROJECT_NAME} ${SOURCES} ${HEADERS})

# Include directories
target_include_directories(${PROJECT_NAME}
    PUBLIC
        ${CMAKE_CURRENT_SOURCE_DIR}/include
        ${OpenCV_INCLUDE_DIRS}
)

# Link OpenCV
target_link_libraries(${PROJECT_NAME} ${OpenCV_LIBS})

# Add example
add_executable(basic_example examples/basic_example.cpp)
target_link_libraries(basic_example ${PROJECT_NAME} ${OpenCV_LIBS})

# Debug prints
message(STATUS "OpenCV_INCLUDE_DIRS: ${OpenCV_INCLUDE_DIRS}")
message(STATUS "OpenCV_LIBS: ${OpenCV_LIBS}")
message(STATUS "Project source dir: ${CMAKE_CURRENT_SOURCE_DIR}")