cmake_minimum_required(VERSION 3.14)
project(medical_vision VERSION 0.1.0)

# C++ standard
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Qt configuration
set(CMAKE_AUTOMOC ON)
set(CMAKE_AUTORCC ON)
set(CMAKE_AUTOUIC ON)

# Find packages
set(OpenCV_DIR "D:/Packages/opencv-4.10.0/build")
set(Qt6_DIR  "D:/Programs/Qt/6.7.3/msvc2022_64/lib/cmake/Qt6")

find_package(OpenCV REQUIRED)
find_package(Qt6 COMPONENTS Core Gui GuiTools Widgets WidgetsTools REQUIRED )

# Define source files
file(GLOB_RECURSE LIB_SOURCES
    "src/*.cpp"
)

file(GLOB_RECURSE GUI_SOURCES
    "gui/*.cpp"
)

file(GLOB_RECURSE GUI_HEADERS
    "gui/*.hpp"
)

# Library
add_library(${PROJECT_NAME} ${LIB_SOURCES})

target_include_directories(${PROJECT_NAME}
    PUBLIC
        ${CMAKE_CURRENT_SOURCE_DIR}
        ${OpenCV_INCLUDE_DIRS} 
)

target_link_libraries(${PROJECT_NAME}
    PUBLIC 
        ${OpenCV_LIBS}
)

# GUI Application
add_executable(medical_vision_gui 
    ${GUI_SOURCES}
    ${GUI_HEADERS}
)

target_link_libraries(medical_vision_gui
    PRIVATE
        ${PROJECT_NAME}
        Qt6::Core
        Qt6::Widgets
        ${OpenCV_LIBS}
)

# Examples (optional)
add_executable(basic_example examples/basic_example.cpp)
target_link_libraries(basic_example 
    PRIVATE 
        ${PROJECT_NAME}
)