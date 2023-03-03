cmake_minimum_required(VERSION 3.21 FATAL_ERROR)
project(DetectWritedNumber_runner)

set(CMAKE_PREFIX_PATH /Users/aidarazizov/PycharmProjects/teest/venv/lib/python3.8/site-packages/torch/share/cmake)

find_package(OpenCV REQUIRED)
find_package(Torch REQUIRED)

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${TORCH_CXX_FLAGS} ")

include_directories( ${OpenCV_INCLUDE_DIRS} )
add_executable(DetectWritedNumber_runner srcs/main.cpp srcs/FrameAnalyzer.cpp includes/FrameAnalyzer.h srcs/PaintWindow.cpp includes/PaintWindow.h srcs/FramePreprocessor.cpp includes/FramePreprocessor.h)


target_link_libraries( DetectWritedNumber_runner  ${TORCH_LIBRARIES})
target_link_libraries( DetectWritedNumber_runner  ${OpenCV_LIBS})

set_property(TARGET DetectWritedNumber_runner PROPERTY CXX_STANDARD 17)
