cmake_minimum_required(VERSION 3.21 FATAL_ERROR)
project(OpenCV_tests)

set(CMAKE_PREFIX_PATH /Users/aidarazizov/PycharmProjects/teest/venv/lib/python3.8/site-packages/torch/share/cmake)

find_package(OpenCV REQUIRED)
find_package(Torch REQUIRED)

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${TORCH_CXX_FLAGS} ")

include_directories( ${OpenCV_INCLUDE_DIRS} )
add_executable(OpenCV_tests main.cpp FrameAnalyzer.cpp FrameAnalyzer.h PaintWindow.cpp PaintWindow.h FramePreprocessor.cpp FramePreprocessor.h)


target_link_libraries( OpenCV_tests  ${TORCH_LIBRARIES})
target_link_libraries( OpenCV_tests  ${OpenCV_LIBS})

set_property(TARGET OpenCV_tests PROPERTY CXX_STANDARD 20)
