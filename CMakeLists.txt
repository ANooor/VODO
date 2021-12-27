cmake_minimum_required(VERSION 2.8)
project(VODO)
find_package(OpenCV 4.2 REQUIRED)

include_directories( ${OpenCV_INCLUDE_DIRS} )

file(GLOB vodo
    "src/*.h"
    "src/*.cpp"
)
add_executable( vodo ${vodo} )
target_link_libraries( vodo ${OpenCV_LIBS} )
