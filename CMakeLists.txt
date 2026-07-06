cmake_minimum_required(VERSION 3.16)
project(infer CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release)
endif()

find_package(OpenMP REQUIRED)
add_compile_options(-O3 -march=native)

add_library(engine STATIC src/engine.cpp)
target_include_directories(engine PUBLIC src)
target_link_libraries(engine PUBLIC OpenMP::OpenMP_CXX)

add_executable(main  src/main.cpp)
add_executable(bench bench/harness.cpp)
target_link_libraries(main  PRIVATE engine)
target_link_libraries(bench PRIVATE engine)
