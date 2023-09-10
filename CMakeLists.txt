cmake_minimum_required(VERSION 3.8.2) # 3.8.0 is minimum version that allow Linux remote building and debugging
project(Spectralysis)


# Enforce c++20 standard.
set (CMAKE_CXX_STANDARD 20)
set (CMAKE_BUILD_TYPE "Release")

# This project consist of 3 components, each in one directory, so add each of them
add_subdirectory(Engine)
add_subdirectory(DemoApp)
add_subdirectory(PythonWrapper)


