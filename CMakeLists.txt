cmake_minimum_required(VERSION 3.17)
project(pv021_project)

set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fopenmp")

set(CMAKE_CXX_FLAGS "-w -Ofast -ffp-contract=fast -funsafe-math-optimizations -march=native -msse2 -mfpmath=sse -ftree-vectorize -std=c++17")


include_directories(src)

add_executable(pv021_project
        src/main.cpp
        src/ActivationFunction.h
        #src/catch.hpp
        src/InputManager.h
        src/NeuralNetwork.h
        #src/test.cpp
        src/WeightLayer.h)

add_executable(tests
        src/test.cpp
        src/ActivationFunction.h
        src/catch.hpp
        src/InputManager.h
        src/NeuralNetwork.h
        src/WeightLayer.h)
