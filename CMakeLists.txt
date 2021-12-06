cmake_minimum_required(VERSION 3.17)
project(pv021_project)

set(CMAKE_CXX_STANDARD 17)

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Ofast -ffp-contract=fast -funsafe-math-optimizations -march=native -msse2 -mfpmath=sse -ftree-vectorize"  )

include_directories(src)

add_executable(pv021_project
        src/main.cpp
        src/ActivationFunction.h
        src/InputManager.h
        src/NeuralNetwork.h
        src/WeightLayer.h)
