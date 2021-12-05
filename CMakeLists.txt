cmake_minimum_required(VERSION 3.17)
project(pv021_project)

find_package(OpenMP)
if (OPENMP_FOUND)
    set (CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${OpenMP_C_FLAGS}")
    set (CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
    set (CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} ${OpenMP_EXE_LINKER_FLAGS}")
endif()

set(CMAKE_CXX_STANDARD 14)
# debug:
#set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O2")

# no debug
#set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O0 -ffp-contract=fast -funsafe-math-optimizations -march=native -msse2 -mfpmath=sse -ftree-vectorize"  )


# -Ofast
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Ofast -ffp-contract=fast -funsafe-math-optimizations -march=native -msse2 -mfpmath=sse -ftree-vectorize"  )
#set(CMAKE_CXX_FLAGS_DEBUG "-g")



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
