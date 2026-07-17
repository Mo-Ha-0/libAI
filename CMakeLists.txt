cmake_minimum_required(VERSION 3.14)
project(libAI VERSION 1.0.0 LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

find_package(OpenMP REQUIRED)

set(SOURCES
    src/layers/dense.cpp
    src/layers/activations.cpp
    src/layers/losses.cpp
    src/layers/regularization.cpp
    src/optimizers/optimizers.cpp
    src/network.cpp
    src/trainer.cpp
    src/tuning.cpp
)

set(HEADERS
    include/libAI/libAI.hpp
    include/libAI/core/tensor.hpp
    include/libAI/core/layer.hpp
    include/libAI/layers/dense.hpp
    include/libAI/layers/activations.hpp
    include/libAI/layers/losses.hpp
    include/libAI/layers/regularization.hpp
    include/libAI/optimizers/optimizers.hpp
    include/libAI/network.hpp
    include/libAI/trainer.hpp
    include/libAI/tuning.hpp
)

add_library(libAI STATIC ${SOURCES} ${HEADERS})

target_include_directories(libAI PUBLIC
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
    $<INSTALL_INTERFACE:include>
)

target_link_libraries(libAI PUBLIC OpenMP::OpenMP_CXX)

target_compile_options(libAI PRIVATE -O2 -march=native -ffast-math)

# Examples
add_executable(simple_example examples/simple_example.cpp)
target_link_libraries(simple_example PRIVATE libAI)

add_executable(iris_example examples/iris_example.cpp)
target_link_libraries(iris_example PRIVATE libAI)

# Install
install(TARGETS libAI DESTINATION lib)
install(DIRECTORY include/libAI DESTINATION include)
