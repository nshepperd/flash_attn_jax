set(CMAKE_EXPORT_COMPILE_COMMANDS ON)
cmake_minimum_required(VERSION 3.18)

project(flash_attn LANGUAGES CXX CUDA)

set(CMAKE_JOB_POOLS cuda=6)
set(CMAKE_INSTALL_RPATH "$ORIGIN/nvidia/cuda_runtime/lib")
# Make sure RPATH is used instead of RUNPATH
set(CMAKE_INSTALL_RPATH_USE_LINK_PATH FALSE)

# == Find dependencies ==
find_package(Python REQUIRED COMPONENTS Interpreter Development.Module)

execute_process(
    COMMAND ${Python_EXECUTABLE} -m pybind11 --cmakedir
    OUTPUT_VARIABLE pybind11_DIR
    OUTPUT_STRIP_TRAILING_WHITESPACE
)

find_package(pybind11 CONFIG REQUIRED)

# == Setup CUDA ==
string(REGEX REPLACE "--generate-code=arch=compute_[0-9]+,code=\\[?compute_[0-9]+,sm_[0-9]+\\]?" "" 
       CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS}")
string(REGEX REPLACE "-gencode arch=compute_[0-9]+,code=sm_[0-9]+" "" 
       CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS}")

message(WARNING "CMAKE_CUDA_FLAGS: ${CMAKE_CUDA_FLAGS}")

# Set up ccache
find_program(CCACHE_PROGRAM ccache)
if(CCACHE_PROGRAM)
    set(CMAKE_CUDA_COMPILER_LAUNCHER "${CCACHE_PROGRAM}")
    message(STATUS "Using ccache: ${CCACHE_PROGRAM}")
endif()

# Options from environment variables
option(FLASH_ATTENTION_FORCE_BUILD "Force building from source" OFF)
option(FLASH_ATTENTION_SKIP_CUDA_BUILD "Skip CUDA build" OFF)
option(FLASH_ATTENTION_FORCE_CXX11_ABI "Force using C++11 ABI" OFF)

# CUDA handling
# Get CUDA architectures from environment or use default
if(DEFINED ENV{FLASH_ATTN_CUDA_ARCHS})
    set(CMAKE_CUDA_ARCHITECTURES $ENV{FLASH_ATTN_CUDA_ARCHS})
else()
    # set(CMAKE_CUDA_ARCHITECTURES "80;90;100;120")
    set(CMAKE_CUDA_ARCHITECTURES "80") 
endif()

find_package(CUDAToolkit REQUIRED)


# CUDA flags
set(CUDA_FLAGS
    -O3
    -std=c++20
    --use_fast_math
    --expt-relaxed-constexpr
    --expt-extended-lambda
    -U__CUDA_NO_HALF_OPERATORS__
    -U__CUDA_NO_HALF_CONVERSIONS__
    -U__CUDA_NO_HALF2_OPERATORS__
    -U__CUDA_NO_BFLOAT16_CONVERSIONS__
)

# Collect source files
file(GLOB CUDA_SOURCES 
    "csrc/flash_attn/src/flash_fwd_hdim*.cu"
    "csrc/flash_attn/src/flash_bwd_hdim*.cu"
    "csrc/flash_attn/src/flash_fwd_split_hdim*.cu"
)

file(GLOB CC_SOURCES 
    "csrc/flash_attn/*.cpp"
)

# Create CUDA extension
pybind11_add_module(flash_api
    ${CC_SOURCES}
    ${CUDA_SOURCES}
)

set_property(TARGET flash_api PROPERTY JOB_POOL_COMPILE cuda)

target_compile_options(flash_api PRIVATE
    $<$<COMPILE_LANGUAGE:CUDA>:${CUDA_FLAGS}>
)

target_include_directories(flash_api PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/csrc/flash_attn
    ${CMAKE_CURRENT_SOURCE_DIR}/csrc/flash_attn/src
    ${CMAKE_CURRENT_SOURCE_DIR}/csrc/cutlass/include
)

target_link_libraries(flash_api PRIVATE
    CUDA::cudart
)

if(FLASH_ATTENTION_FORCE_CXX11_ABI)
    target_compile_definitions(flash_api PRIVATE
        _GLIBCXX_USE_CXX11_ABI=1
    )
endif()

# Installation
install(TARGETS flash_api
        DESTINATION ${SKBUILD_PLATLIB_DIR}/flash_attn_jax_lib
)

install(DIRECTORY src/flash_attn_jax/
        DESTINATION ${SKBUILD_PLATLIB_DIR}/flash_attn_jax
        FILES_MATCHING PATTERN "*.py"
)