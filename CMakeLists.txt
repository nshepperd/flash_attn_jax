cmake_minimum_required(VERSION 3.18)

# Get CUDA architectures from environment or use default
if(DEFINED ENV{FLASH_ATTN_CUDA_ARCHS})
    set(CMAKE_CUDA_ARCHITECTURES $ENV{FLASH_ATTN_CUDA_ARCHS})
else()
    # set(CMAKE_CUDA_ARCHITECTURES "80;90;100;120")
    set(CMAKE_CUDA_ARCHITECTURES "80")
endif()

project(flash_attn LANGUAGES CXX CUDA)

set(CMAKE_JOB_POOLS cuda=6)
set(CMAKE_INSTALL_RPATH "$ORIGIN/nvidia/cuda_runtime/lib")
# Make sure RPATH is used instead of RUNPATH
set(CMAKE_INSTALL_RPATH_USE_LINK_PATH FALSE)

if(EXISTS ${Python_ROOT_DIR}/bin/python)
    execute_process(
        COMMAND ${Python_ROOT_DIR}/bin/python -c "import sysconfig; print(sysconfig.get_path('purelib'))"
        OUTPUT_VARIABLE PYTHON_SITE_PACKAGES
        OUTPUT_STRIP_TRAILING_WHITESPACE
    )
    message(STATUS "Python site-packages: ${PYTHON_SITE_PACKAGES}")
    set(Python_EXECUTABLE ${Python_ROOT_DIR}/bin/python)
    set(CMAKE_PREFIX_PATH 
        ${PYTHON_SITE_PACKAGES}
        ${Python_ROOT_DIR})
endif()

# == Find dependencies ==
find_package(Python REQUIRED COMPONENTS Interpreter Development.Module)

message(STATUS "Python executable: ${Python_EXECUTABLE}")

execute_process(
    COMMAND ${Python_EXECUTABLE} -m nanobind --cmake_dir
    OUTPUT_VARIABLE nanobind_DIR
    OUTPUT_STRIP_TRAILING_WHITESPACE
)

find_package(nanobind CONFIG REQUIRED)

# == Setup CUDA ==
string(REGEX REPLACE "--generate-code=arch=compute_[0-9]+,code=\\[?compute_[0-9]+,sm_[0-9]+\\]?" "" 
       CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS}")
string(REGEX REPLACE "-gencode arch=compute_[0-9]+,code=sm_[0-9]+" "" 
       CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS}")

message(STATUS "CMAKE_CUDA_FLAGS: ${CMAKE_CUDA_FLAGS}")

# Set up ccache
find_program(CCACHE_PROGRAM ccache)
if(CCACHE_PROGRAM)
    set(CMAKE_CUDA_COMPILER_LAUNCHER "${CCACHE_PROGRAM}")
    message(STATUS "Using ccache: ${CCACHE_PROGRAM}")
endif()

# CUDA handling
message(STATUS "CMAKE_CUDA_ARCHITECTURES: ${CMAKE_CUDA_ARCHITECTURES}")

find_package(CUDAToolkit REQUIRED)

if (${NO_OPTIONS_FILE})
    set(CMAKE_CUDA_USE_RESPONSE_FILE_FOR_INCLUDES "0")
    set(CMAKE_CXX_USE_RESPONSE_FILE_FOR_INCLUDES "0")
endif()

# CUDA flags
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
set(CUDA_FLAGS
    -O3
    --use_fast_math
    --expt-relaxed-constexpr
    --expt-extended-lambda
    -U__CUDA_NO_HALF_OPERATORS__
    -U__CUDA_NO_HALF_CONVERSIONS__
    -U__CUDA_NO_HALF2_OPERATORS__
    -U__CUDA_NO_BFLOAT16_CONVERSIONS__
    -DFLASHATTENTION_DISABLE_DROPOUT=1
    -DFLASHATTENTION_DISABLE_ALIBI=1
)
set(CXX_FLAGS
    -DFLASHATTENTION_DISABLE_DROPOUT=1
    -DFLASHATTENTION_DISABLE_ALIBI=1
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
nanobind_add_module(flash_api
    ${CC_SOURCES}
    ${CUDA_SOURCES}
)

set_property(TARGET flash_api PROPERTY JOB_POOL_COMPILE cuda)

target_compile_options(flash_api PRIVATE
    $<$<COMPILE_LANGUAGE:CUDA>:${CUDA_FLAGS}>
    $<$<COMPILE_LANGUAGE:CXX>:${CXX_FLAGS}>
)

target_include_directories(flash_api PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/csrc/flash_attn
    ${CMAKE_CURRENT_SOURCE_DIR}/csrc/flash_attn/src
    ${CMAKE_CURRENT_SOURCE_DIR}/csrc/cutlass/include
    ${CMAKE_CURRENT_SOURCE_DIR}/csrc

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
