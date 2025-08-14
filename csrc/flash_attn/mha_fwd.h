#pragma once

#include <cstdint>
#include <cuda_runtime_api.h>
#include <cutlass/numeric_types.h>
#include <pybind11/pybind11.h>
#include <stddef.h>

#include "xla/ffi/api/ffi.h"

namespace ffi = xla::ffi;

ffi::Error mha_fwd_impl(cudaStream_t stream, ffi::ScratchAllocator scratch,
                   ffi::AnyBuffer q, ffi::AnyBuffer k, ffi::AnyBuffer v,
                   ffi::Result<ffi::AnyBuffer> o,
                   ffi::ResultBuffer<ffi::F32> lse, double softmax_scale,
                   bool is_causal, int64_t window_size_left, int64_t window_size_right);
