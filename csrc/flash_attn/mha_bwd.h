#pragma once

#include <cuda_runtime_api.h>
#include <cutlass/numeric_types.h>
#include <pybind11/pybind11.h>
#include <stddef.h>

#include "xla/ffi/api/ffi.h"

namespace ffi = xla::ffi;

ffi::Error mha_bwd_impl(cudaStream_t stream, ffi::ScratchAllocator scratch,
                        ffi::AnyBuffer dout, ffi::AnyBuffer q, ffi::AnyBuffer k,
                        ffi::AnyBuffer v, ffi::AnyBuffer o,
                        ffi::Buffer<ffi::F32> lse, ffi::Result<ffi::AnyBuffer> dq,
                        ffi::Result<ffi::AnyBuffer> dk, ffi::Result<ffi::AnyBuffer> dv,
                        double softmax_scale, bool is_causal,
                        int64_t window_size_left, int64_t window_size_right);
