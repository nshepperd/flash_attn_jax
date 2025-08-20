#pragma once

#include <cstdint>
#include <cuda_runtime_api.h>
#include <cutlass/numeric_types.h>
#include <pybind11/pybind11.h>
#include <stddef.h>

#include "xla/ffi/api/ffi.h"

namespace ffi = xla::ffi;

ffi::Error mha_fwd_impl(
    cudaStream_t stream, 
    ffi::ScratchAllocator scratch,
    int32_t device,
    ffi::AnyBuffer q,
    ffi::AnyBuffer k,
    ffi::AnyBuffer v,
    ffi::Result<ffi::AnyBuffer> o,
    ffi::ResultBuffer<ffi::F32> lse,
    double softmax_scale,
    bool is_causal,
    int64_t window_size_left,
    int64_t window_size_right);

ffi::Error
mha_varlen_fwd_impl(
    cudaStream_t stream,
    ffi::ScratchAllocator scratch,
    int32_t device,
    ffi::AnyBuffer q,  // total_q x num_heads x head_size, total_q := \sum_{i=0}^{b} s_i
    ffi::AnyBuffer k,  // total_k x num_heads_k x head_size, total_k := \sum_{i=0}^{b} s_i
    ffi::AnyBuffer v,  // total_k x num_heads_k x head_size, total_k := \sum_{i=0}^{b} s_i
    ffi::Buffer<ffi::S32> cu_seqlens_q,  // b+1
    ffi::Buffer<ffi::S32> cu_seqlens_k,  // b+1
    ffi::Buffer<ffi::S32> seqused_k, // b. If given, only this many elements of each batch element's keys are used.
    ffi::Result<ffi::AnyBuffer> out, // total_q x num_heads x head_size, total_k := \sum_{i=0}^{b} s_i
    ffi::ResultBuffer<ffi::F32> lse, // total_q x num_heads
    int max_seqlen_q,
    int max_seqlen_k,
    bool has_seqused_k,
    float softmax_scale,
    bool zero_tensors,
    bool is_causal,
    int window_size_left,
    int window_size_right);