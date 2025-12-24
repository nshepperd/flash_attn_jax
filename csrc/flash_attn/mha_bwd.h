#pragma once

#include <cuda_runtime_api.h>
#include <cutlass/numeric_types.h>
#include <stddef.h>

#include "xla/ffi/api/ffi.h"

namespace ffi = xla::ffi;

ffi::Error mha_bwd_impl(cudaStream_t stream,
                        int32_t device,
                        ffi::AnyBuffer dout, ffi::AnyBuffer q, ffi::AnyBuffer k,
                        ffi::AnyBuffer v, ffi::AnyBuffer o,
                        ffi::Buffer<ffi::F32> lse, ffi::Result<ffi::AnyBuffer> dq,
                        ffi::Result<ffi::AnyBuffer> dk, ffi::Result<ffi::AnyBuffer> dv,
                        ffi::ResultBuffer<ffi::F32> softmax_d,  // batch_size x num_heads x seqlen_q_rounded
                        ffi::ResultBuffer<ffi::F32> dq_accum,   // batch_size x seqlen_q_rounded x num_heads x head_size_rounded
                        double softmax_scale, bool is_causal,
                        int64_t window_size_left, int64_t window_size_right, bool deterministic);

ffi::Error
mha_varlen_bwd_impl(
    cudaStream_t stream,
    int32_t device,
    ffi::AnyBuffer dout,  // total_q x num_heads, x head_size
    ffi::AnyBuffer q,     // total_q x num_heads x head_size, total_q := \sum_{i=0}^{b} s_i
    ffi::AnyBuffer k,     // total_k x num_heads_k x head_size, total_k := \sum_{i=0}^{b} s_i
    ffi::AnyBuffer v,     // total_k x num_heads_k x head_size, total_k := \sum_{i=0}^{b} s_i
    ffi::AnyBuffer o,     // total_q x num_heads x head_size,
    ffi::Buffer<ffi::F32> lse, // b x h x s   softmax logsumexp
    ffi::Buffer<ffi::S32> cu_seqlens_q,  // b+1
    ffi::Buffer<ffi::S32> cu_seqlens_k,  // b+1
    ffi::Result<ffi::AnyBuffer> dq,   // total_q x num_heads x head_size, total_q := \sum_{i=0}^{b} s_i
    ffi::Result<ffi::AnyBuffer> dk,   // total_k x num_heads_k x head_size, total_k := \sum_{i=0}^{b} s_i
    ffi::Result<ffi::AnyBuffer> dv,   // total_k x num_heads_k x head_size, total_k := \sum_{i=0}^{b} s_i
    ffi::ResultBuffer<ffi::F32> softmax_d,  // batch_size x num_heads x seqlen_q_rounded
    ffi::ResultBuffer<ffi::F32> dq_accum,   // (total_q + 128 * batch_size) x num_heads x head_size_rounded
    int64_t max_seqlen_q,
    int64_t max_seqlen_k,          // max sequence length to choose the kernel
    float softmax_scale,
    bool zero_tensors,
    bool is_causal,
    int64_t window_size_left,
    int64_t window_size_right,
    bool deterministic);