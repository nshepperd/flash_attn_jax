#pragma once

#include <cuda_runtime_api.h>
#include <cutlass/numeric_types.h>
#include <stddef.h>

#include "check.h"
#include "flash.h"
#include "xla/ffi/api/ffi.h"

namespace ffi = xla::ffi;

ffi::Error set_params_fprop(Flash_fwd_params &params, ffi::DataType element_type,
                            // sizes
                            const size_t b, const size_t seqlen_q, const size_t seqlen_k,
                            const size_t seqlen_q_rounded, const size_t seqlen_k_rounded,
                            const size_t h, const size_t h_k, const size_t d,
                            const size_t d_rounded,
                            // device pointers
                            void *q_ptr, void *k_ptr, void *v_ptr, void *out_ptr,
                            void *cu_seqlens_q_d, void *cu_seqlens_k_d, void *seqused_k, void *p_d,
                            void *softmax_lse_d, float p_dropout, float softmax_scale,
                            int window_size_left, int window_size_right,
                            bool seqlenq_ngroups_swapped = false);

ffi::Error set_params_splitkv(Flash_fwd_params &params,
                              const int batch_size, const int num_heads, const int head_size,
                              const int max_seqlen_k, const int max_seqlen_q,
                              const int head_size_rounded, const float p_dropout,
                              const int num_splits, int multiProcessorCount, ffi::DataType dtype,
                              const int max_splits,
                              void *oaccum_ptr = nullptr, void *lseaccum_ptr = nullptr);
