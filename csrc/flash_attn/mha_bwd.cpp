#include <driver_types.h>
#include <stddef.h>
#include <cutlass/numeric_types.h>
#include <cuda_runtime_api.h>
#include <pybind11/pybind11.h>
#include <cute/layout.hpp>

#include "flash.h"
#include "static_switch.h"
#include "check.h"

#include "flash_common.h"
#include "mha_bwd.h"
#include "flash_common.h"
#include "xla/ffi/api/ffi.h"

namespace ffi = xla::ffi;

ffi::Error set_params_dgrad(Flash_bwd_params &params,
					  ffi::DataType element_type,
                      // sizes
                      const size_t b,
                      const size_t seqlen_q,
                      const size_t seqlen_k,
                      const size_t seqlen_q_rounded,
                      const size_t seqlen_k_rounded,
                      const size_t h,
                      const size_t h_k,
                      const size_t d,
                      const size_t d_rounded,
                      // device pointers
                      void* q_ptr,
                      void* k_ptr,
                      void* v_ptr,
                      void* out_ptr,
                      void* dout_ptr,
                      void *dq_ptr,
                      void *dk_ptr,
                      void *dv_ptr,
                      void *cu_seqlens_q_d,
                      void *cu_seqlens_k_d,
                      void *dq_accum_d,
                      void *dk_accum_d,
                      void *dv_accum_d,
                      void *softmax_lse_d,
                      void *dsoftmax_sum_d,
                      float p_dropout,
                      float softmax_scale,
                      int window_size_left,
                      int window_size_right,
                      bool deterministic) {

    FFI_RET_CHECK(set_params_fprop(params, element_type,
                     b, seqlen_q, seqlen_k, seqlen_q_rounded, seqlen_k_rounded, h, h_k, d, d_rounded,
                     q_ptr, k_ptr, v_ptr, out_ptr,
                     cu_seqlens_q_d,
                     cu_seqlens_k_d,
                     nullptr,
                     nullptr,
                     softmax_lse_d,
                     p_dropout,
                     softmax_scale,
                     window_size_left,
                     window_size_right));

    // Set the pointers and strides.
    params.do_ptr = dout_ptr;
	params.do_row_stride = params.o_row_stride;
	params.do_head_stride = params.o_head_stride;
    params.dq_ptr = dq_ptr;
    params.dk_ptr = dk_ptr;
    params.dv_ptr = dv_ptr;

    // dk&dv is expanded to the same h as dq for MQA, we sum it later
    auto dq = cute::compact_row_major(cute::make_shape(b, seqlen_q, h, d));
	auto dk = cute::compact_row_major(cute::make_shape(b, seqlen_k, h, d));
	auto dv = cute::compact_row_major(cute::make_shape(b, seqlen_k, h, d));

    params.dq_row_stride = cute::get<1>(dq);
    params.dk_row_stride = cute::get<1>(dk);
    params.dv_row_stride = cute::get<1>(dv);
    params.dq_head_stride = cute::get<2>(dq);
    params.dk_head_stride = cute::get<2>(dk);
    params.dv_head_stride = cute::get<2>(dv);

    if (cu_seqlens_q_d == nullptr) {
        params.do_batch_stride = params.o_batch_stride;
        params.dq_batch_stride = cute::get<0>(dq);
        params.dk_batch_stride = cute::get<0>(dk);
        params.dv_batch_stride = cute::get<0>(dv);
    }

    params.dq_accum_ptr = dq_accum_d;
    params.dk_accum_ptr = dk_accum_d;
    params.dv_accum_ptr = dv_accum_d;

    // Softmax sum
    params.dsoftmax_sum = dsoftmax_sum_d;

    params.deterministic = deterministic;
    return ffi::Error();
}

void run_mha_bwd(Flash_bwd_params &params, cudaStream_t stream) {
    FP16_SWITCH(!params.is_bf16, [&] {
        HEADDIM_SWITCH(params.d, [&] {
            run_mha_bwd_<elem_type, kHeadDim>(params, stream);
        });
    });
}

ffi::Error mha_bwd_impl(cudaStream_t stream, ffi::ScratchAllocator scratch,
                        int32_t device,
                        ffi::AnyBuffer dout, // batch_size x seqlen_q x num_heads x head_size_og
                        ffi::AnyBuffer q,    // batch_size x seqlen_q x num_heads x head_size
                        ffi::AnyBuffer k,    // batch_size x seqlen_k x num_heads_k x head_size
                        ffi::AnyBuffer v,    // batch_size x seqlen_k x num_heads_k x head_size
                        ffi::AnyBuffer o,    // batch_size x seqlen_q x num_heads x head_size
                        ffi::Buffer<ffi::F32> lse, // b x h x seqlen_q
                        ffi::Result<ffi::AnyBuffer> dq,    // batch_size x seqlen_q x num_heads x head_size
                        ffi::Result<ffi::AnyBuffer> dk,    // batch_size x seqlen_k x num_heads_k x head_size
                        ffi::Result<ffi::AnyBuffer> dv,    // batch_size x seqlen_k x num_heads_k x head_size
                        double softmax_scale, bool is_causal,
                        int64_t window_size_left, int64_t window_size_right) {
	int major, minor, sm_count;
    FFI_CUDA_CHECK(cudaDeviceGetAttribute(&major, cudaDevAttrComputeCapabilityMajor, device));
	FFI_CUDA_CHECK(cudaDeviceGetAttribute(&minor, cudaDevAttrComputeCapabilityMinor, device));
	FFI_CUDA_CHECK(cudaDeviceGetAttribute(&sm_count, cudaDevAttrMultiProcessorCount, device));

    if (is_causal) { window_size_right = 0; }
    bool is_sm8x = major == 8 && minor >= 0;
    bool is_sm80 = major == 8 && minor == 0;
    bool is_sm90 = major == 9 && minor == 0;
    FFI_CHECK(is_sm90 || is_sm8x) << "FlashAttention only supports Ampere GPUs or newer.";

    auto q_dtype = q.element_type();
    FFI_CHECK(q_dtype == ffi::BF16 || q_dtype == ffi::F16) << ffi::ErrorCode::kInvalidArgument
        << "FlashAttention only support fp16 and bf16 data type";
    if (q_dtype == ffi::BF16) {
        FFI_CHECK(is_sm90 || is_sm8x) << "bfloat16 is only supported on Ampere GPUs or newer";
    }

    FFI_CHECK(k.element_type() == q_dtype) << "query and key must have the same dtype";
    FFI_CHECK(v.element_type() == q_dtype) << "query and value must have the same dtype";
    FFI_CHECK(o.element_type() == q_dtype) << "query and out must have the same dtype";
    FFI_CHECK(dout.element_type() == q_dtype) << "query and dout must have the same dtype";
    FFI_CHECK(dq->element_type() == q_dtype) << "dq must have the same dtype as q";
    FFI_CHECK(dk->element_type() == q_dtype) << "dk must have the same dtype as q";
    FFI_CHECK(dv->element_type() == q_dtype) << "dv must have the same dtype as q";

    const int batch_size = q.dimensions()[0];
    const int seqlen_q = q.dimensions()[1];
    const int num_heads = q.dimensions()[2];
    const int head_size = q.dimensions()[3];
    const int seqlen_k = k.dimensions()[1];
    const int num_heads_k = k.dimensions()[2];
    FFI_CHECK(batch_size > 0) << "batch size must be positive";
    FFI_CHECK(head_size % 8 == 0) << "head_size should be a multiple of 8";
    FFI_CHECK(head_size <= 256) << "FlashAttention backward only supports head dimension at most 256";
    if (head_size > 192) {
        FFI_CHECK(is_sm80 || is_sm90) << "FlashAttention backward for head dim > 192 requires A100/A800 or H100/H800";
    }
    FFI_CHECK(num_heads % num_heads_k == 0) << "Number of heads in key/value must divide number of heads in query";

    auto round_multiple = [](int x, int m) { return (x + m - 1) / m * m; };
    const int head_size_rounded = round_multiple(head_size, 32);
    const int seqlen_q_rounded = round_multiple(seqlen_q, 128);
    const int seqlen_k_rounded = round_multiple(seqlen_k, 128);

    if (window_size_left >= seqlen_k) { window_size_left = -1; }
    if (window_size_right >= seqlen_k) { window_size_right = -1; }

    // bool loop = seqlen_k > blocksize_c;
    // TODO: change later, for now set to true for simplicity
    bool loop = true;

    void* softmax_d = nullptr;
    FFI_CHECK_OPTIONAL(softmax_d, scratch.Allocate(batch_size * num_heads * seqlen_q_rounded * 4, 4))
        << "Failed to allocate memory for softmax_d";
    void* dq_accum = nullptr;
    void* dk_accum = nullptr;
	void* dv_accum = nullptr;
    bool deterministic = false;
    if (loop) {
        if (!deterministic) {
            FFI_CHECK_OPTIONAL(dq_accum, scratch.Allocate(batch_size * seqlen_q_rounded * num_heads * head_size_rounded * 4, 4))
                << "Failed to allocate memory for dq_accum";
            FFI_CUDA_CHECK(cudaMemset(dq_accum, 0, batch_size * seqlen_q_rounded * num_heads * head_size_rounded * 4));
        } else {
    //         const int nsplits = (sm_count + batch_size * num_heads - 1) / (batch_size * num_heads);
	// 		C10_CUDA_CHECK(cudaMalloc(&dq_accum, nsplits * batch_size * seqlen_q_rounded * num_heads * head_size_rounded * 4));
	// 		// previously allocated with torch.zeros, so i guess we need to zero it
	// 		C10_CUDA_CHECK(cudaMemset(dq_accum, 0, nsplits * batch_size * seqlen_q_rounded * num_heads * head_size_rounded * 4));
        }
    }


    // For MQA, dk and dv are expanded to the same n_heads as dq (handled in xla).
    // After returning the result, it gets reduced to the original size by summing, so we don't need to do anything here.
	void* dk_expanded = dk->untyped_data();
	void* dv_expanded = dv->untyped_data();

    Flash_bwd_params params;

    FFI_RET_CHECK(set_params_dgrad(params,
					 q_dtype,
                     batch_size,
                     seqlen_q, seqlen_k,
                     seqlen_q_rounded, seqlen_k_rounded,
                     num_heads, num_heads_k,
                     head_size, head_size_rounded,
                     q.untyped_data(), k.untyped_data(), v.untyped_data(), o.untyped_data(),
                     dout.untyped_data(), dq->untyped_data(), dk_expanded, dv_expanded,
                     nullptr,
                     nullptr,
                     loop ? dq_accum : nullptr,
                     // loop ? dk_accum.data_ptr() : nullptr,
                     // loop ? dv_accum.data_ptr() : nullptr,
                     nullptr,
                     nullptr,
                     lse.untyped_data(),
                     softmax_d,
                     0.0,
                     softmax_scale,
                     window_size_left,
                     window_size_right,
                     deterministic));
    params.dq_accum_split_stride = !deterministic ? 0 : (batch_size * seqlen_q_rounded * num_heads * head_size_rounded);

    auto launch = &run_mha_bwd;

    FFI_CHECK_OPTIONAL(*(void**)&params.rng_state, scratch.Allocate(2 * 8, 8))
        << "Failed to allocate memory for RNG state";

    if (seqlen_q > 0) {
        launch(params, stream);
        FFI_CUDA_CHECK(cudaStreamSynchronize(stream));
    } else {
        // If seqlen_q == 0, then we have an empty tensor. We need to set the output to 0.
        FFI_CUDA_CHECK(cudaMemset(dq->untyped_data(), 0, dq->size_bytes()));
        FFI_CUDA_CHECK(cudaMemset(dk->untyped_data(), 0, dk->size_bytes()));
        FFI_CUDA_CHECK(cudaMemset(dv->untyped_data(), 0, dv->size_bytes()));
    }

    return ffi::Error();
}

ffi::Error
mha_varlen_bwd_impl(
    cudaStream_t stream,
    ffi::ScratchAllocator scratch,
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
    int64_t max_seqlen_q,
    int64_t max_seqlen_k,          // max sequence length to choose the kernel
    float softmax_scale,
    bool zero_tensors,
    bool is_causal,
    int64_t window_size_left,
    int64_t window_size_right,
    bool deterministic) {

    if (is_causal) { window_size_right = 0; }
	int major, minor, sm_count;
	FFI_CUDA_CHECK(cudaDeviceGetAttribute(&major, cudaDevAttrComputeCapabilityMajor, device));
	FFI_CUDA_CHECK(cudaDeviceGetAttribute(&minor, cudaDevAttrComputeCapabilityMinor, device));
	FFI_CUDA_CHECK(cudaDeviceGetAttribute(&sm_count, cudaDevAttrMultiProcessorCount, device));
    // bool is_sm75 = major == 7 && minor == 5;
    bool is_sm8x = major == 8 && minor >= 0;
    bool is_sm80 = major == 8 && minor == 0;
    bool is_sm90 = major == 9 && minor == 0;
    FFI_CHECK(is_sm90 || is_sm8x) << "FlashAttention only supports Ampere GPUs or newer.";
    // We will support Turing in the near future
    // TORCH_CHECK(is_sm90 || is_sm8x || is_sm75, "FlashAttention only supports Turing GPUs or newer.");
    bool is_dropout = false;

    auto q_dtype = q.element_type();
    FFI_CHECK(q_dtype == ffi::BF16 || q_dtype == ffi::F16) << ffi::ErrorCode::kInvalidArgument
        << "FlashAttention only support fp16 and bf16 data type";
    if (q_dtype == ffi::BF16) {
        FFI_CHECK(is_sm90 || is_sm8x) << "bfloat16 is only supported on Ampere GPUs or newer";
    }
    FFI_CHECK(k.element_type() == q_dtype) << ffi::ErrorCode::kInvalidArgument << "query and key must have the same dtype";
    FFI_CHECK(v.element_type() == q_dtype) << ffi::ErrorCode::kInvalidArgument << "query and value must have the same dtype";
    FFI_CHECK(o.element_type() == q_dtype) << ffi::ErrorCode::kInvalidArgument << "query and out must have the same dtype";
    FFI_CHECK(dout.element_type() == q_dtype) << ffi::ErrorCode::kInvalidArgument << "query and dout must have the same dtype";
    FFI_CHECK(dq->element_type() == q_dtype) << ffi::ErrorCode::kInvalidArgument << "dq must have the same dtype as q";
    FFI_CHECK(dk->element_type() == q_dtype) << ffi::ErrorCode::kInvalidArgument << "dk must have the same dtype as q";
    FFI_CHECK(dv->element_type() == q_dtype) << ffi::ErrorCode::kInvalidArgument << "dv must have the same dtype as q";

    const auto sizes = q.dimensions();

    const int total_q = sizes[0];
    const int batch_size = cu_seqlens_q.element_count() - 1;
    const int num_heads = sizes[1];
    const int head_size_og = dout.dimensions()[2];
    const int head_size = sizes[2];
    const int total_k = k.dimensions()[0];
    const int num_heads_k = k.dimensions()[1];
    FFI_CHECK(batch_size > 0) << "batch size must be positive";
    FFI_CHECK(head_size % 8 == 0) << "head_size should be a multiple of 8";
    FFI_CHECK(head_size <= 256) << "FlashAttention backward only supports head dimension at most 256";
    if (head_size > 192) {
        FFI_CHECK(is_sm80 || is_sm90) << "FlashAttention backward for head dim > 192 requires A100/A800 or H100/H800";
    }
    FFI_CHECK(num_heads % num_heads_k == 0) << "Number of heads in key/value must divide number of heads in query";

    auto round_multiple = [](int x, int m) { return (x + m - 1) / m * m; };
    const int head_size_rounded = round_multiple(head_size, 32);
    const int seqlen_q_rounded = round_multiple(max_seqlen_q, 128);
    const int seqlen_k_rounded = round_multiple(max_seqlen_k, 128);

    FFI_CHECK(head_size == round_multiple(head_size_og, 8)) << "head_size must be head_size_og rounded to a multiple of 8";

    if (window_size_left >= max_seqlen_k) { window_size_left = -1; }
    if (window_size_right >= max_seqlen_k) { window_size_right = -1; }

    // CHECK_SHAPE(q, total_q, num_heads, head_size);
    // CHECK_SHAPE(k, total_k, num_heads_k, head_size);
    // CHECK_SHAPE(v, total_k, num_heads_k, head_size);
    // CHECK_SHAPE(out, total_q, num_heads, head_size);
    // CHECK_SHAPE(dout, total_q, num_heads, head_size_og);
    // CHECK_SHAPE(cu_seqlens_q, batch_size + 1);
    // CHECK_SHAPE(cu_seqlens_k, batch_size + 1);

    // bool loop = max_seqlen_k > blocksize_c;
    // TODO: change later, for now set to true for simplicity
    bool loop = true;

    // Otherwise the kernel will be launched from cuda:0 device
    // Cast to char to avoid compiler warning about narrowing

    FFI_CHECK_ALLOC(softmax_d, scratch.Allocate(batch_size * num_heads * seqlen_q_rounded * 4, 4))
        << "Failed to allocate softmax_d";
    void* dq_accum = nullptr;
    int dq_accum_split_stride = 0;
    if (loop) {
        // We don't want to allocate dq_accum of size (batch, seqlen_q_rounded, num_heads, head_size_rounded)
        // because that would be too large if there is a very long sequence and the rest of the sequences are short.
        // Instead, we allocate dq_accum of size (total_q + 128 * batch, num_heads, head_size_rounded).
        // Note that 128 is the max block size on the seqlen_q dimension.
        // For dQ, the i-th sequence is stored in indices from cu_seqlens[i] + 128 * i to
        // cu_seqlens[i + 1] * 128 * i - 1. This ensures that the i-th sequence and (i + 1)-th sequence will
        // be at least 128 apart. It's ok for us to do atomicAdds up to 128 rows beyond what we're normally
        // allowed to do. So we won't have to do any bound checking, and performance should stay the same.
        if (!deterministic) {
            FFI_CHECK_OPTIONAL(dq_accum, scratch.Allocate((total_q + 128 * batch_size) * num_heads * head_size_rounded * 4, 4))
                << "Failed to allocate memory for dq_accum";
        } else {
            const int nsplits = (sm_count + batch_size * num_heads - 1) / (batch_size * num_heads);
            FFI_CHECK_OPTIONAL(dq_accum, scratch.Allocate(nsplits * (total_q + 128 * batch_size) * num_heads * head_size_rounded * 4, 4))
                << "Failed to allocate memory for dq_accum";
            FFI_CUDA_CHECK(cudaMemsetAsync(dq_accum, 0, nsplits * (total_q + 128 * batch_size) * num_heads * head_size_rounded * 4, stream));
            dq_accum_split_stride = (total_q + 128 * batch_size) * num_heads * head_size_rounded;
        }
    }

    // at::Tensor dk_expanded, dv_expanded;
    // if (num_heads_k != num_heads) {  // MQA / GQA
    //     dk_expanded = torch::empty({total_k, num_heads, head_size}, opts);
    //     dv_expanded = torch::empty({total_k, num_heads, head_size}, opts);
    // } else {
    //     dk_expanded = dk;
    //     dv_expanded = dv;
    // }

    // if( zero_tensors ) {
    //     dq.zero_();
    //     dk_expanded.zero_();
    //     dv_expanded.zero_();
    //     softmax_d.zero_();
    // }

    Flash_bwd_params params;

    FFI_RET_CHECK(set_params_dgrad(params,
                        q_dtype,
                     batch_size,
                     max_seqlen_q, max_seqlen_k,
                     seqlen_q_rounded, seqlen_k_rounded,
                     num_heads, num_heads_k,
                     head_size, head_size_rounded,
                     q.untyped_data(), k.untyped_data(), v.untyped_data(), o.untyped_data(),
                     dout.untyped_data(), dq->untyped_data(), dk->untyped_data(), dv->untyped_data(),
                     cu_seqlens_q.untyped_data(),
                     cu_seqlens_k.untyped_data(),
                     loop ? dq_accum : nullptr,
                     nullptr,
                     nullptr,
                     lse.untyped_data(),
                     softmax_d,
                     0.0,
                     softmax_scale,
                     window_size_left,
                     window_size_right,
                     deterministic));
    params.dq_accum_split_stride = dq_accum_split_stride;

    auto launch = &run_mha_bwd;

    FFI_CHECK_OPTIONAL(*(void**)&params.rng_state, scratch.Allocate(2 * 8, 8))
        << "Failed to allocate memory for RNG state";

    params.alibi_slopes_ptr = nullptr;

    if (max_seqlen_q > 0) {
        launch(params, stream);
        FFI_CUDA_CHECK(cudaStreamSynchronize(stream));
    } else {
        // If seqlen_q == 0, then we have an empty tensor. We need to set the output to 0.
        FFI_CUDA_CHECK(cudaMemsetAsync(dq->untyped_data(), 0, dq->size_bytes(), stream));
        FFI_CUDA_CHECK(cudaMemsetAsync(dk->untyped_data(), 0, dk->size_bytes(), stream));
        FFI_CUDA_CHECK(cudaMemsetAsync(dv->untyped_data(), 0, dv->size_bytes(), stream));
        FFI_CUDA_CHECK(cudaMemsetAsync(softmax_d, 0, batch_size * num_heads * seqlen_q_rounded * 4, stream));
    }
    // // For MQA/GQA we need to sum dK and dV across the groups
    // if (num_heads_k != num_heads) {
    //     at::sum_out(dk, at::reshape(dk_expanded, {total_k, num_heads_k, num_heads / num_heads_k, head_size}), {2});
    //     at::sum_out(dv, at::reshape(dv_expanded, {total_k, num_heads_k, num_heads / num_heads_k, head_size}), {2});
    // }
    // if (head_size_og % 8 != 0) {
    //     dq = dq.index({"...", torch::indexing::Slice(torch::indexing::None, head_size_og)});
    //     dk = dk.index({"...", torch::indexing::Slice(torch::indexing::None, head_size_og)});
    //     dv = dv.index({"...", torch::indexing::Slice(torch::indexing::None, head_size_og)});
    // }

    return ffi::Error(); // Success
    // return { dq, dk, dv, softmax_d };
}
