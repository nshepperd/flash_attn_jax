/******************************************************************************
 * Copyright (c) 2024, Tri Dao.
 ******************************************************************************/

#include <stddef.h>
#include <cutlass/numeric_types.h>
#include <cuda_runtime_api.h>
#include <pybind11/pybind11.h>

#include "flash.h"
#include "static_switch.h"
#include "check.h"
#include "flash_common.h"
#include "mha_fwd.h"
#include "xla/ffi/api/api.h"
#include "xla/ffi/api/ffi.h"

namespace ffi = xla::ffi;

void run_mha_fwd(Flash_fwd_params &params, cudaStream_t stream, bool force_split_kernel=false) {
    FP16_SWITCH(!params.is_bf16, [&] {
        HEADDIM_SWITCH(params.d, [&] {
            if (params.num_splits <= 1 && !force_split_kernel) {  // If we don't set it num_splits == 0
                run_mha_fwd_<elem_type, kHeadDim>(params, stream);
            } else {
                run_mha_fwd_splitkv_dispatch<elem_type, kHeadDim>(params, stream);
            }
        });
    });
}


ffi::Error mha_fwd_impl(cudaStream_t stream, 
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
    int64_t window_size_right) {
    int major, minor;
	FFI_CUDA_CHECK(cudaDeviceGetAttribute(&major, cudaDevAttrComputeCapabilityMajor, device));
	FFI_CUDA_CHECK(cudaDeviceGetAttribute(&minor, cudaDevAttrComputeCapabilityMinor, device));

    // bool is_sm75 = dprops->major == 7 && dprops->minor == 5;
    bool is_sm8x = major == 8 && minor >= 0;
    bool is_sm90 = major == 9 && minor == 0;
    FFI_CHECK(is_sm90 || is_sm8x) << ffi::ErrorCode::kUnimplemented << "FlashAttention only supports Ampere GPUs or newer.";
    // We will support Turing in the near future
    // TORCH_CHECK(is_sm90 || is_sm8x || is_sm75, "FlashAttention only supports Turing GPUs or newer.");

    ffi::DataType dtype = q.element_type();
    FFI_CHECK(dtype == k.element_type() && dtype == v.element_type() && dtype == o->element_type()) << ffi::ErrorCode::kInvalidArgument
        << "query, key and value must have the same dtype";
    FFI_CHECK(dtype == ffi::DataType::F16 || dtype == ffi::DataType::BF16) << ffi::ErrorCode::kInvalidArgument << "FlashAttention only support fp16 and bf16 data type";
    if (dtype == ffi::DataType::BF16) {
        FFI_CHECK(is_sm90 || is_sm8x) << ffi::ErrorCode::kInvalidArgument << "bfloat16 is only supported on Ampere GPUs or newer";
    }

    const int batch_size = q.dimensions()[0];
    int seqlen_q = q.dimensions()[1];
    int num_heads = q.dimensions()[2];
    const int head_size = q.dimensions()[3];
    const int seqlen_k = k.dimensions()[1];
    const int num_heads_k = k.dimensions()[2];
    FFI_CHECK(batch_size > 0) << "batch size must be postive";
    FFI_CHECK(head_size <= 256) << "FlashAttention forward only supports head dimension at most 256";
    FFI_CHECK(num_heads % num_heads_k == 0) << "Number of heads in key/value must divide number of heads in query";

    if (window_size_left >= seqlen_k) { window_size_left = -1; }
    if (window_size_right >= seqlen_k) { window_size_right = -1; }

    // causal=true is the same as causal=false in this case
	if (seqlen_q == 1) { is_causal = false; }
    if (is_causal) { window_size_right = 0; }

    // Faster to transpose q from (b, 1, (nheads_kv ngroups), d) to (b, ngroups, nheads_kv, d) in this case
    // H/t Daniel Haziza
    // const int seqlenq_ngroups_swapped = seqlen_q == 1 && num_heads > num_heads_k && window_size_left < 0 && window_size_right < 0 && p_dropout == 0.f && head_size_og % 8 == 0 && !has_alibi;
	const int seqlenq_ngroups_swapped = false;
    // if (seqlenq_ngroups_swapped) {
    //     const int ngroups = num_heads / num_heads_k;
    //     q = q.reshape({batch_size, num_heads_k, ngroups, head_size_og}).transpose(1, 2);
    //     seqlen_q = ngroups;
    //     num_heads = num_heads_k;
    // }

    // Inputs and outputs are already padded to be divisible by 8.
	void *q_padded=q.untyped_data(), *k_padded=k.untyped_data(), *v_padded=v.untyped_data();

    FFI_CHECK(head_size % 8 == 0) << ffi::ErrorCode::kInvalidArgument
        << "head_size must be divisible by 8, got " << head_size;
    auto round_multiple = [](int x, int m) { return (x + m - 1) / m * m; };
    const int head_size_rounded = round_multiple(head_size, 32);
    const int seqlen_q_rounded = round_multiple(seqlen_q, 128);
    const int seqlen_k_rounded = round_multiple(seqlen_k, 128);

    void* p = nullptr;

    Flash_fwd_params params;
    FFI_RET_CHECK(set_params_fprop(params, dtype,
                     batch_size,
                     seqlen_q, seqlen_k,
                     seqlen_q_rounded, seqlen_k_rounded,
                     num_heads, num_heads_k,
                     head_size, head_size_rounded,
                     q_padded, k_padded, v_padded, o->untyped_data(),
                     /*cu_seqlens_q_d=*/nullptr,
                     /*cu_seqlens_k_d=*/nullptr,
                     /*seqused_k=*/nullptr,
                     nullptr,
                     lse->untyped_data(),
                     0.0,
                     softmax_scale,
                     window_size_left,
                     window_size_right));


	int sm_count;
	FFI_CUDA_CHECK(cudaDeviceGetAttribute(&sm_count, cudaDevAttrMultiProcessorCount, device));

    FFI_RET_CHECK(set_params_splitkv(&scratch, params, batch_size, num_heads,
                       head_size, seqlen_k, seqlen_q,
                       head_size_rounded, 0.0, /*num_splits*/0, sm_count,
                        dtype));

    int64_t counter_offset = params.b * params.h * 32;
    auto rng_state = scratch.Allocate(2 * sizeof(uint64_t), 8); // 2 * float64
    FFI_CHECK(rng_state.has_value()) << "Failed to allocate memory for RNG state";
    params.rng_state = reinterpret_cast<uint64_t*>(rng_state.value());

    // auto options = torch::TensorOptions().dtype(torch::kFloat32).device(torch::kCUDA);
    // auto rng_state = torch::empty({2}, options.dtype(torch::kInt64));
    // // Forward kernel will populate memory with the seed and offset.
    // params.rng_state = reinterpret_cast<uint64_t*>(rng_state.data_ptr());


    params.alibi_slopes_ptr = nullptr;

    if (seqlen_k > 0) {
		run_mha_fwd(params, stream);
		FFI_CUDA_CHECK(cudaStreamSynchronize(stream));
    } else {
		FFI_CHECK(false) << "seqlen_k is zero";
        // If seqlen_k == 0, then we have an empty tensor. We need to set the output to 0.
        // out.zero_();
        // softmax_lse.fill_(std::numeric_limits<float>::infinity());
    }

    return ffi::Error();
}

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
    ffi::ResultBuffer<ffi::F32> lse, // batch_size x num_heads x max_seqlen_q
    int max_seqlen_q,
    int max_seqlen_k,
    bool has_seqused_k,
    float softmax_scale,
    bool zero_tensors,
    bool is_causal,
    int window_size_left,
    int window_size_right) {

    // at::Tensor &q,  // total_q x num_heads x head_size, total_q := \sum_{i=0}^{b} s_i
    //            const at::Tensor &k,  // total_k x num_heads_k x head_size, total_k := \sum_{i=0}^{b} s_i
    //            const at::Tensor &v,  // total_k x num_heads_k x head_size, total_k := \sum_{i=0}^{b} s_i
    //            c10::optional<at::Tensor> &out_, // total_q x num_heads x head_size, total_k := \sum_{i=0}^{b} s_i
    //            const at::Tensor &cu_seqlens_q,  // b+1
    //            const at::Tensor &cu_seqlens_k,  // b+1
    //            c10::optional<at::Tensor> &seqused_k, // b. If given, only this many elements of each batch element's keys are used.
    //            c10::optional<at::Tensor> &alibi_slopes_, // num_heads or b x num_heads
    //            int max_seqlen_q,
    //            const int max_seqlen_k,
    //            const float p_dropout,
    //            const float softmax_scale,
    //            const bool zero_tensors,
    //            bool is_causal,
    //            int window_size_left,
    //            int window_size_right,
    //            const bool return_softmax,
    //            c10::optional<at::Generator> gen_) {

	int major, minor, sm_count;
	FFI_CUDA_CHECK(cudaDeviceGetAttribute(&major, cudaDevAttrComputeCapabilityMajor, device));
	FFI_CUDA_CHECK(cudaDeviceGetAttribute(&minor, cudaDevAttrComputeCapabilityMinor, device));
	FFI_CUDA_CHECK(cudaDeviceGetAttribute(&sm_count, cudaDevAttrMultiProcessorCount, device));

    // bool is_sm75 = major == 7 && minor == 5;
    bool is_sm8x = major == 8 && minor >= 0;
    bool is_sm90 = major == 9 && minor == 0;
    FFI_CHECK(is_sm90 || is_sm8x) << "FlashAttention only supports Ampere GPUs or newer.";
    // We will support Turing in the near future
    // TORCH_CHECK(is_sm90 || is_sm8x || is_sm75, "FlashAttention only supports Turing GPUs or newer.");

    auto q_dtype = q.element_type();
    FFI_CHECK(q_dtype == ffi::F16 || q_dtype == ffi::BF16) << 
                "FlashAttention only support fp16 and bf16 data type";
    if (q_dtype == ffi::BF16) {
        FFI_CHECK(is_sm90 || is_sm8x) << "bfloat16 is only supported on Ampere GPUs or newer";
    }
    FFI_CHECK(k.element_type() == q_dtype) << "query and key must have the same dtype";
    FFI_CHECK(v.element_type() == q_dtype) << "query and value must have the same dtype";
    FFI_CHECK(out->element_type() == q_dtype) << "query and out must have the same dtype";

    const auto sizes = q.dimensions();

    const int batch_size = cu_seqlens_q.element_count() - 1;
    int num_heads = sizes[1];
    const int head_size_og = sizes[2];
    const int total_k = k.dimensions()[0];
    const int num_heads_k = k.dimensions()[1];

    if (max_seqlen_q == 1) { is_causal = false; }  // causal=true is the same as causal=false in this case
    if (is_causal) { window_size_right = 0; }

    // void *cu_seqlens_q_d = cu_seqlens_q.untyped_data();

    // Faster to transpose q from (b, 1, (nheads_kv ngroups), d) to (b, ngroups, nheads_kv, d) in this case
    // H/t Daniel Haziza
    const bool seqlenq_ngroups_swapped = false;
    // const int seqlenq_ngroups_swapped = max_seqlen_q == 1 && num_heads > num_heads_k && window_size_left < 0 && window_size_right < 0 && p_dropout == 0.f && head_size_og % 8 == 0 && !alibi_slopes_.has_value();
    // if (seqlenq_ngroups_swapped) {
    //     const int ngroups = num_heads / num_heads_k;
    //     q = q.reshape({batch_size, num_heads_k, ngroups, head_size_og}).transpose(1, 2).reshape({batch_size * ngroups, num_heads_k, head_size_og});
    //     max_seqlen_q = ngroups;
    //     num_heads = num_heads_k;
    //     cu_seqlens_q_d = nullptr;
    // }

    const int total_q = q.dimensions()[0];

    FFI_CHECK(batch_size > 0) << "batch size must be positive";
    FFI_CHECK(head_size_og <= 256) << "FlashAttention forward only supports head dimension at most 256";
    FFI_CHECK(num_heads % num_heads_k == 0) << "Number of heads in key/value must divide number of heads in query";

    if (window_size_left >= max_seqlen_k) { window_size_left = -1; }
    if (window_size_right >= max_seqlen_k) { window_size_right = -1; }

    // CHECK_SHAPE(q, total_q, num_heads, head_size_og);
    // CHECK_SHAPE(k, total_k, num_heads_k, head_size_og);
    // CHECK_SHAPE(v, total_k, num_heads_k, head_size_og);
    // CHECK_SHAPE(cu_seqlens_q, batch_size + 1);
    // CHECK_SHAPE(cu_seqlens_k, batch_size + 1);

    if (has_seqused_k) {
        FFI_CHECK(seqused_k.dimensions().size() == 1 && seqused_k.dimensions()[0] == batch_size)
            << "seqused_k must be a 1D tensor of size batch_size";
    }

    // at::Tensor q_padded, k_padded, v_padded;
    // if (head_size_og % 8 != 0) {
    //     q_padded = torch::nn::functional::pad(q, torch::nn::functional::PadFuncOptions({0, 8 - head_size_og % 8}));
    //     k_padded = torch::nn::functional::pad(k, torch::nn::functional::PadFuncOptions({0, 8 - head_size_og % 8}));
    //     v_padded = torch::nn::functional::pad(v, torch::nn::functional::PadFuncOptions({0, 8 - head_size_og % 8}));
    // } else {
    //     q_padded = q;
    //     k_padded = k;
    //     v_padded = v;
    // }

    // at::Tensor out;
    // if (out_.has_value()) {
    //     out = out_.value();
    //     TORCH_CHECK(out.dtype() == q_dtype, "Output must have the same dtype as inputs");
    //     CHECK_DEVICE(out);
    //     TORCH_CHECK(out.stride(-1) == 1, "Output tensor must have contiguous last dimension");
    //     CHECK_SHAPE(out, total_q, num_heads, head_size_og);
    //     if (head_size_og % 8 != 0) { out = torch::empty_like(q_padded); }
    // } else {
    //     out = torch::empty_like(q_padded);
    // }

    auto round_multiple = [](int x, int m) { return (x + m - 1) / m * m; };
    const int head_size = round_multiple(head_size_og, 8);
    const int head_size_rounded = round_multiple(head_size, 32);
    const int seqlen_q_rounded = round_multiple(max_seqlen_q, 128);
    const int seqlen_k_rounded = round_multiple(max_seqlen_k, 128);

    // Otherwise the kernel will be launched from cuda:0 device
    // Cast to char to avoid compiler warning about narrowing
    // auto softmax_lse = torch::empty({batch_size, num_heads, max_seqlen_q}, opts.dtype(at::kFloat));

    if (zero_tensors) {
        FFI_CUDA_CHECK(cudaMemsetAsync(out->untyped_data(), 0, out->size_bytes(), stream));
        FFI_CUDA_CHECK(cudaMemsetAsync(lse->untyped_data(), 0, lse->size_bytes(), stream));
        // TODO: Should set to -inf instead of 0
        // softmax_lse.fill_(-std::numeric_limits<float>::infinity());
    }

    Flash_fwd_params params;
    set_params_fprop(params,
                    q_dtype,
                     batch_size,
                     max_seqlen_q, max_seqlen_k,
                     seqlen_q_rounded, seqlen_k_rounded,
                     num_heads, num_heads_k,
                     head_size, head_size_rounded,
                     q.untyped_data(), k.untyped_data(), v.untyped_data(), out->untyped_data(),
                     cu_seqlens_q.untyped_data(),
                     cu_seqlens_k.untyped_data(),
                     has_seqused_k ? seqused_k.untyped_data() : nullptr,
                     nullptr,
                     lse->untyped_data(),
                     0.0,
                     softmax_scale,
                     window_size_left,
                     window_size_right,
                     seqlenq_ngroups_swapped);
    if (seqlenq_ngroups_swapped) {
        // Only apply split-k for decoding
        set_params_splitkv(&scratch, params, batch_size, num_heads,
                           head_size, max_seqlen_k, max_seqlen_q,
                           head_size_rounded, 0.0, /*num_splits*/0, sm_count, q_dtype);
    }

    // Forward kernel will populate memory with the seed and offset.
    auto rng_state = scratch.Allocate(2 * 8, 8); // 2 * int64
    params.rng_state = (uint64_t*)rng_state.value();

    params.alibi_slopes_ptr = nullptr;

    if (max_seqlen_k > 0) {
        run_mha_fwd(params, stream);
        FFI_CUDA_CHECK(cudaStreamSynchronize(stream));
    } else {
        // If seqlen_k == 0, then we have an empty tensor. We need to set the output to 0.
        FFI_CUDA_CHECK(cudaMemsetAsync(out->untyped_data(), 0, out->size_bytes(), stream));
        FFI_CUDA_CHECK(cudaMemsetAsync(lse->untyped_data(), 0, lse->size_bytes(), stream));
    }

    // at::Tensor out_padded = out;
    // if (head_size_og % 8 != 0) {
    //     out = out.index({"...", torch::indexing::Slice(torch::indexing::None, head_size_og)});
    //     if (out_.has_value()) { out_.value().copy_(out); }
    // }

    // if (seqlenq_ngroups_swapped) {
    //     long size_before[] = {batch_size, max_seqlen_q, num_heads_k, head_size_og};
    //     long size_after[] = {batch_size, num_heads_k * max_seqlen_q, head_size_og};
    //     out = out.reshape(size_before).transpose(1, 2).reshape(size_after);
    //     out_padded = out_padded.reshape(size_before).transpose(1, 2).reshape(size_after);
    //     q_padded = q_padded.reshape(size_before).transpose(1, 2).reshape(size_after);
    //     softmax_lse = softmax_lse.reshape({batch_size, num_heads_k * max_seqlen_q, 1});
    // }

    return ffi::Error(); // Success
}
