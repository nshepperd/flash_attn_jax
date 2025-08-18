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


ffi::Error mha_fwd_impl(cudaStream_t stream, ffi::ScratchAllocator scratch, ffi::AnyBuffer q, ffi::AnyBuffer k,
             ffi::AnyBuffer v, ffi::Result<ffi::AnyBuffer> o,
             ffi::ResultBuffer<ffi::F32> lse, double softmax_scale,
            bool is_causal, int64_t window_size_left, int64_t window_size_right) {
	int device, major, minor;
    FFI_CUDA_CHECK(cudaStreamGetDevice(stream, &device));
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
		// C10_CUDA_CHECK(cudaStreamSynchronize(stream));
		// C10_CUDA_CHECK(cudaDeviceSynchronize());
    } else {
		FFI_CHECK(false) << "seqlen_k is zero";
        // If seqlen_k == 0, then we have an empty tensor. We need to set the output to 0.
        // out.zero_();
        // softmax_lse.fill_(std::numeric_limits<float>::infinity());
    }

    return ffi::Error();
}


// std::vector<at::Tensor>
// mha_varlen_fwd(at::Tensor &q,  // total_q x num_heads x head_size, total_q := \sum_{i=0}^{b} s_i
//                const at::Tensor &k,  // total_k x num_heads_k x head_size, total_k := \sum_{i=0}^{b} s_i
//                const at::Tensor &v,  // total_k x num_heads_k x head_size, total_k := \sum_{i=0}^{b} s_i
//                c10::optional<at::Tensor> &out_, // total_q x num_heads x head_size, total_k := \sum_{i=0}^{b} s_i
//                const at::Tensor &cu_seqlens_q,  // b+1
//                const at::Tensor &cu_seqlens_k,  // b+1
//                c10::optional<at::Tensor> &seqused_k, // b. If given, only this many elements of each batch element's keys are used.
//                c10::optional<at::Tensor> &alibi_slopes_, // num_heads or b x num_heads
//                int max_seqlen_q,
//                const int max_seqlen_k,
//                const float p_dropout,
//                const float softmax_scale,
//                const bool zero_tensors,
//                bool is_causal,
//                int window_size_left,
//                int window_size_right,
//                const bool return_softmax,
//                c10::optional<at::Generator> gen_) {

//     auto dprops = at::cuda::getCurrentDeviceProperties();
//     // bool is_sm75 = dprops->major == 7 && dprops->minor == 5;
//     bool is_sm8x = dprops->major == 8 && dprops->minor >= 0;
//     bool is_sm90 = dprops->major == 9 && dprops->minor == 0;
//     TORCH_CHECK(is_sm90 || is_sm8x, "FlashAttention only supports Ampere GPUs or newer.");
//     // We will support Turing in the near future
//     // TORCH_CHECK(is_sm90 || is_sm8x || is_sm75, "FlashAttention only supports Turing GPUs or newer.");

//     auto q_dtype = q.dtype();
//     TORCH_CHECK(q_dtype == torch::kFloat16 || q_dtype == torch::kBFloat16,
//                 "FlashAttention only support fp16 and bf16 data type");
//     if (q_dtype == torch::kBFloat16) {
//         TORCH_CHECK(is_sm90 || is_sm8x, "bfloat16 is only supported on Ampere GPUs or newer");
//     }
//     TORCH_CHECK(k.dtype() == q_dtype, "query and key must have the same dtype");
//     TORCH_CHECK(v.dtype() == q_dtype, "query and value must have the same dtype");
//     TORCH_CHECK(cu_seqlens_q.dtype() == torch::kInt32, "cu_seqlens_q must have dtype int32");
//     TORCH_CHECK(cu_seqlens_k.dtype() == torch::kInt32, "cu_seqlens_k must have dtype int32");

//     CHECK_DEVICE(q); CHECK_DEVICE(k); CHECK_DEVICE(v);
//     CHECK_DEVICE(cu_seqlens_q);
//     CHECK_DEVICE(cu_seqlens_k);

//     TORCH_CHECK(q.stride(-1) == 1, "Input tensor must have contiguous last dimension");
//     TORCH_CHECK(k.stride(-1) == 1, "Input tensor must have contiguous last dimension");
//     TORCH_CHECK(v.stride(-1) == 1, "Input tensor must have contiguous last dimension");
//     CHECK_CONTIGUOUS(cu_seqlens_q);
//     CHECK_CONTIGUOUS(cu_seqlens_k);

//     const auto sizes = q.sizes();

//     const int batch_size = cu_seqlens_q.numel() - 1;
//     int num_heads = sizes[1];
//     const int head_size_og = sizes[2];
//     const int total_k = k.size(0);
//     const int num_heads_k = k.size(1);

//     if (max_seqlen_q == 1 && !alibi_slopes_.has_value()) { is_causal = false; }  // causal=true is the same as causal=false in this case
//     if (is_causal) { window_size_right = 0; }

//     void *cu_seqlens_q_d = cu_seqlens_q.data_ptr();

//     // Faster to transpose q from (b, 1, (nheads_kv ngroups), d) to (b, ngroups, nheads_kv, d) in this case
//     // H/t Daniel Haziza
//     const int seqlenq_ngroups_swapped = max_seqlen_q == 1 && num_heads > num_heads_k && window_size_left < 0 && window_size_right < 0 && p_dropout == 0.f && head_size_og % 8 == 0 && !alibi_slopes_.has_value();
//     if (seqlenq_ngroups_swapped) {
//         const int ngroups = num_heads / num_heads_k;
//         q = q.reshape({batch_size, num_heads_k, ngroups, head_size_og}).transpose(1, 2).reshape({batch_size * ngroups, num_heads_k, head_size_og});
//         max_seqlen_q = ngroups;
//         num_heads = num_heads_k;
//         cu_seqlens_q_d = nullptr;
//     }

//     const int total_q = q.sizes()[0];

//     TORCH_CHECK(batch_size > 0, "batch size must be positive");
//     TORCH_CHECK(head_size_og <= 256, "FlashAttention forward only supports head dimension at most 256");
//     TORCH_CHECK(num_heads % num_heads_k == 0, "Number of heads in key/value must divide number of heads in query");

//     if (window_size_left >= max_seqlen_k) { window_size_left = -1; }
//     if (window_size_right >= max_seqlen_k) { window_size_right = -1; }

//     CHECK_SHAPE(q, total_q, num_heads, head_size_og);
//     CHECK_SHAPE(k, total_k, num_heads_k, head_size_og);
//     CHECK_SHAPE(v, total_k, num_heads_k, head_size_og);
//     CHECK_SHAPE(cu_seqlens_q, batch_size + 1);
//     CHECK_SHAPE(cu_seqlens_k, batch_size + 1);
//     if (seqused_k.has_value()){
//         auto seqused_k_ = seqused_k.value();
//         TORCH_CHECK(seqused_k_.dtype() == torch::kInt32, "seqused_k must have dtype int32");
//         TORCH_CHECK(seqused_k_.is_cuda(), "seqused_k must be on CUDA device");
//         TORCH_CHECK(seqused_k_.is_contiguous(), "seqused_k must be contiguous");
//         CHECK_SHAPE(seqused_k_, batch_size);
//     }

//     at::Tensor q_padded, k_padded, v_padded;
//     if (head_size_og % 8 != 0) {
//         q_padded = torch::nn::functional::pad(q, torch::nn::functional::PadFuncOptions({0, 8 - head_size_og % 8}));
//         k_padded = torch::nn::functional::pad(k, torch::nn::functional::PadFuncOptions({0, 8 - head_size_og % 8}));
//         v_padded = torch::nn::functional::pad(v, torch::nn::functional::PadFuncOptions({0, 8 - head_size_og % 8}));
//     } else {
//         q_padded = q;
//         k_padded = k;
//         v_padded = v;
//     }

//     at::Tensor out;
//     if (out_.has_value()) {
//         out = out_.value();
//         TORCH_CHECK(out.dtype() == q_dtype, "Output must have the same dtype as inputs");
//         CHECK_DEVICE(out);
//         TORCH_CHECK(out.stride(-1) == 1, "Output tensor must have contiguous last dimension");
//         CHECK_SHAPE(out, total_q, num_heads, head_size_og);
//         if (head_size_og % 8 != 0) { out = torch::empty_like(q_padded); }
//     } else {
//         out = torch::empty_like(q_padded);
//     }

//     auto round_multiple = [](int x, int m) { return (x + m - 1) / m * m; };
//     const int head_size = round_multiple(head_size_og, 8);
//     const int head_size_rounded = round_multiple(head_size, 32);
//     const int seqlen_q_rounded = round_multiple(max_seqlen_q, 128);
//     const int seqlen_k_rounded = round_multiple(max_seqlen_k, 128);

//     // Otherwise the kernel will be launched from cuda:0 device
//     // Cast to char to avoid compiler warning about narrowing
//     at::cuda::CUDAGuard device_guard{(char)q.get_device()};

//     auto opts = q.options();

//     auto softmax_lse = torch::empty({batch_size, num_heads, max_seqlen_q}, opts.dtype(at::kFloat));
//     at::Tensor p;
//     // Only return softmax if there's dropout to reduce compilation time
//     if (return_softmax) {
//         TORCH_CHECK(p_dropout > 0.0f, "return_softmax is only supported when p_dropout > 0.0");
//         p = torch::empty({ batch_size, num_heads, seqlen_q_rounded, seqlen_k_rounded }, opts);
//     }

//     if (zero_tensors) {
//         out.zero_();
//         softmax_lse.fill_(-std::numeric_limits<float>::infinity());
//         if (return_softmax) {p.zero_();}
//     }

//     Flash_fwd_params params;
//     set_params_fprop(params,
//                      batch_size,
//                      max_seqlen_q, max_seqlen_k,
//                      seqlen_q_rounded, seqlen_k_rounded,
//                      num_heads, num_heads_k,
//                      head_size, head_size_rounded,
//                      q_padded, k_padded, v_padded, out,
//                      cu_seqlens_q_d,
//                      cu_seqlens_k.data_ptr(),
//                      seqused_k.has_value() ? seqused_k.value().data_ptr() : nullptr,
//                      return_softmax ? p.data_ptr() : nullptr,
//                      softmax_lse.data_ptr(),
//                      p_dropout,
//                      softmax_scale,
//                      window_size_left,
//                      window_size_right,
//                      seqlenq_ngroups_swapped);
//     if (seqlenq_ngroups_swapped) {
//         // Only apply split-k for decoding
//         set_params_splitkv(params, batch_size, num_heads,
//                            head_size, max_seqlen_k, max_seqlen_q,
//                            head_size_rounded, p_dropout, /*num_splits*/0, dprops, opts);
//     }

//     // number of times random will be generated per thread, to offset philox counter in thc random
//     // state
//     // We use a custom RNG that increases the offset by batch_size * nheads * 32.
//     int64_t counter_offset = params.b * params.h * 32;
//     auto options = torch::TensorOptions().dtype(torch::kFloat32).device(torch::kCUDA);
//     auto rng_state = torch::empty({2}, options.dtype(torch::kInt64));
//     // Forward kernel will populate memory with the seed and offset.
//     params.rng_state = reinterpret_cast<uint64_t*>(rng_state.data_ptr());

//     if (p_dropout > 0.0)  {
//         auto gen = at::get_generator_or_default<at::CUDAGeneratorImpl>(
//             gen_, at::cuda::detail::getDefaultCUDAGenerator());
//         // See Note [Acquire lock when using random generators]
//         std::lock_guard<std::mutex> lock(gen->mutex_);
//         params.philox_args = gen->philox_cuda_state(counter_offset);
//     }

//     if (alibi_slopes_.has_value()) {
//         auto alibi_slopes = alibi_slopes_.value();
//         TORCH_CHECK(alibi_slopes.dtype() == torch::kFloat32, "ALiBi slopes must have dtype fp32");
//         CHECK_DEVICE(alibi_slopes);
//         TORCH_CHECK(alibi_slopes.stride(-1) == 1, "ALiBi slopes tensor must have contiguous last dimension");
//         TORCH_CHECK(alibi_slopes.sizes() == torch::IntArrayRef({num_heads}) || alibi_slopes.sizes() == torch::IntArrayRef({batch_size, num_heads}));
//         params.alibi_slopes_ptr = alibi_slopes.data_ptr();
//         params.alibi_slopes_batch_stride = alibi_slopes.dim() == 2 ? alibi_slopes.stride(0) : 0;
//     } else {
//         params.alibi_slopes_ptr = nullptr;
//     }

//     if (max_seqlen_k > 0) {
//         auto stream = at::cuda::getCurrentCUDAStream().stream();
//         run_mha_fwd(params, stream);
//     } else {
//         // If seqlen_k == 0, then we have an empty tensor. We need to set the output to 0.
//         out.zero_();
//         softmax_lse.fill_(std::numeric_limits<float>::infinity());
//     }

//     at::Tensor out_padded = out;
//     if (head_size_og % 8 != 0) {
//         out = out.index({"...", torch::indexing::Slice(torch::indexing::None, head_size_og)});
//         if (out_.has_value()) { out_.value().copy_(out); }
//     }

//     if (seqlenq_ngroups_swapped) {
//         long size_before[] = {batch_size, max_seqlen_q, num_heads_k, head_size_og};
//         long size_after[] = {batch_size, num_heads_k * max_seqlen_q, head_size_og};
//         out = out.reshape(size_before).transpose(1, 2).reshape(size_after);
//         out_padded = out_padded.reshape(size_before).transpose(1, 2).reshape(size_after);
//         q_padded = q_padded.reshape(size_before).transpose(1, 2).reshape(size_after);
//         softmax_lse = softmax_lse.reshape({batch_size, num_heads_k * max_seqlen_q, 1});
//     }

//     return {out, q_padded, k_padded, v_padded, out_padded, softmax_lse, p, rng_state};
// }
