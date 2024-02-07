/******************************************************************************
 * Copyright (c) 2024, Tri Dao.
 ******************************************************************************/

#include <stddef.h>
#include <cutlass/numeric_types.h>
#include <cuda_runtime_api.h>
#include <pybind11/pybind11.h>

#include "flash.h"
#include "exception.h"
#include "static_switch.h"
#include "check.h"

#include "flash_common.h"
#include "mha_fwd.h"
#include "mha_bwd.h"

// std::vector<at::Tensor>
// mha_fwd_kvcache(at::Tensor &q,                 // batch_size x seqlen_q x num_heads x head_size
//                 const at::Tensor &kcache,            // batch_size_c x seqlen_k x num_heads_k x head_size or num_blocks x page_block_size x num_heads_k x head_size if there's a block_table.
//                 const at::Tensor &vcache,            // batch_size_c x seqlen_k x num_heads_k x head_size or num_blocks x page_block_size x num_heads_k x head_size if there's a block_table.
//                 c10::optional<const at::Tensor> &k_, // batch_size x seqlen_knew x num_heads_k x head_size
//                 c10::optional<const at::Tensor> &v_, // batch_size x seqlen_knew x num_heads_k x head_size
//                 c10::optional<const at::Tensor> &seqlens_k_, // batch_size
//                 c10::optional<const at::Tensor> &rotary_cos_, // seqlen_ro x (rotary_dim / 2)
//                 c10::optional<const at::Tensor> &rotary_sin_, // seqlen_ro x (rotary_dim / 2)
//                 c10::optional<const at::Tensor> &cache_batch_idx_, // indices to index into the KV cache
//                 c10::optional<at::Tensor> &block_table_, // batch_size x max_num_blocks_per_seq
//                 c10::optional<at::Tensor> &alibi_slopes_, // num_heads or batch_size x num_heads
//                 c10::optional<at::Tensor> &out_,             // batch_size x seqlen_q x num_heads x head_size
//                 const float softmax_scale,
//                 bool is_causal,
//                 int window_size_left,
//                 int window_size_right,
//                 bool is_rotary_interleaved,   // if true, rotary combines indices 0 & 1, else indices 0 & rotary_dim / 2
//                 int num_splits
//                 ) {

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
//     TORCH_CHECK(kcache.dtype() == q_dtype, "query and key must have the same dtype");
//     TORCH_CHECK(vcache.dtype() == q_dtype, "query and value must have the same dtype");

//     CHECK_DEVICE(q); CHECK_DEVICE(kcache); CHECK_DEVICE(vcache);

//     TORCH_CHECK(q.stride(-1) == 1, "Input tensor must have contiguous last dimension");
//     TORCH_CHECK(kcache.stride(-1) == 1, "Input tensor must have contiguous last dimension");
//     TORCH_CHECK(vcache.stride(-1) == 1, "Input tensor must have contiguous last dimension");

//     at::Tensor block_table;
//     const bool paged_KV = block_table_.has_value();
//     if (paged_KV) {
//         TORCH_CHECK(!cache_batch_idx_.has_value(), "Paged KVcache does not support cache_batch_idx");
//         block_table = block_table_.value();
//         CHECK_DEVICE(block_table);
//         TORCH_CHECK(block_table.dtype() == torch::kInt32, "block_table must have dtype torch.int32");
//         TORCH_CHECK(block_table.stride(-1) == 1, "block_table must have contiguous last dimension");
//     }

//     const auto sizes = q.sizes();

//     const int batch_size = sizes[0];
//     int seqlen_q = sizes[1];
//     int num_heads = sizes[2];
//     const int head_size_og = sizes[3];

//     const int max_num_blocks_per_seq = !paged_KV ? 0 : block_table.size(1);
//     const int num_blocks = !paged_KV ? 0 : kcache.size(0);
//     const int page_block_size = !paged_KV ? 1 : kcache.size(1);
//     TORCH_CHECK(!paged_KV || page_block_size % 256 == 0, "Paged KV cache block size must be divisible by 256");
//     const int seqlen_k = !paged_KV ? kcache.size(1) : max_num_blocks_per_seq * page_block_size;
//     const int num_heads_k = kcache.size(2);
//     const int batch_size_c = !paged_KV ? kcache.size(0) : batch_size;
//     TORCH_CHECK(batch_size > 0, "batch size must be postive");
//     TORCH_CHECK(head_size_og <= 256, "FlashAttention forward only supports head dimension at most 256");
//     TORCH_CHECK(num_heads % num_heads_k == 0, "Number of heads in key/value must divide number of heads in query");

//     // causal=true is the same as causal=false in this case
//     if (seqlen_q == 1 && !alibi_slopes_.has_value()) { is_causal = false; }
//     if (is_causal) { window_size_right = 0; }

//     // Faster to transpose q from (b, 1, (nheads_kv ngroups), d) to (b, ngroups, nheads_kv, d) in this case
//     // H/t Daniel Haziza
//     const int seqlenq_ngroups_swapped = seqlen_q == 1 && num_heads > num_heads_k && window_size_left < 0 && window_size_right < 0 && head_size_og % 8 == 0 && !alibi_slopes_.has_value();
//     if (seqlenq_ngroups_swapped) {
//         const int ngroups = num_heads / num_heads_k;
//         q = q.reshape({batch_size, num_heads_k, ngroups, head_size_og}).transpose(1, 2);
//         seqlen_q = ngroups;
//         num_heads = num_heads_k;
//     }

//     if (window_size_left >= seqlen_k) { window_size_left = -1; }
//     if (window_size_right >= seqlen_k) { window_size_right = -1; }

//     CHECK_SHAPE(q, batch_size, seqlen_q, num_heads, head_size_og);
//     if (!paged_KV) {
//         CHECK_SHAPE(kcache, batch_size_c, seqlen_k, num_heads_k, head_size_og);
//         CHECK_SHAPE(vcache, batch_size_c, seqlen_k, num_heads_k, head_size_og);
//     } else {
//         CHECK_SHAPE(kcache, num_blocks, page_block_size, num_heads_k, head_size_og);
//         CHECK_SHAPE(vcache, num_blocks, page_block_size, num_heads_k, head_size_og);
//         CHECK_SHAPE(block_table, batch_size, max_num_blocks_per_seq);
//     }

//     at::Tensor q_padded, kcache_padded, vcache_padded;
//     if (head_size_og % 8 != 0) {
//         q_padded = torch::nn::functional::pad(q, torch::nn::functional::PadFuncOptions({0, 8 - head_size_og % 8}));
//         kcache_padded = torch::nn::functional::pad(kcache, torch::nn::functional::PadFuncOptions({0, 8 - head_size_og % 8}));
//         vcache_padded = torch::nn::functional::pad(vcache, torch::nn::functional::PadFuncOptions({0, 8 - head_size_og % 8}));
//     } else {
//         q_padded = q;
//         kcache_padded = kcache;
//         vcache_padded = vcache;
//     }

//     at::Tensor out;
//     if (out_.has_value()) {
//         out = out_.value();
//         TORCH_CHECK(out.dtype() == q_dtype, "Output must have the same dtype as inputs");
//         CHECK_DEVICE(out);
//         TORCH_CHECK(out.stride(-1) == 1, "Output tensor must have contiguous last dimension");
//         CHECK_SHAPE(out, batch_size, seqlen_q, num_heads, head_size_og);
//         if (head_size_og % 8 != 0) { out = torch::empty_like(q_padded); }
//     } else {
//         out = torch::empty_like(q_padded);
//     }

//     auto round_multiple = [](int x, int m) { return (x + m - 1) / m * m; };
//     const int head_size = round_multiple(head_size_og, 8);
//     const int head_size_rounded = round_multiple(head_size, 32);
//     const int seqlen_q_rounded = round_multiple(seqlen_q, 128);
//     const int seqlen_k_rounded = round_multiple(seqlen_k, 128);

//     // Otherwise the kernel will be launched from cuda:0 device
//     // Cast to char to avoid compiler warning about narrowing
//     at::cuda::CUDAGuard device_guard{(char)q.get_device()};

//     auto opts = q.options();

//     auto softmax_lse = torch::empty({batch_size, num_heads, seqlen_q}, opts.dtype(at::kFloat));

//     Flash_fwd_params params;
//     set_params_fprop(params,
//                      batch_size,
//                      seqlen_q, seqlen_k,
//                      seqlen_q_rounded, seqlen_k_rounded,
//                      num_heads, num_heads_k,
//                      head_size, head_size_rounded,
//                      q_padded, kcache_padded, vcache_padded, out,
//                      /*cu_seqlens_q_d=*/nullptr,
//                      /*cu_seqlens_k_d=*/nullptr,
//                      /*seqused_k=*/nullptr,
//                      /*p_ptr=*/nullptr,
//                      softmax_lse.data_ptr(),
//                      /*p_dropout=*/0.f,
//                      softmax_scale,
//                      window_size_left,
//                      window_size_right);

//     at::Tensor k, v, k_padded, v_padded;
//     if (k_.has_value()) {
//         TORCH_CHECK(v_.has_value(), "If key is supplied, value must also be passed in");
//         TORCH_CHECK(seqlens_k_.has_value(), "If key is supplied, seqlens_k must also be passed in");
//         TORCH_CHECK(seqlen_q <= seqlen_k, "If key is supplied, it must have seqlen <= the seqlen of the KV cache");
//         k = k_.value();
//         v = v_.value();
//         TORCH_CHECK(k.dtype() == q_dtype, "Key must have the same dtype as query");
//         TORCH_CHECK(v.dtype() == q_dtype, "Value must have the same dtype as query");
//         CHECK_DEVICE(k); CHECK_DEVICE(v);
//         TORCH_CHECK(k.stride(-1) == 1, "Key tensor must have contiguous last dimension");
//         TORCH_CHECK(v.stride(-1) == 1, "Value tensor must have contiguous last dimension");
//         int seqlen_knew = k.size(1);
//         CHECK_SHAPE(k, batch_size, seqlen_knew, num_heads_k, head_size_og);
//         CHECK_SHAPE(v, batch_size, seqlen_knew, num_heads_k, head_size_og);
//         if (head_size_og % 8 != 0) {
//             k_padded = torch::nn::functional::pad(k, torch::nn::functional::PadFuncOptions({0, 8 - head_size_og % 8}));
//             v_padded = torch::nn::functional::pad(v, torch::nn::functional::PadFuncOptions({0, 8 - head_size_og % 8}));
//         } else {
//             k_padded = k;
//             v_padded = v;
//         }
//         params.seqlen_knew = seqlen_knew;
//         params.knew_ptr = k_padded.data_ptr();
//         params.vnew_ptr = v_padded.data_ptr();
//         // All stride are in elements, not bytes.
//         params.knew_batch_stride = k_padded.stride(0);
//         params.vnew_batch_stride = v_padded.stride(0);
//         params.knew_row_stride = k_padded.stride(-3);
//         params.vnew_row_stride = v_padded.stride(-3);
//         params.knew_head_stride = k_padded.stride(-2);
//         params.vnew_head_stride = v_padded.stride(-2);
//     }

//     if (seqlens_k_.has_value()) {
//         auto seqlens_k = seqlens_k_.value();
//         TORCH_CHECK(seqlens_k.dtype() == torch::kInt32, "seqlens_k must have dtype int32");
//         CHECK_DEVICE(seqlens_k);
//         CHECK_CONTIGUOUS(seqlens_k);
//         CHECK_SHAPE(seqlens_k, batch_size);
//         params.cu_seqlens_k = static_cast<int *>(seqlens_k.data_ptr());
//     }
//     params.is_seqlens_k_cumulative = !(seqlens_k_.has_value());

//     if (rotary_cos_.has_value()) {
//         TORCH_CHECK(k_.has_value(), "If rotary cos/sin are provided, new key / value to be appended to KV cache must also be provided");
//         auto rotary_cos = rotary_cos_.value();
//         CHECK_DEVICE(rotary_cos);
//         params.rotary_dim = rotary_cos.size(1) * 2;
//         TORCH_CHECK(params.rotary_dim <= head_size, "rotary_dim must be <= headdim");
//         TORCH_CHECK(params.rotary_dim % 16 == 0, "Only rotary dimensions divisible by 16 are currently supported");
//         const int seqlen_ro = rotary_cos.size(0);
//         TORCH_CHECK(seqlen_ro >= seqlen_k, "cos/sin seqlen must be at least the seqlen of KV cache");
//         CHECK_SHAPE(rotary_cos, seqlen_ro, params.rotary_dim / 2);
//         CHECK_CONTIGUOUS(rotary_cos);
//         TORCH_CHECK(rotary_cos.scalar_type() == q_dtype, "rotary_cos must have the same dtype as query");

//         TORCH_CHECK(rotary_sin_.has_value(), "If rotary cos is provided, rotary sin must also be provided");
//         auto rotary_sin = rotary_sin_.value();
//         CHECK_DEVICE(rotary_sin);
//         CHECK_SHAPE(rotary_sin, seqlen_ro, params.rotary_dim / 2);
//         CHECK_CONTIGUOUS(rotary_sin);
//         TORCH_CHECK(rotary_sin.scalar_type() == q_dtype, "rotary_cos must have the same dtype as query");
//         params.rotary_cos_ptr = rotary_cos.data_ptr();
//         params.rotary_sin_ptr = rotary_sin.data_ptr();
//         params.is_rotary_interleaved = is_rotary_interleaved;
//     } else {
//         params.rotary_dim = 0;
//     }

//     if (cache_batch_idx_.has_value()) {
//         auto cache_batch_idx = cache_batch_idx_.value();
//         CHECK_DEVICE(cache_batch_idx);
//         CHECK_CONTIGUOUS(cache_batch_idx);
//         TORCH_CHECK(cache_batch_idx.scalar_type() == torch::kInt32, "cache_batch_idx must have dtype int32");
//         params.cache_batch_idx = reinterpret_cast<int *>(cache_batch_idx.data_ptr());
//     }

//     set_params_splitkv(params, batch_size, num_heads,
//                        head_size, seqlen_k, seqlen_q,
//                        head_size_rounded, /*dropout*/0.f, num_splits, dprops, opts);

//     if (paged_KV) {
//         params.block_table = block_table.data_ptr<int>();
//         params.block_table_batch_stride = block_table.stride(0);
//     }
//     params.page_block_size = page_block_size;

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

//     auto stream = at::cuda::getCurrentCUDAStream().stream();
//     // Only split kernel supports appending to KV cache, or indexing to the cache with cache_batch_idx,
//     // or paged KV cache
//     run_mha_fwd(params, stream, /*force_split_kernel=*/k_.has_value() || cache_batch_idx_.has_value() || paged_KV);

//     if (head_size_og % 8 != 0) {
//         out = out.index({"...", torch::indexing::Slice(torch::indexing::None, head_size_og)});
//         if (out_.has_value()) { out_.value().copy_(out); }
//         if (k_.has_value()) {
//             // It's expensive to copy the KV cache here for the case where head size not divisible by 8,
//             // but we don't expect to get this case in practice. This is just so that the code works for that case.
//             kcache.copy_(kcache_padded.index({"...", torch::indexing::Slice(torch::indexing::None, head_size_og)}));
//             vcache.copy_(vcache_padded.index({"...", torch::indexing::Slice(torch::indexing::None, head_size_og)}));
//         }
//     }

//     if (seqlenq_ngroups_swapped) {
//         out = out.transpose(1, 2).reshape({batch_size, 1, num_heads_k * seqlen_q, head_size_og});
//         softmax_lse = softmax_lse.reshape({batch_size, num_heads_k * seqlen_q, 1});
//     }
//     return {out, softmax_lse};
// }

namespace {

template <typename T> pybind11::capsule EncapsulateFunction(T *fn) {
  return pybind11::capsule(reinterpret_cast<void *>(fn), "xla._CUSTOM_CALL_TARGET");
}

template <typename T>
inline std::string PackDescriptorAsString(const T& descriptor) {
  return std::string(reinterpret_cast<const char*>(&descriptor), sizeof(T));
}

template <typename T> pybind11::bytes PackDescriptor(const T &descriptor) {
  return pybind11::bytes(PackDescriptorAsString(descriptor));
}

pybind11::bytes make_mha_fwd_args(	float p_dropout,
									float softmax_scale,
									bool is_causal,
									int window_size_left,
									int window_size_right,
									bool return_softmax,
									int n, int l, int h, int d,
									int l_k, int h_k,
									ElementType dtype,
									uint64_t seed) {
	return PackDescriptor(mha_fwd_args{p_dropout, softmax_scale, is_causal, window_size_left, window_size_right, return_softmax, n, l, h, d, l_k, h_k, dtype, seed});
}

pybind11::bytes make_mha_bwd_args(	float p_dropout,
									float softmax_scale,
									bool is_causal,
									int window_size_left,
									int window_size_right,
									bool deterministic,
									int n, int l, int h, int d,
									int l_k, int h_k,
									ElementType dtype,
									uint64_t seed) {
	return PackDescriptor(mha_bwd_args{p_dropout, softmax_scale, is_causal, window_size_left, window_size_right, deterministic, n, l, h, d, l_k, h_k, dtype, seed});
}

pybind11::dict Registrations() {
  pybind11::dict dict;
  dict["flash_mha_fwd"] = EncapsulateFunction(mha_fwd);
  dict["flash_mha_bwd"] = EncapsulateFunction(mha_bwd);
  return dict;
}


PYBIND11_MODULE(flash_api, m) {
    m.doc() = "FlashAttention";
	m.def("get_registrations", &Registrations);
	m.def("make_flash_mha_fwd_args", &make_mha_fwd_args);
	m.def("make_flash_mha_bwd_args", &make_mha_bwd_args);
	pybind11::enum_<ElementType>(m, "ElementType")
		.value("BF16", BF16)
		.value("FP16", FP16)
		.export_values();

    // m.def("varlen_fwd", &mha_varlen_fwd, "Forward pass (variable length)");
    // m.def("bwd", &mha_bwd, "Backward pass");
    // m.def("varlen_bwd", &mha_varlen_bwd, "Backward pass (variable length)");
    // m.def("fwd_kvcache", &mha_fwd_kvcache, "Forward pass, with KV-cache");
}

}
