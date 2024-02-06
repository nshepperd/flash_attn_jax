#include <stddef.h>
#include <cutlass/numeric_types.h>
#include <cute/layout.hpp>
#include <cuda_runtime_api.h>
#include <pybind11/pybind11.h>

#include "flash.h"
#include "exception.h"
#include "static_switch.h"
#include "check.h"
#include "flash_common.h"

void set_params_fprop(Flash_fwd_params &params,
					  ElementType element_type,
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
                      void *cu_seqlens_q_d,
                      void *cu_seqlens_k_d,
                      void *seqused_k,
                      void *p_d,
                      void *softmax_lse_d,
                      float p_dropout,
                      float softmax_scale,
                      int window_size_left,
                      int window_size_right,
                      bool seqlenq_ngroups_swapped) {

    // Reset the parameters
    memset(&params, 0, sizeof(params));

    params.is_bf16 = element_type == BF16;

    // Set the pointers and strides.
    params.q_ptr = q_ptr;
    params.k_ptr = k_ptr;
    params.v_ptr = v_ptr;
    // All stride are in elements, not bytes.
	// n l h d
	auto qs = cute::compact_row_major(cute::make_shape(b, seqlen_q, h, d));
	auto ks = cute::compact_row_major(cute::make_shape(b, seqlen_k, h_k, d));
	auto vs = cute::compact_row_major(cute::make_shape(b, seqlen_k, h_k, d));
	auto os = cute::compact_row_major(cute::make_shape(b, seqlen_q, h, d));
	params.q_row_stride = cute::get<1>(qs); // -3, seqlen
	params.k_row_stride = cute::get<1>(ks);
	params.v_row_stride = cute::get<1>(vs);
	params.q_head_stride = cute::get<2>(qs); // -2, h
	params.k_head_stride = cute::get<2>(ks);
	params.v_head_stride = cute::get<2>(vs);
    params.o_ptr = out_ptr;
	params.o_row_stride = cute::get<1>(os);
	params.o_head_stride = cute::get<2>(os);

    if (cu_seqlens_q_d == nullptr) {
        params.q_batch_stride = cute::get<0>(os);
        params.k_batch_stride = cute::get<0>(ks);
        params.v_batch_stride = cute::get<0>(vs);
        params.o_batch_stride = cute::get<0>(os);
		// XXX idk
        // if (seqlenq_ngroups_swapped) {
        //      params.q_batch_stride *= seqlen_q;
        //      params.o_batch_stride *= seqlen_q;
        // }
    }

    params.cu_seqlens_q = static_cast<int *>(cu_seqlens_q_d);
    params.cu_seqlens_k = static_cast<int *>(cu_seqlens_k_d);
    params.seqused_k = static_cast<int *>(seqused_k);

    // P = softmax(QK^T)
    params.p_ptr = p_d;

    // Softmax sum
    params.softmax_lse_ptr = softmax_lse_d;

    // Set the dimensions.
    params.b = b;
    params.h = h;
    params.h_k = h_k;
    params.h_h_k_ratio = h / h_k;
    params.seqlen_q = seqlen_q;
    params.seqlen_k = seqlen_k;
    params.seqlen_q_rounded = seqlen_q_rounded;
    params.seqlen_k_rounded = seqlen_k_rounded;
    params.d = d;
    params.d_rounded = d_rounded;

    // Set the different scale values.
    params.scale_softmax = softmax_scale;
    params.scale_softmax_log2 = softmax_scale * M_LOG2E;

    // Set this to probability of keeping an element to simplify things.
    params.p_dropout = 1.f - p_dropout;
    // Convert p from float to int so we don't have to convert the random uint to float to compare.
    // [Minor] We want to round down since when we do the comparison we use <= instead of <
    // params.p_dropout_in_uint = uint32_t(std::floor(params.p_dropout * 4294967295.0));
    // params.p_dropout_in_uint16_t = uint16_t(std::floor(params.p_dropout * 65535.0));
    params.p_dropout_in_uint8_t = uint8_t(std::floor(params.p_dropout * 255.0));
    params.rp_dropout = 1.f / params.p_dropout;
    params.scale_softmax_rp_dropout = params.rp_dropout * params.scale_softmax;
    CHECK(p_dropout < 1.f, "dropout must be <1");

    // Causal is the special case where window_size_right == 0 and window_size_left < 0.
    // Local is the more general case where window_size_right >= 0 or window_size_left >= 0.
    params.is_causal = window_size_left < 0 && window_size_right == 0;

    if (window_size_left < 0 && window_size_right >= 0) { window_size_left = seqlen_k; }
    if (window_size_left >= 0 && window_size_right < 0) { window_size_right = seqlen_k; }
    params.window_size_left = window_size_left;
    params.window_size_right = window_size_right;

    params.is_seqlens_k_cumulative = true;
}

// Find the number of splits that maximizes the occupancy. For example, if we have
// batch * n_heads = 48 and we have 108 SMs, having 2 splits (efficiency = 0.89) is
// better than having 3 splits (efficiency = 0.67). However, we also don't want too many
// splits as that would incur more HBM reads/writes.
// So we find the best efficiency, then find the smallest number of splits that gets 85%
// of the best efficiency.
int num_splits_heuristic(int batch_nheads_mblocks, int num_SMs, int num_n_blocks, int max_splits) {
    // If we have enough to almost fill the SMs, then just use 1 split
    if (batch_nheads_mblocks >= 0.8f * num_SMs) { return 1; }
    max_splits = std::min({max_splits, num_SMs, num_n_blocks});
    float max_efficiency = 0.f;
    std::vector<float> efficiency;
    efficiency.reserve(max_splits);
    auto ceildiv = [](int a, int b) { return (a + b - 1) / b; };
    // Some splits are not eligible. For example, if we have 64 blocks and choose 11 splits,
    // we'll have 6 * 10 + 4 blocks. If we choose 12 splits, we'll have 6 * 11 + (-2) blocks
    // (i.e. it's 11 splits anyway).
    // So we check if the number of blocks per split is the same as the previous num_splits.
    auto is_split_eligible = [&ceildiv, &num_n_blocks](int num_splits) {
        return num_splits == 1 || ceildiv(num_n_blocks, num_splits) != ceildiv(num_n_blocks, num_splits - 1);
    };
    for (int num_splits = 1; num_splits <= max_splits; num_splits++) {
        if (!is_split_eligible(num_splits)) {
            efficiency.push_back(0.f);
        } else {
            float n_waves = float(batch_nheads_mblocks * num_splits) / num_SMs;
            float eff = n_waves / ceil(n_waves);
            // printf("num_splits = %d, eff = %f\n", num_splits, eff);
            if (eff > max_efficiency) { max_efficiency = eff; }
            efficiency.push_back(eff);
        }
    }
    for (int num_splits = 1; num_splits <= max_splits; num_splits++) {
        if (!is_split_eligible(num_splits)) { continue; }
        if (efficiency[num_splits - 1] >= 0.85 * max_efficiency) {
            // printf("num_splits chosen = %d\n", num_splits);
            return num_splits;
        }
    }
    return 1;
}

void set_params_splitkv(Flash_fwd_params &params, const int batch_size,
						const int num_heads, const int head_size, const int max_seqlen_k, const int max_seqlen_q,
						const int head_size_rounded, const float p_dropout,
						const int num_splits, int multiProcessorCount, ElementType dtype) {
    // This needs to match with run_mha_fwd_splitkv_dispatch
    const int block_n = head_size <= 64 ? 256 : (head_size <= 128 ? 128 : 64);
    const int num_n_blocks = (max_seqlen_k + block_n - 1) / block_n;
    // Technically kBlockM = 64 only for the splitKV kernels, not the standard kernel.
    // In any case we don't expect seqlen_q to be larger than 64 for inference.
    const int num_m_blocks = (max_seqlen_q + 64 - 1) / 64;
    params.num_splits = num_splits;
	params.softmax_lseaccum_ptr = nullptr;
	params.oaccum_ptr = nullptr;
    if (p_dropout == 0.0f) {  // SplitKV is not implemented for dropout
        if (num_splits < 1) {
            params.num_splits = num_splits_heuristic(batch_size * num_heads * num_m_blocks, multiProcessorCount, num_n_blocks, 128);
        }
        if (params.num_splits > 1) {
            // at::Tensor softmax_lse_accum = torch::empty({params.num_splits, batch_size, num_heads, max_seqlen_q}, opts.dtype(at::kFloat));
            // at::Tensor out_accum = torch::empty({params.num_splits, batch_size, num_heads, max_seqlen_q, head_size_rounded}, opts.dtype(at::kFloat));
			C10_CUDA_CHECK(cudaMalloc((void**)&params.softmax_lseaccum_ptr, params.num_splits * batch_size * num_heads * max_seqlen_q * 4)); // float32
			C10_CUDA_CHECK(cudaMalloc((void**)&params.oaccum_ptr, params.num_splits * batch_size * num_heads * max_seqlen_q * head_size_rounded * 4));
            // params.softmax_lseaccum_ptr = softmax_lse_accum.data_ptr();
            // params.oaccum_ptr = out_accum.data_ptr();
        }
        CHECK(params.num_splits <= 128, "num_splits > 128 not supported");
    }
}
