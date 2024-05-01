#include <stddef.h>
#include <cutlass/numeric_types.h>
#include <cuda_runtime_api.h>
#include <pybind11/pybind11.h>
#include <cute/layout.hpp>

#include "flash.h"
#include "exception.h"
#include "static_switch.h"
#include "check.h"

#include "flash_common.h"
#include "mha_bwd.h"

void set_params_dgrad(Flash_bwd_params &params,
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

    set_params_fprop(params, element_type,
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
                     window_size_right);

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
}

void run_mha_bwd(Flash_bwd_params &params, cudaStream_t stream) {
    FP16_SWITCH(!params.is_bf16, [&] {
        HEADDIM_SWITCH(params.d, [&] {
            run_mha_bwd_<elem_type, kHeadDim>(params, stream);
        });
    });
}

void
mha_bwd(cudaStream_t stream, void **buffers, const char* opaque, size_t opaque_len) {
	// (const at::Tensor &dout,  // batch_size x seqlen_q x num_heads, x head_size_og
    //     const at::Tensor &q,   // batch_size x seqlen_q x num_heads x head_size
    //     const at::Tensor &k,   // batch_size x seqlen_k x num_heads_k x head_size
    //     const at::Tensor &v,   // batch_size x seqlen_k x num_heads_k x head_size
    //     const at::Tensor &out,   // batch_size x seqlen_q x num_heads x head_size
    //     const at::Tensor &softmax_lse,     // b x h x seqlen_q
    //     c10::optional<at::Tensor> &dq_,   // batch_size x seqlen_q x num_heads x head_size
    //     c10::optional<at::Tensor> &dk_,   // batch_size x seqlen_k x num_heads_k x head_size
    //     c10::optional<at::Tensor> &dv_,   // batch_size x seqlen_k x num_heads_k x head_size
    //     c10::optional<at::Tensor> &alibi_slopes_, // num_heads or batch_size x num_heads
	void* dout = buffers[0];
	void* q = buffers[1];
	void* k = buffers[2];
	void* v = buffers[3];
	void* o = buffers[4];
	void* lse = buffers[5];

	void* dq = buffers[6];
	void* dk = buffers[7];
	void* dv = buffers[8];

	auto args = Unpack<mha_bwd_args>(opaque, opaque_len);

	int window_size_right = args.window_size_right;
	int window_size_left = args.window_size_left;

    //     const float p_dropout,         // probability to drop
    //     const float softmax_scale,
    //     const bool is_causal,
    //     int window_size_left,
    //     int window_size_right,
    //     const bool deterministic,
    //     c10::optional<at::Generator> gen_,
    //     c10::optional<at::Tensor> &rng_state) {

	int device, major, minor, sm_count;
	C10_CUDA_CHECK(cudaGetDevice(&device));
	C10_CUDA_CHECK(cudaDeviceGetAttribute(&major, cudaDevAttrComputeCapabilityMajor, device));
	C10_CUDA_CHECK(cudaDeviceGetAttribute(&minor, cudaDevAttrComputeCapabilityMinor, device));
	C10_CUDA_CHECK(cudaDeviceGetAttribute(&sm_count, cudaDevAttrMultiProcessorCount, device));


    if (args.is_causal) { window_size_right = 0; }
    // auto dprops = at::cuda::getCurrentDeviceProperties();
    // bool is_sm75 = dprops->major == 7 && dprops->minor == 5;
    bool is_sm8x = major == 8 && minor >= 0;
    bool is_sm80 = major == 8 && minor == 0;
    bool is_sm90 = major == 9 && minor == 0;
    CHECK(is_sm90 || is_sm8x, "FlashAttention only supports Ampere GPUs or newer.");
    // We will support Turing in the near future
    // TORCH_CHECK(is_sm90 || is_sm8x || is_sm75, "FlashAttention only supports Turing GPUs or newer.");

    bool is_dropout = args.p_dropout > 0.0;

    auto q_dtype = args.dtype;
    // TORCH_CHECK(q_dtype == torch::kFloat16 || q_dtype == torch::kBFloat16,
    //             "FlashAttention only support fp16 and bf16 data type");
    if (q_dtype == BF16) {
        CHECK(is_sm90 || is_sm8x, "bfloat16 is only supported on Ampere GPUs or newer");
    }

    // TORCH_CHECK(k.dtype() == q_dtype, "query and key must have the same dtype");
    // TORCH_CHECK(v.dtype() == q_dtype, "query and value must have the same dtype");
    // TORCH_CHECK(out.dtype() == q_dtype, "query and out must have the same dtype");
    // TORCH_CHECK(dout.dtype() == q_dtype, "query and dout must have the same dtype");

    // CHECK_DEVICE(q); CHECK_DEVICE(k); CHECK_DEVICE(v);
    // CHECK_DEVICE(out); CHECK_DEVICE(dout); CHECK_DEVICE(softmax_lse);

    // TORCH_CHECK(q.stride(-1) == 1, "Input tensor must have contiguous last dimension");
    // TORCH_CHECK(k.stride(-1) == 1, "Input tensor must have contiguous last dimension");
    // TORCH_CHECK(v.stride(-1) == 1, "Input tensor must have contiguous last dimension");
    // TORCH_CHECK(out.stride(-1) == 1, "out tensor must have contiguous last dimension");
    // TORCH_CHECK(dout.stride(-1) == 1, "dout tensor must have contiguous last dimension");

    // const auto sizes = q.sizes();

    const int batch_size = args.n;
    const int seqlen_q = args.l;
    const int num_heads = args.h;
    const int head_size_og = args.d; //dout.size(3);
    const int head_size = args.d + (8 - head_size_og%8) % 8; //sizes[3];
    const int seqlen_k = args.l_k;
    const int num_heads_k = args.h_k; //k.size(2);
    CHECK(batch_size > 0, "batch size must be positive");
    CHECK(head_size % 8 == 0, "head_size should be a multiple of 8");
    CHECK(head_size <= 256, "FlashAttention backward only supports head dimension at most 256");
    if (head_size > 192) {
        CHECK(is_sm80 || is_sm90, "FlashAttention backward for head dim > 192 requires A100/A800 or H100/H800");
    }
    CHECK(num_heads % num_heads_k == 0, "Number of heads in key/value must divide number of heads in query");

    auto round_multiple = [](int x, int m) { return (x + m - 1) / m * m; };
    const int head_size_rounded = round_multiple(head_size, 32);
    const int seqlen_q_rounded = round_multiple(seqlen_q, 128);
    const int seqlen_k_rounded = round_multiple(seqlen_k, 128);

    CHECK(head_size == round_multiple(head_size_og, 8), "head_size must be head_size_og rounded to a multiple of 8");

    if (window_size_left >= seqlen_k) { window_size_left = -1; }
    if (window_size_right >= seqlen_k) { window_size_right = -1; }

    // CHECK_SHAPE(q, batch_size, seqlen_q, num_heads, head_size);
    // CHECK_SHAPE(k, batch_size, seqlen_k, num_heads_k, head_size);
    // CHECK_SHAPE(v, batch_size, seqlen_k, num_heads_k, head_size);
    // CHECK_SHAPE(out, batch_size, seqlen_q, num_heads, head_size);
    // CHECK_SHAPE(dout, batch_size, seqlen_q, num_heads, head_size_og);

    // at::Tensor dq, dk, dv;
    // if (dq_.has_value()) {
    //     dq = dq_.value();
    //     TORCH_CHECK(dq.dtype() == q_dtype, "dq must have the same dtype as q");
    //     CHECK_DEVICE(dq);
    //     TORCH_CHECK(dq.stride(-1) == 1, "dq must have contiguous last dimension");
    //     CHECK_SHAPE(dq, batch_size, seqlen_q, num_heads, head_size);
    // } else {
    //     dq = torch::empty_like(q);
    // }
    // if (dk_.has_value()) {
    //     dk = dk_.value();
    //     TORCH_CHECK(dk.dtype() == q_dtype, "dk must have the same dtype as q");
    //     CHECK_DEVICE(dk);
    //     TORCH_CHECK(dk.stride(-1) == 1, "dk must have contiguous last dimension");
    //     CHECK_SHAPE(dk, batch_size, seqlen_k, num_heads_k, head_size);
    // } else {
    //     dk = torch::empty_like(k);
    // }
    // if (dv_.has_value()) {
    //     dv = dv_.value();
    //     TORCH_CHECK(dv.dtype() == q_dtype, "dv must have the same dtype as q");
    //     CHECK_DEVICE(dv);
    //     TORCH_CHECK(dv.stride(-1) == 1, "dv must have contiguous last dimension");
    //     CHECK_SHAPE(dv, batch_size, seqlen_k, num_heads_k, head_size);
    // } else {
    //     dv = torch::empty_like(k);
    // }

    // at::Tensor dout_padded;

    // if (head_size_og % 8 != 0) {
	// 	CHECK(false, "can't pad");
    //     // dout_padded = torch::nn::functional::pad(dout, torch::nn::functional::PadFuncOptions({0, 8 - head_size_og % 8}));
    // } else {
    //     // dout_padded = dout;
    // }

    // bool loop = seqlen_k > blocksize_c;
    // TODO: change later, for now set to true for simplicity
    bool loop = true;

    // Otherwise the kernel will be launched from cuda:0 device
    // Cast to char to avoid compiler warning about narrowing
    // at::cuda::CUDAGuard device_guard{(char)q.get_device()};

    void* softmax_d = nullptr;
	C10_CUDA_CHECK(cudaMalloc(&softmax_d, batch_size * num_heads * seqlen_q_rounded * 4));
    void* dq_accum = nullptr;
    void* dk_accum = nullptr;
	void* dv_accum = nullptr;
    if (loop) {
        if (!args.deterministic) {
			C10_CUDA_CHECK(cudaMalloc(&dq_accum, batch_size * seqlen_q_rounded * num_heads * head_size_rounded * 4));
			C10_CUDA_CHECK(cudaMemset(dq_accum, 0, batch_size * seqlen_q_rounded * num_heads * head_size_rounded * 4));
        } else {
            const int nsplits = (sm_count + batch_size * num_heads - 1) / (batch_size * num_heads);
			C10_CUDA_CHECK(cudaMalloc(&dq_accum, nsplits * batch_size * seqlen_q_rounded * num_heads * head_size_rounded * 4));
			// previously allocated with torch.zeros, so i guess we need to zero it
			C10_CUDA_CHECK(cudaMemset(dq_accum, 0, nsplits * batch_size * seqlen_q_rounded * num_heads * head_size_rounded * 4));
        }
        // dk_accum = torch::empty({batch_size, num_heads_k, seqlen_k_rounded, head_size_rounded}, opts.dtype(at::kFloat));
        // dv_accum = torch::empty({batch_size, num_heads_k, seqlen_k_rounded, head_size_rounded}, opts.dtype(at::kFloat));
    }


    // For MQA, dk and dv are expanded to the same n_heads as dq (handled in xla).
    // After returning the result, it gets reduced to the original size by summing, so we don't need to do anything here.
	void* dk_expanded = dk;
	void* dv_expanded = dv;
    // at::Tensor dk_expanded, dv_expanded;
    // if (num_heads_k != num_heads) {  // MQA / GQA
    //     dk_expanded = torch::empty({batch_size, seqlen_k, num_heads, head_size}, opts);
    //     dv_expanded = torch::empty({batch_size, seqlen_k, num_heads, head_size}, opts);
    // } else {
    //     dk_expanded = dk;
    //     dv_expanded = dv;
    // }

    Flash_bwd_params params;

    set_params_dgrad(params,
					 args.dtype,
                     batch_size,
                     seqlen_q, seqlen_k,
                     seqlen_q_rounded, seqlen_k_rounded,
                     num_heads, num_heads_k,
                     head_size, head_size_rounded,
                     q, k, v, o,
                     dout, dq, dk_expanded, dv_expanded,
                     nullptr,
                     nullptr,
                     loop ? dq_accum : nullptr,
                     // loop ? dk_accum.data_ptr() : nullptr,
                     // loop ? dv_accum.data_ptr() : nullptr,
                     nullptr,
                     nullptr,
                     lse,
                     softmax_d,
                     args.p_dropout,
                     args.softmax_scale,
                     window_size_left,
                     window_size_right,
                     args.deterministic);
    params.dq_accum_split_stride = !args.deterministic ? 0 : (batch_size * seqlen_q_rounded * num_heads * head_size_rounded);

    auto launch = &run_mha_bwd;

	C10_CUDA_CHECK(cudaMalloc((void**)&params.rng_state, 2 * 8)); // 2 * float64
    if (is_dropout)  {
		CHECK(false, "don't support dropout yet");
    }

    // auto gen = at::get_generator_or_default<at::CUDAGeneratorImpl>(
    //     gen_, at::cuda::detail::getDefaultCUDAGenerator());

    // // We use a custom RNG that increases the offset by batch_size * nheads * 32.
    // int64_t counter_offset = params.b * params.h * 32;

    // if ( rng_state.has_value() ) {
    //     params.rng_state = reinterpret_cast<uint64_t*>(rng_state.value().data_ptr());
    // } else if( is_dropout ) {
    //     // See Note [Acquire lock when using random generators]
    //     std::lock_guard<std::mutex> lock(gen->mutex_);
    //     params.philox_args = gen->philox_cuda_state(counter_offset);
    //     auto seeds = at::cuda::philox::unpack(params.philox_args);
    //     params.rng_state[0] = std::get<0>(seeds);
    //     params.rng_state[1] = std::get<1>(seeds);
    // }

    // if (alibi_slopes_.has_value()) {
    //     auto alibi_slopes = alibi_slopes_.value();
    //     TORCH_CHECK(alibi_slopes.dtype() == torch::kFloat32, "ALiBi slopes must have dtype fp32");
    //     CHECK_DEVICE(alibi_slopes);
    //     TORCH_CHECK(alibi_slopes.stride(-1) == 1, "ALiBi slopes tensor must have contiguous last dimension");
    //     TORCH_CHECK(alibi_slopes.sizes() == torch::IntArrayRef({num_heads}) || alibi_slopes.sizes() == torch::IntArrayRef({batch_size, num_heads}));
    //     params.alibi_slopes_ptr = alibi_slopes.data_ptr();
    //     params.alibi_slopes_batch_stride = alibi_slopes.dim() == 2 ? alibi_slopes.stride(0) : 0;
    // } else {
    //     params.alibi_slopes_ptr = nullptr;
    // }

    if (seqlen_q > 0) {
        launch(params, stream);
    } else {
		CHECK(false, "seqlen_q == 0");
        // If seqlen_q == 0, then we have an empty tensor. We need to set the output to 0.
        // dk_expanded.zero_();
        // dv_expanded.zero_();
        // softmax_d.zero_();
    }

	C10_CUDA_CHECK(cudaFree(params.rng_state));
	if(softmax_d != nullptr) {
		C10_CUDA_CHECK(cudaFree(softmax_d));
	}
	if(dq_accum != nullptr) {
		C10_CUDA_CHECK(cudaFree(dq_accum));
	}
	if(dk_accum != nullptr) {
		C10_CUDA_CHECK(cudaFree(dk_accum));
	}
	if(dv_accum != nullptr) {
		C10_CUDA_CHECK(cudaFree(dv_accum));
	}

    // For MQA/GQA we need to sum dK and dV across the groups
    if (num_heads_k != num_heads) {
		// CHECK(false, "don't handle MQA yet");
        // at::sum_out(dk, at::reshape(dk_expanded, {batch_size, seqlen_k, num_heads_k, num_heads / num_heads_k, head_size}), {3});
        // at::sum_out(dv, at::reshape(dv_expanded, {batch_size, seqlen_k, num_heads_k, num_heads / num_heads_k, head_size}), {3});
    }
    // if (head_size_og % 8 != 0) {
	// 	CHECK(false, "can't slice");
    //     // dq = dq.index({"...", torch::indexing::Slice(torch::indexing::None, head_size_og)});
    //     // dk = dk.index({"...", torch::indexing::Slice(torch::indexing::None, head_size_og)});
    //     // dv = dv.index({"...", torch::indexing::Slice(torch::indexing::None, head_size_og)});
    // }

    // return { dq, dk, dv, softmax_d };
}

// std::vector<at::Tensor>
// mha_varlen_bwd(const at::Tensor &dout,  // total_q x num_heads, x head_size
//                const at::Tensor &q,   // total_q x num_heads x head_size, total_q := \sum_{i=0}^{b} s_i
//                const at::Tensor &k,   // total_k x num_heads_k x head_size, total_k := \sum_{i=0}^{b} s_i
//                const at::Tensor &v,   // total_k x num_heads_k x head_size, total_k := \sum_{i=0}^{b} s_i
//                const at::Tensor &out,   // total_q x num_heads x head_size
//                const at::Tensor &softmax_lse,     // b x h x s   softmax logsumexp
//                c10::optional<at::Tensor> &dq_,   // total_q x num_heads x head_size, total_q := \sum_{i=0}^{b} s_i
//                c10::optional<at::Tensor> &dk_,   // total_k x num_heads_k x head_size, total_k := \sum_{i=0}^{b} s_i
//                c10::optional<at::Tensor> &dv_,   // total_k x num_heads_k x head_size, total_k := \sum_{i=0}^{b} s_i
//                const at::Tensor &cu_seqlens_q,  // b+1
//                const at::Tensor &cu_seqlens_k,  // b+1
//                c10::optional<at::Tensor> &alibi_slopes_, // num_heads or b x num_heads
//                const int max_seqlen_q,
//                const int max_seqlen_k,          // max sequence length to choose the kernel
//                const float p_dropout,         // probability to drop
//                const float softmax_scale,
//                const bool zero_tensors,
//                const bool is_causal,
//                int window_size_left,
//                int window_size_right,
//                const bool deterministic,
//                c10::optional<at::Generator> gen_,
//                c10::optional<at::Tensor> &rng_state) {

//     if (is_causal) { window_size_right = 0; }
//     auto dprops = at::cuda::getCurrentDeviceProperties();
//     // bool is_sm75 = dprops->major == 7 && dprops->minor == 5;
//     bool is_sm8x = dprops->major == 8 && dprops->minor >= 0;
//     bool is_sm80 = dprops->major == 8 && dprops->minor == 0;
//     bool is_sm90 = dprops->major == 9 && dprops->minor == 0;
//     TORCH_CHECK(is_sm90 || is_sm8x, "FlashAttention only supports Ampere GPUs or newer.");
//     // We will support Turing in the near future
//     // TORCH_CHECK(is_sm90 || is_sm8x || is_sm75, "FlashAttention only supports Turing GPUs or newer.");
//     bool is_dropout = p_dropout > 0.0;
//     auto stream = at::cuda::getCurrentCUDAStream().stream();

//     auto q_dtype = q.dtype();
//     TORCH_CHECK(q_dtype == torch::kFloat16 || q_dtype == torch::kBFloat16,
//                 "FlashAttention only support fp16 and bf16 data type");
//     if (q_dtype == torch::kBFloat16) {
//         TORCH_CHECK(is_sm90 || is_sm8x, "bfloat16 is only supported on Ampere GPUs or newer");
//     }
//     TORCH_CHECK(k.dtype() == q_dtype, "query and key must have the same dtype");
//     TORCH_CHECK(v.dtype() == q_dtype, "query and value must have the same dtype");
//     TORCH_CHECK(out.dtype() == q_dtype, "query and out must have the same dtype");
//     TORCH_CHECK(dout.dtype() == q_dtype, "query and dout must have the same dtype");
//     TORCH_CHECK(cu_seqlens_q.dtype() == torch::kInt32, "cu_seqlens_q must have dtype int32");
//     TORCH_CHECK(cu_seqlens_k.dtype() == torch::kInt32, "cu_seqlens_k must have dtype int32");

//     CHECK_DEVICE(q); CHECK_DEVICE(k); CHECK_DEVICE(v);
//     CHECK_DEVICE(out); CHECK_DEVICE(dout); CHECK_DEVICE(softmax_lse);
//     CHECK_DEVICE(cu_seqlens_q); CHECK_DEVICE(cu_seqlens_k);

//     TORCH_CHECK(q.stride(-1) == 1, "Input tensor must have contiguous last dimension");
//     TORCH_CHECK(k.stride(-1) == 1, "Input tensor must have contiguous last dimension");
//     TORCH_CHECK(v.stride(-1) == 1, "Input tensor must have contiguous last dimension");
//     TORCH_CHECK(out.stride(-1) == 1, "out tensor must have contiguous last dimension");
//     TORCH_CHECK(dout.stride(-1) == 1, "dout tensor must have contiguous last dimension");
//     CHECK_CONTIGUOUS(cu_seqlens_q);
//     CHECK_CONTIGUOUS(cu_seqlens_k);

//     const auto sizes = q.sizes();

//     const int total_q = sizes[0];
//     const int batch_size = cu_seqlens_q.numel() - 1;
//     const int num_heads = sizes[1];
//     const int head_size_og = dout.size(2);
//     const int head_size = sizes[2];
//     const int total_k = k.size(0);
//     const int num_heads_k = k.size(1);
//     TORCH_CHECK(batch_size > 0, "batch size must be positive");
//     TORCH_CHECK(head_size % 8 == 0, "head_size should be a multiple of 8");
//     TORCH_CHECK(head_size <= 256, "FlashAttention backward only supports head dimension at most 256");
//     if (head_size > 192) {
//         TORCH_CHECK(is_sm80 || is_sm90, "FlashAttention backward for head dim > 192 requires A100/A800 or H100/H800");
//     }
//     TORCH_CHECK(num_heads % num_heads_k == 0, "Number of heads in key/value must divide number of heads in query");

//     auto round_multiple = [](int x, int m) { return (x + m - 1) / m * m; };
//     const int head_size_rounded = round_multiple(head_size, 32);
//     const int seqlen_q_rounded = round_multiple(max_seqlen_q, 128);
//     const int seqlen_k_rounded = round_multiple(max_seqlen_k, 128);

//     TORCH_CHECK(head_size == round_multiple(head_size_og, 8), "head_size must be head_size_og rounded to a multiple of 8");

//     if (window_size_left >= max_seqlen_k) { window_size_left = -1; }
//     if (window_size_right >= max_seqlen_k) { window_size_right = -1; }

//     CHECK_SHAPE(q, total_q, num_heads, head_size);
//     CHECK_SHAPE(k, total_k, num_heads_k, head_size);
//     CHECK_SHAPE(v, total_k, num_heads_k, head_size);
//     CHECK_SHAPE(out, total_q, num_heads, head_size);
//     CHECK_SHAPE(dout, total_q, num_heads, head_size_og);
//     CHECK_SHAPE(cu_seqlens_q, batch_size + 1);
//     CHECK_SHAPE(cu_seqlens_k, batch_size + 1);

//     at::Tensor dq, dk, dv;
//     if (dq_.has_value()) {
//         dq = dq_.value();
//         TORCH_CHECK(dq.dtype() == q_dtype, "dq must have the same dtype as q");
//         CHECK_DEVICE(dq);
//         TORCH_CHECK(dq.stride(-1) == 1, "dq must have contiguous last dimension");
//         CHECK_SHAPE(dq, total_q, num_heads, head_size);
//     } else {
//         dq = torch::empty_like(q);
//     }
//     if (dk_.has_value()) {
//         dk = dk_.value();
//         TORCH_CHECK(dk.dtype() == q_dtype, "dk must have the same dtype as q");
//         CHECK_DEVICE(dk);
//         TORCH_CHECK(dk.stride(-1) == 1, "dk must have contiguous last dimension");
//         CHECK_SHAPE(dk, total_k, num_heads_k, head_size);
//     } else {
//         dk = torch::empty_like(k);
//     }
//     if (dv_.has_value()) {
//         dv = dv_.value();
//         TORCH_CHECK(dv.dtype() == q_dtype, "dv must have the same dtype as q");
//         CHECK_DEVICE(dv);
//         TORCH_CHECK(dv.stride(-1) == 1, "dv must have contiguous last dimension");
//         CHECK_SHAPE(dv, total_k, num_heads_k, head_size);
//     } else {
//         dv = torch::empty_like(k);
//     }

//     at::Tensor dout_padded;
//     if (head_size_og % 8 != 0) {
//         dout_padded = torch::nn::functional::pad(dout, torch::nn::functional::PadFuncOptions({0, 8 - head_size_og % 8}));
//     } else {
//         dout_padded = dout;
//     }

//     // bool loop = max_seqlen_k > blocksize_c;
//     // TODO: change later, for now set to true for simplicity
//     bool loop = true;

//     // Otherwise the kernel will be launched from cuda:0 device
//     // Cast to char to avoid compiler warning about narrowing
//     at::cuda::CUDAGuard device_guard{(char)q.get_device()};

//     auto opts = q.options();
//     auto softmax_d = torch::empty({batch_size, num_heads, seqlen_q_rounded}, opts.dtype(at::kFloat));
//     at::Tensor dq_accum;
//     if (loop) {
//         // We don't want to allocate dq_accum of size (batch, seqlen_q_rounded, num_heads, head_size_rounded)
//         // because that would be too large if there is a very long sequence and the rest of the sequences are short.
//         // Instead, we allocate dq_accum of size (total_q + 128 * batch, num_heads, head_size_rounded).
//         // Note that 128 is the max block size on the seqlen_q dimension.
//         // For dQ, the i-th sequence is stored in indices from cu_seqlens[i] + 128 * i to
//         // cu_seqlens[i + 1] * 128 * i - 1. This ensures that the i-th sequence and (i + 1)-th sequence will
//         // be at least 128 apart. It's ok for us to do atomicAdds up to 128 rows beyond what we're normally
//         // allowed to do. So we won't have to do any bound checking, and performance should stay the same.
//         if (!deterministic) {
//             dq_accum = torch::empty({total_q + 128 * batch_size, num_heads, head_size_rounded}, opts.dtype(at::kFloat));
//         } else {
//             const int nsplits = (dprops->multiProcessorCount + batch_size * num_heads - 1) / (batch_size * num_heads);
//             dq_accum = torch::zeros({nsplits, total_q + 128 * batch_size, num_heads, head_size_rounded}, opts.dtype(at::kFloat));
//         }
//     }

//     at::Tensor dk_expanded, dv_expanded;
//     if (num_heads_k != num_heads) {  // MQA / GQA
//         dk_expanded = torch::empty({total_k, num_heads, head_size}, opts);
//         dv_expanded = torch::empty({total_k, num_heads, head_size}, opts);
//     } else {
//         dk_expanded = dk;
//         dv_expanded = dv;
//     }

//     if( zero_tensors ) {
//         dq.zero_();
//         dk_expanded.zero_();
//         dv_expanded.zero_();
//         softmax_d.zero_();
//     }

//     Flash_bwd_params params;

//     set_params_dgrad(params,
//                      batch_size,
//                      max_seqlen_q, max_seqlen_k,
//                      seqlen_q_rounded, seqlen_k_rounded,
//                      num_heads, num_heads_k,
//                      head_size, head_size_rounded,
//                      q, k, v, out,
//                      dout_padded, dq, dk_expanded, dv_expanded,
//                      cu_seqlens_q.data_ptr(),
//                      cu_seqlens_k.data_ptr(),
//                      loop ? dq_accum.data_ptr() : nullptr,
//                      nullptr,
//                      nullptr,
//                      softmax_lse.data_ptr(),
//                      softmax_d.data_ptr(),
//                      p_dropout,
//                      softmax_scale,
//                      window_size_left,
//                      window_size_right,
//                      deterministic);
//     params.dq_accum_split_stride = !deterministic ? 0 : dq_accum.stride(0);

//     auto launch = &run_mha_bwd;

//     auto gen = at::get_generator_or_default<at::CUDAGeneratorImpl>(
//         gen_, at::cuda::detail::getDefaultCUDAGenerator());

//     // We use a custom RNG that increases the offset by batch_size * nheads * 32.
//     int64_t counter_offset = params.b * params.h * 32;

//     if ( rng_state.has_value() ) {
//         params.rng_state = reinterpret_cast<uint64_t*>(rng_state.value().data_ptr());
//     } else if( is_dropout ) {
//         // See Note [Acquire lock when using random generators]
//         std::lock_guard<std::mutex> lock(gen->mutex_);
//         params.philox_args = gen->philox_cuda_state(counter_offset);
//         auto seeds = at::cuda::philox::unpack(params.philox_args);
//         params.rng_state[0] = std::get<0>(seeds);
//         params.rng_state[1] = std::get<1>(seeds);
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

//     if (max_seqlen_q > 0) {
//         launch(params, stream);
//     } else {
//         // If seqlen_q == 0, then we have an empty tensor. We need to set the output to 0.
//         dk_expanded.zero_();
//         dv_expanded.zero_();
//         softmax_d.zero_();
//     }

//     // For MQA/GQA we need to sum dK and dV across the groups
//     if (num_heads_k != num_heads) {
//         at::sum_out(dk, at::reshape(dk_expanded, {total_k, num_heads_k, num_heads / num_heads_k, head_size}), {2});
//         at::sum_out(dv, at::reshape(dv_expanded, {total_k, num_heads_k, num_heads / num_heads_k, head_size}), {2});
//     }
//     if (head_size_og % 8 != 0) {
//         dq = dq.index({"...", torch::indexing::Slice(torch::indexing::None, head_size_og)});
//         dk = dk.index({"...", torch::indexing::Slice(torch::indexing::None, head_size_og)});
//         dv = dv.index({"...", torch::indexing::Slice(torch::indexing::None, head_size_og)});
//     }

//     return { dq, dk, dv, softmax_d };
// }
