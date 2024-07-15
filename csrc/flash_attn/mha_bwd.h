#pragma once

#include <stddef.h>
#include <cutlass/numeric_types.h>
#include <cuda_runtime_api.h>
#include <pybind11/pybind11.h>

#include "flash_common.h"

struct mha_bwd_args {
	float p_dropout;
	float softmax_scale;
	bool is_causal;
	int window_size_left;
	int window_size_right;
	bool deterministic;
	int n, l, h, d;
	int l_k, h_k;

	ElementType dtype;
	uint64_t seed;
	SimilarityType similarity_type;
};

void mha_bwd(cudaStream_t stream, void **buffers, const char* opaque, size_t opaque_len);
