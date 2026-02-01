/******************************************************************************
 * Copyright (c) 2024, Tri Dao.
 ******************************************************************************/

#include <stddef.h>
#include <cutlass/numeric_types.h>
#include <cuda_runtime_api.h>
#include <nanobind/nanobind.h>

#include "check.h"

#include "mha_fwd.h"
#include "mha_bwd.h"
#include "xla/ffi/api/c_api.h"
#include "xla/ffi/api/ffi.h"

namespace ffi = xla::ffi;

namespace {

namespace nb = nanobind;

template <typename T>
nb::capsule EncapsulateFfiCall(T *fn) {
  static_assert(std::is_invocable_r_v<XLA_FFI_Error *, T, XLA_FFI_CallFrame *>,
                "Encapsulated function must be an XLA FFI handler");
  return nb::capsule(reinterpret_cast<void *>(fn));
}

XLA_FFI_DEFINE_HANDLER(
	mha_fwd, mha_fwd_impl,
	ffi::Ffi::Bind()
        .Ctx<ffi::PlatformStream<cudaStream_t>>()
		.Ctx<ffi::DeviceOrdinal>()
		.Arg<ffi::AnyBuffer>()
		.Arg<ffi::AnyBuffer>()
		.Arg<ffi::AnyBuffer>()
		.Ret<ffi::AnyBuffer>()
		.Ret<ffi::Buffer<ffi::F32>>()
		.Ret<ffi::Buffer<ffi::F32>>()
		.Ret<ffi::Buffer<ffi::F32>>()
		.Attr<double>("softmax_scale")
		.Attr<bool>("is_causal")
		.Attr<int64_t>("window_size_left")
		.Attr<int64_t>("window_size_right")
);

XLA_FFI_DEFINE_HANDLER(
	mha_bwd, mha_bwd_impl,
	ffi::Ffi::Bind()
		.Ctx<ffi::PlatformStream<cudaStream_t>>()
		.Ctx<ffi::DeviceOrdinal>()
		.Arg<ffi::AnyBuffer>() // dout
		.Arg<ffi::AnyBuffer>() // q
		.Arg<ffi::AnyBuffer>() // k
		.Arg<ffi::AnyBuffer>() // v
		.Arg<ffi::AnyBuffer>() // o
		.Arg<ffi::Buffer<ffi::F32>>() // lse
		.Ret<ffi::AnyBuffer>() // dq
		.Ret<ffi::AnyBuffer>() // dk
		.Ret<ffi::AnyBuffer>() // dv
		.Ret<ffi::Buffer<ffi::F32>>() // softmax_d
		.Ret<ffi::Buffer<ffi::F32>>() // dq_accum
		.Attr<double>("softmax_scale")
		.Attr<bool>("is_causal")
		.Attr<int64_t>("window_size_left")
		.Attr<int64_t>("window_size_right")
		.Attr<bool>("deterministic")
);

XLA_FFI_DEFINE_HANDLER(
	mha_varlen_fwd, mha_varlen_fwd_impl,
	ffi::Ffi::Bind()
		.Ctx<ffi::PlatformStream<cudaStream_t>>()
		.Ctx<ffi::DeviceOrdinal>()
		.Arg<ffi::AnyBuffer>() // q
		.Arg<ffi::AnyBuffer>() // k
		.Arg<ffi::AnyBuffer>() // v
		.Arg<ffi::Buffer<ffi::S32>>() // cu_seqlens_q
		.Arg<ffi::Buffer<ffi::S32>>() // cu_seqlens_k
		.OptionalArg<ffi::Buffer<ffi::S32>>() // seqused_k
		.Ret<ffi::AnyBuffer>() // o
		.Ret<ffi::Buffer<ffi::F32>>() // lse
		.Ret<ffi::Buffer<ffi::F32>>()
		.Ret<ffi::Buffer<ffi::F32>>()
		.Attr<int>("max_seqlen_q")
		.Attr<int>("max_seqlen_k")
		.Attr<double>("softmax_scale")
		.Attr<bool>("zero_tensors")
		.Attr<bool>("is_causal")
		.Attr<int64_t>("window_size_left")
		.Attr<int64_t>("window_size_right")
);

XLA_FFI_DEFINE_HANDLER(
	mha_varlen_bwd, mha_varlen_bwd_impl,
	ffi::Ffi::Bind()
		.Ctx<ffi::PlatformStream<cudaStream_t>>()
		.Ctx<ffi::DeviceOrdinal>()
		.Arg<ffi::AnyBuffer>() // dout
		.Arg<ffi::AnyBuffer>() // q
		.Arg<ffi::AnyBuffer>() // k
		.Arg<ffi::AnyBuffer>() // v
		.Arg<ffi::AnyBuffer>() // o
		.Arg<ffi::Buffer<ffi::F32>>() // lse
		.Arg<ffi::Buffer<ffi::S32>>() // cu_seqlens_q
		.Arg<ffi::Buffer<ffi::S32>>() // cu_seqlens_k
		.Ret<ffi::AnyBuffer>() // dq
		.Ret<ffi::AnyBuffer>() // dk
		.Ret<ffi::AnyBuffer>() // dv
		.Ret<ffi::Buffer<ffi::F32>>() // softmax_d
		.Ret<ffi::Buffer<ffi::F32>>() // dq_accum
		.Attr<int64_t>("max_seqlen_q")
		.Attr<int64_t>("max_seqlen_k")
		.Attr<float>("softmax_scale")
		.Attr<bool>("zero_tensors")
		.Attr<bool>("is_causal")
		.Attr<int64_t>("window_size_left")
		.Attr<int64_t>("window_size_right")
		.Attr<bool>("deterministic")
);

nb::dict FFIRegistrations() {
  nb::dict dict;
  dict["flash_mha_fwd"] = EncapsulateFfiCall(mha_fwd);
  dict["flash_mha_bwd"] = EncapsulateFfiCall(mha_bwd);
  dict["flash_mha_varlen_fwd"] = EncapsulateFfiCall(mha_varlen_fwd);
  dict["flash_mha_varlen_bwd"] = EncapsulateFfiCall(mha_varlen_bwd);
  return dict;
}


NB_MODULE(flash_api, m) {
    m.doc() = "FlashAttention";
	m.def("get_ffi_registrations", &FFIRegistrations);

    // m.def("varlen_fwd", &mha_varlen_fwd, "Forward pass (variable length)");
    // m.def("bwd", &mha_bwd, "Backward pass");
    // m.def("varlen_bwd", &mha_varlen_bwd, "Backward pass (variable length)");
    // m.def("fwd_kvcache", &mha_fwd_kvcache, "Forward pass, with KV-cache");
}

} // namespace