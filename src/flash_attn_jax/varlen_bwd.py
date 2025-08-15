from functools import partial, wraps
from typing import Optional

import numpy as np
import jax
import jax.numpy as jnp
from jax import core, dtypes
from jax.core import ShapedArray
from jax.interpreters import batching
from jax.interpreters import mlir
from jax.interpreters import xla
from jax.extend.core import Primitive
import jax._src.dispatch

from einops import rearrange
import einops
import math

# ==== Register primitives ====

_flash_mha_varlen_bwd_p = Primitive("flash_mha_varlen_bwd")
_flash_mha_varlen_bwd_p.multiple_results = True
_flash_mha_varlen_bwd_p.def_impl(partial(xla.apply_primitive, _flash_mha_varlen_bwd_p))
jax._src.dispatch.prim_requires_devices_during_lowering.add(_flash_mha_varlen_bwd_p)

# ==== Frontend ====

    # ffi::AnyBuffer dout,  // total_q x num_heads, x head_size
    # ffi::AnyBuffer q,     // total_q x num_heads x head_size, total_q := \sum_{i=0}^{b} s_i
    # ffi::AnyBuffer k,     // total_k x num_heads_k x head_size, total_k := \sum_{i=0}^{b} s_i
    # ffi::AnyBuffer v,     // total_k x num_heads_k x head_size, total_k := \sum_{i=0}^{b} s_i
    # ffi::AnyBuffer o,     // total_q x num_heads x head_size,
    # ffi::Buffer<ffi::F32> lse, // b x h x s   softmax logsumexp
    # ffi::Buffer<ffi::S32> cu_seqlens_q,  // b+1
    # ffi::Buffer<ffi::S32> cu_seqlens_k,  // b+1
    # ffi::Result<ffi::AnyBuffer> dq,   // total_q x num_heads x head_size, total_q := \sum_{i=0}^{b} s_i
    # ffi::Result<ffi::AnyBuffer> dk,   // total_k x num_heads_k x head_size, total_k := \sum_{i=0}^{b} s_i
    # ffi::Result<ffi::AnyBuffer> dv,   // total_k x num_heads_k x head_size, total_k := \sum_{i=0}^{b} s_i
    # const int max_seqlen_q,
    # const int max_seqlen_k,          // max sequence length to choose the kernel
    # const float softmax_scale,
    # const bool zero_tensors,
    # const bool is_causal,
    # int window_size_left,
    # int window_size_right,
    # const bool deterministic)
def flash_mha_varlen_bwd(dout, q, k, v, o, lse, seqlens_q, seqlens_k,
                         max_seqlen_q: int = -1, max_seqlen_k: int = -1,
                         softmax_scale: Optional[float] = None, zero_tensors=False, is_causal: bool = False,
                         window_size: tuple = (-1, -1), deterministic: bool = False):
    if max_seqlen_q  == -1:
        max_seqlen_q = q.shape[0]
    if max_seqlen_k == -1:
        max_seqlen_k = k.shape[0]
    assert seqlens_q.shape == seqlens_k.shape, "seqlens_q and seqlens_k must have the same shape."
    d = q.shape[-1]
    if softmax_scale is None:
        softmax_scale = 1.0 / math.sqrt(d)
    kwargs = dict(
        max_seqlen_q=max_seqlen_q,
        max_seqlen_k=max_seqlen_k,
        softmax_scale=softmax_scale,
        zero_tensors=zero_tensors,
        is_causal=is_causal,
        window_size_left=window_size[0],
        window_size_right=window_size[1],
        deterministic=deterministic,
    )
    return tuple(_flash_mha_varlen_bwd_p.bind(dout, q, k, v, o, lse, seqlens_q, seqlens_k, **kwargs))

# ==== HLO lowering ====

def _flash_mha_varlen_bwd_hlo_lowering(ctx, dout, q, k, v, o, lse, seqlens_q, seqlens_k,
                                       max_seqlen_q: int, max_seqlen_k: int,
                                       softmax_scale: float, zero_tensors: bool,
                                       is_causal: bool, window_size_left: int, window_size_right: int,
                                       deterministic: bool):
    def bwd(dout, q, k, v, o, lse, seqlens_q, seqlens_k):
        q_dtype = dtypes.canonicalize_dtype(q.dtype)
        k_dtype = dtypes.canonicalize_dtype(k.dtype)
        v_dtype = dtypes.canonicalize_dtype(v.dtype)
        [totalq, h, d] = q.shape
        [totalk, hk, dk] = k.shape
        b = seqlens_q.shape[0] - 1
        assert q_dtype == k_dtype and q_dtype == v_dtype
        assert q_dtype in [jnp.bfloat16, jnp.float16]
        assert b >= 1
        assert d == dk, "q and k must have the same head size."

        dpad = 8 - (d % 8)
        if dpad > 0:
            q = jnp.pad(q, ((0, 0), (0, 0), (0, dpad)), mode='constant', constant_values=0)
            k = jnp.pad(k, ((0, 0), (0, 0), (0, dpad)), mode='constant', constant_values=0)
            v = jnp.pad(v, ((0, 0), (0, 0), (0, dpad)), mode='constant', constant_values=0)
            dout = jnp.pad(dout, ((0, 0), (0, 0), (0, dpad)), mode='constant', constant_values=0)
            o = jnp.pad(o, ((0, 0), (0, 0), (0, dpad)), mode='constant', constant_values=0)

        dq_shape = [totalq, h, d+dpad]
        dk_shape = [totalk, h, d+dpad]
        dv_shape = [totalk, h, d+dpad]

        out_types = [jax.ShapeDtypeStruct(dq_shape, q_dtype), 
                        jax.ShapeDtypeStruct(dk_shape, k_dtype),
                        jax.ShapeDtypeStruct(dv_shape, v_dtype)]

        kwargs = dict(
            max_seqlen_q=mlir.i64_attr(max_seqlen_q),
            max_seqlen_k=mlir.i64_attr(max_seqlen_k),
            softmax_scale=mlir.ir.FloatAttr.get_f32(softmax_scale),
            zero_tensors=zero_tensors,
            is_causal=is_causal,
            window_size_left=mlir.i64_attr(window_size_left),
            window_size_right=mlir.i64_attr(window_size_right),
            deterministic=deterministic,
        )
        dq, dk, dv = jax.ffi.ffi_call(
            "flash_mha_varlen_bwd", 
            result_shape_dtypes=out_types,
            has_side_effect=False,
            input_layouts=[None]*8, # default row major
            output_layouts=[None]*3,
            )(dout, q, k, v, o, lse, seqlens_q, seqlens_k, **kwargs)
        
        if dpad > 0:
            dq = dq[:,:,:d]
            dk = dk[:,:,:d]
            dv = dv[:,:,:d]
        
        if h > hk:
            # MQA
            assert h % hk == 0, "h must be divisible by hk for MQA."
            dk = einops.reduce(dk, "b (hk m) d -> b hk d", hk=hk, reduction="sum")
            dv = einops.reduce(dv, "b (hk m) d -> b hk d", hk=hk, reduction="sum")

        return dq, dk, dv
    return mlir.lower_fun(bwd, multiple_results=True)(ctx, dout, q, k, v, o, lse, seqlens_q, seqlens_k)

mlir.register_lowering(
    _flash_mha_varlen_bwd_p,
    _flash_mha_varlen_bwd_hlo_lowering,  # type: ignore
    platform="gpu",
)

# ==== Abstract Evaluation ====

def _flash_mha_varlen_bwd_abstract(dout, q, k, v, o, lse, seqlens_q, seqlens_k,
                                   max_seqlen_q: int, max_seqlen_k: int,
                                   softmax_scale: float, zero_tensors: bool,
                                   is_causal: bool, window_size_left: int, window_size_right: int,
                                   deterministic: bool):
    q_dtype = dtypes.canonicalize_dtype(q.dtype)
    k_dtype = dtypes.canonicalize_dtype(k.dtype)
    v_dtype = dtypes.canonicalize_dtype(v.dtype)
    [totalq, h, d] = q.shape
    b = seqlens_q.shape[0] - 1
    assert q_dtype == k_dtype and q_dtype == v_dtype
    assert q_dtype in [jnp.bfloat16, jnp.float16]
    assert b >= 1

    dq_shape = q.shape
    dk_shape = k.shape
    dv_shape = v.shape

    return (
        ShapedArray(dq_shape, q_dtype),
        ShapedArray(dk_shape, k_dtype),
        ShapedArray(dv_shape, v_dtype),
    )
_flash_mha_varlen_bwd_p.def_abstract_eval(_flash_mha_varlen_bwd_abstract)