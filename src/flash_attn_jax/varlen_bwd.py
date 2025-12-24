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

from flash_attn_jax.util import round_multiple

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
def flash_mha_varlen_bwd(dout, q, k, v, o, lse, seqlens_q, seqlens_k, *,
                         max_seqlen_q: int = -1, max_seqlen_k: int = -1,
                         softmax_scale: Optional[float] = None, is_causal: bool = False,
                         window_size_left: int = -1, window_size_right: int = -1):
    if max_seqlen_q  == -1:
        max_seqlen_q = q.shape[0]
    if max_seqlen_k == -1:
        max_seqlen_k = k.shape[0]
    assert seqlens_q.shape == seqlens_k.shape, "seqlens_q and seqlens_k must have the same shape."
    kwargs = dict(
        max_seqlen_q=max_seqlen_q,
        max_seqlen_k=max_seqlen_k,
        softmax_scale=softmax_scale,
        is_causal=is_causal,
        window_size_left=window_size_left,
        window_size_right=window_size_right,
    )
    return tuple(_flash_mha_varlen_bwd_p.bind(dout, q, k, v, o, lse, seqlens_q, seqlens_k, **kwargs))

# ==== HLO lowering ====

def _flash_mha_varlen_bwd_hlo_lowering(dout, q, k, v, o, lse, seqlens_q, seqlens_k, *,
                                       max_seqlen_q: int, max_seqlen_k: int,
                                       softmax_scale: float, is_causal: bool, window_size_left: int, window_size_right: int):
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

    if softmax_scale is None:
        softmax_scale = 1.0 / math.sqrt(d)

    dpad = (8 - (d % 8)) % 8
    if dpad > 0:
        q = jnp.pad(q, ((0, 0), (0, 0), (0, dpad)), mode='constant', constant_values=0)
        k = jnp.pad(k, ((0, 0), (0, 0), (0, dpad)), mode='constant', constant_values=0)
        v = jnp.pad(v, ((0, 0), (0, 0), (0, dpad)), mode='constant', constant_values=0)
        dout = jnp.pad(dout, ((0, 0), (0, 0), (0, dpad)), mode='constant', constant_values=0)
        o = jnp.pad(o, ((0, 0), (0, 0), (0, dpad)), mode='constant', constant_values=0)

    dq_shape = [totalq, h, d+dpad]
    dk_shape = [totalk, h, d+dpad]
    dv_shape = [totalk, h, d+dpad]

    # Calculate scratch array shapes
    seqlen_q_rounded = round_multiple(max_seqlen_q, 128)
    d_rounded = round_multiple(d+dpad, 32)
    softmax_d_shape = (b, h, seqlen_q_rounded)
    
    # Calculate nsplits for deterministic mode (varlen-specific)
    sm_count = 114  # H100, should ideally get this from device query
    if False: # deterministic mode
        nsplits = max(1, (sm_count + b * h - 1) // (b * h))
        dq_accum_shape = (nsplits, totalq + 128 * b, h, d_rounded)  # varlen-specific sizing with splits
    else:
        dq_accum_shape = (totalq + 128 * b, h, d_rounded)  # varlen-specific sizing
    
    out_types = [jax.ShapeDtypeStruct(dq_shape, q_dtype),              # dq
                    jax.ShapeDtypeStruct(dk_shape, k_dtype),           # dk
                    jax.ShapeDtypeStruct(dv_shape, v_dtype),           # dv
                    jax.ShapeDtypeStruct(softmax_d_shape, jnp.float32), # softmax_d
                    jax.ShapeDtypeStruct(dq_accum_shape, jnp.float32)]  # dq_accum

    kwargs = dict(
        max_seqlen_q=mlir.i64_attr(max_seqlen_q),
        max_seqlen_k=mlir.i64_attr(max_seqlen_k),
        softmax_scale=mlir.ir.FloatAttr.get_f32(softmax_scale),
        zero_tensors=False,
        is_causal=is_causal,
        window_size_left=mlir.i64_attr(window_size_left),
        window_size_right=mlir.i64_attr(window_size_right),
        deterministic=False,
    )
    dq, dk, dv = jax.ffi.ffi_call(
        "flash_mha_varlen_bwd", 
        result_shape_dtypes=out_types,
        has_side_effect=False,
        input_layouts=[None]*8, # default row major
        output_layouts=[None]*5,
        )(dout, q, k, v, o, lse, seqlens_q, seqlens_k, **kwargs)[:3]  # Only return first 3 outputs (dq, dk, dv)
    
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

def _flash_mha_varlen_bwd_hlo_lowering_mlir(ctx, *args, **keywords):
    return mlir.lower_fun(_flash_mha_varlen_bwd_hlo_lowering, multiple_results=True)(ctx, *args, **keywords)

mlir.register_lowering(
    _flash_mha_varlen_bwd_p,
    _flash_mha_varlen_bwd_hlo_lowering_mlir,  # type: ignore
    platform="gpu",
)

# ==== Abstract Evaluation ====

def _flash_mha_varlen_bwd_abstract(dout, q, k, v, o, lse, seqlens_q, seqlens_k,
                                   max_seqlen_q: int, max_seqlen_k: int,
                                   softmax_scale: float,
                                   is_causal: bool, window_size_left: int, window_size_right: int):
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

# ==== VMap rules ====

def _flash_mha_varlen_bwd_batch(vector_arg_values, batch_axes, *, max_seqlen_q: int, max_seqlen_k: int, **kwargs):
    dout, q, k, v, o, lse, seqlens_q, seqlens_k = vector_arg_values
    assert all(isinstance(b, int) or b is None for b in batch_axes)
    mapped = tuple(isinstance(b, int) for b in batch_axes)
    if mapped == (True,)*8:
        assert all(b == 0 or b is None for b in batch_axes), "Batch axis must be at front"
        b, sq, hq, cq = q.shape
        b, sk, hk, ck = k.shape
        assert cq == ck
        assert k.shape == v.shape
        assert seqlens_q.shape == seqlens_k.shape
        assert dout.shape == q.shape == o.shape
        b, n_plus_1 = seqlens_q.shape
        b, n, hlse, slse = lse.shape
        assert hlse == hq and slse == max_seqlen_q
        new_q = q.reshape((b*sq, hq, cq))
        new_k = k.reshape((b*sk, hk, ck))
        new_v = v.reshape((b*sk, hk, ck))
        new_o = o.reshape((b*sq, hq, cq))
        new_dout = dout.reshape((b*sq, hq, cq))
        new_seqlens_q = (seqlens_q + (jnp.arange(b)[:,None]*sq)).reshape((b*n_plus_1,))
        new_seqlens_k = (seqlens_k + (jnp.arange(b)[:,None]*sk)).reshape((b*n_plus_1,)) # now we have b*(n+1)-1 sequences. we added b-1 so b*n + b-1
        new_lse = jnp.pad(lse, ((0,0),(0,1),(0,0),(0,0))) # b n_plus_1 h max_seqlen_q
        new_lse = new_lse.reshape((b*n_plus_1, hlse, slse))[:-1]
        dq, dk, dv = flash_mha_varlen_bwd(new_dout, new_q, new_k, new_v, new_o, new_lse, new_seqlens_q, new_seqlens_k,
                                        max_seqlen_q=max_seqlen_q, max_seqlen_k=max_seqlen_k, **kwargs)
        dq = dq.reshape((b, sq, hq, cq))
        dk = dk.reshape((b, sk, hk, ck))
        dv = dv.reshape((b, sk, hk, ck))
        return (dq, dk, dv), (0,0,0)
    elif mapped == (1,0,1,1,1,1,0,1):
        # Everything mapped except q and seqlens_q, broadcast them.
        assert all(b == 0 or b is None for b in batch_axes), "Batch axis must be at front"
        x = dout.shape[0]
        q = einops.repeat(q, "... -> x ...", x=x)
        seqlens_q = einops.repeat(seqlens_q, "... -> x ...", x=x)
        dq, dk, dv = jax.vmap(
            partial(
                flash_mha_varlen_bwd,
                max_seqlen_q=max_seqlen_q,
                max_seqlen_k=max_seqlen_k,
                **kwargs,
            )
        )(dout, q, k, v, o, lse, seqlens_q, seqlens_k)
        return (dq, dk, dv), (0, 0, 0)

    else:
        raise NotImplementedError("flash_mha_varlen_bwd: vmap only for all inputs: " + str(mapped))
batching.primitive_batchers[_flash_mha_varlen_bwd_p] = _flash_mha_varlen_bwd_batch