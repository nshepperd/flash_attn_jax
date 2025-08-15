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

_flash_mha_varlen_fwd_p = Primitive("flash_mha_varlen_fwd")
_flash_mha_varlen_fwd_p.multiple_results = True
_flash_mha_varlen_fwd_p.def_impl(partial(xla.apply_primitive, _flash_mha_varlen_fwd_p))
jax._src.dispatch.prim_requires_devices_during_lowering.add(_flash_mha_varlen_fwd_p)

# ==== Frontend ====

def flash_mha_varlen_fwd(q, k, v, seqlens_q, seqlens_k, seqused_k=None,
                         max_seqlen_q: int = -1, max_seqlen_k: int = -1,
                         softmax_scale: Optional[float] = None, is_causal: bool = False,
                         window_size: tuple = (-1, -1),
                         zero_tensors: bool = False, deterministic: bool = False):
    if max_seqlen_q  == -1:
        max_seqlen_q = q.shape[0]
    if max_seqlen_k == -1:
        max_seqlen_k = k.shape[0]
    assert seqlens_q.shape == seqlens_k.shape, "seqlens_q and seqlens_k must have the same shape."
    d = q.shape[-1]
    if softmax_scale is None:
        softmax_scale = 1.0 / math.sqrt(d)
    has_seqused_k = seqused_k is not None
    if seqused_k is None:
        seqused_k = jnp.empty([], dtype=jnp.int32)
    kwargs = dict(
        max_seqlen_q=max_seqlen_q,
        max_seqlen_k=max_seqlen_k,
        has_seqused_k=has_seqused_k,
        softmax_scale=softmax_scale,
        is_causal=is_causal,
        window_size_left=window_size[0],
        window_size_right=window_size[1],
    )
    return tuple(_flash_mha_varlen_fwd_p.bind(q, k, v, seqlens_q, seqlens_k, seqused_k, **kwargs))

# ==== HLO lowering ====

def _flash_mha_varlen_fwd_hlo_lowering(ctx, q, k, v, seqlens_q, seqlens_k, seqused_k, 
                                       max_seqlen_q: int, max_seqlen_k: int, has_seqused_k: bool,
                                       softmax_scale: float, is_causal: bool, window_size_left: int, window_size_right: int):
    def fwd(q,k,v, seqlens_q, seqlens_k, seqused_k):
        q_dtype = dtypes.canonicalize_dtype(q.dtype)
        k_dtype = dtypes.canonicalize_dtype(k.dtype)
        v_dtype = dtypes.canonicalize_dtype(v.dtype)
        [totalq, h, d] = q.shape
        b = seqlens_q.shape[0] - 1
        assert q_dtype == k_dtype and q_dtype == v_dtype
        assert q_dtype in [jnp.bfloat16, jnp.float16]
        assert b >= 1

        dpad = 8 - (d % 8)
        if dpad > 0:
            q = jnp.pad(q, ((0, 0), (0, 0), (0, dpad)), mode='constant', constant_values=0)
            k = jnp.pad(k, ((0, 0), (0, 0), (0, dpad)), mode='constant', constant_values=0)
            v = jnp.pad(v, ((0, 0), (0, 0), (0, dpad)), mode='constant', constant_values=0)

        out_shape = [totalq, h, d+dpad]
        lse_shape = [b, h, max_seqlen_q]

        out_types = [jax.ShapeDtypeStruct(out_shape, q_dtype), 
                     jax.ShapeDtypeStruct(lse_shape, jnp.float32)]

        out, lse = jax.ffi.ffi_call(
            "flash_mha_varlen_fwd", 
            result_shape_dtypes=out_types,
            has_side_effect=False,
            input_layouts=[None]*6, # default row major
            output_layouts=[None]*2,
            )(q, k, v, seqlens_q, seqlens_k, seqused_k,
            max_seqlen_q=mlir.i32_attr(max_seqlen_q),
            max_seqlen_k=mlir.i32_attr(max_seqlen_k),
            has_seqused_k=has_seqused_k,
            softmax_scale=softmax_scale,
            zero_tensors=False,
            is_causal=is_causal,
            window_size_left=window_size_left,
            window_size_right=window_size_right)
        
        if dpad > 0:
            out = out[:,:,:d]

        return out, lse
    return mlir.lower_fun(fwd, multiple_results=True)(ctx, q, k, v, seqlens_q, seqlens_k, seqused_k)

mlir.register_lowering(
    _flash_mha_varlen_fwd_p,
    _flash_mha_varlen_fwd_hlo_lowering,  # type: ignore
    platform="gpu",
)

# ==== Abstract Evaluation ====

def _flash_mha_varlen_fwd_abstract(q, k, v, seqlens_q, seqlens_k, seqused_k, 
                                   max_seqlen_q, max_seqlen_k, has_seqused_k, 
                                   softmax_scale=None, is_causal=None, window_size_left=None, window_size_right=None):
    q_dtype = dtypes.canonicalize_dtype(q.dtype)
    k_dtype = dtypes.canonicalize_dtype(k.dtype)
    v_dtype = dtypes.canonicalize_dtype(v.dtype)
    [totalq, h, d] = q.shape
    b = seqlens_q.shape[0] - 1
    assert q_dtype == k_dtype and q_dtype == v_dtype
    assert q_dtype in [jnp.bfloat16, jnp.float16]
    assert b >= 1

    out_shape = [totalq, h, d]
    lse_shape = [b, h, max_seqlen_q]
    
    return (
        ShapedArray(out_shape, q_dtype),
        ShapedArray(lse_shape, jnp.float32)
    )
_flash_mha_varlen_fwd_p.def_abstract_eval(_flash_mha_varlen_fwd_abstract)
