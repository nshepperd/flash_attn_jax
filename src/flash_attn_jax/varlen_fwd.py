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

from flash_attn_jax.util import num_splits_heuristic, round_multiple

# ==== Register primitives ====

# [n, sq, hq, d] [n, sk, sk, d] [n, sk, sk, d] [n, b+1] [n, b+1] [(optional) n, b] -> [n, sq, hq, d] [n, b, hq, max_seqlen_q]
_flash_mha_varlen_fwd_p = Primitive("flash_mha_varlen_fwd")
_flash_mha_varlen_fwd_p.multiple_results = True
_flash_mha_varlen_fwd_p.def_impl(partial(xla.apply_primitive, _flash_mha_varlen_fwd_p))
jax._src.dispatch.prim_requires_devices_during_lowering.add(_flash_mha_varlen_fwd_p)

# ==== Frontend ====

def flash_mha_varlen_fwd(q, k, v, seqlens_q, seqlens_k, seqused_k=None,
                         *,
                         max_seqlen_q: int = -1, max_seqlen_k: int = -1,
                         softmax_scale: Optional[float] = None, is_causal: bool = False,
                         window_size_left: int = -1, window_size_right: int = -1):
    if max_seqlen_q == -1:
        max_seqlen_q = q.shape[0]
    if max_seqlen_k == -1:
        max_seqlen_k = k.shape[0]
    assert seqlens_q.shape == seqlens_k.shape, "seqlens_q and seqlens_k must have the same shape."
    has_seqused_k = seqused_k is not None
    kwargs = dict(
        max_seqlen_q=max_seqlen_q,
        max_seqlen_k=max_seqlen_k,
        has_seqused_k=has_seqused_k,
        softmax_scale=softmax_scale,
        is_causal=is_causal,
        window_size_left=window_size_left,
        window_size_right=window_size_right,
    )
    if seqused_k is not None:
        return tuple(_flash_mha_varlen_fwd_p.bind(q, k, v, seqlens_q, seqlens_k, seqused_k, **kwargs))
    else:
        return tuple(_flash_mha_varlen_fwd_p.bind(q, k, v, seqlens_q, seqlens_k, **kwargs))

# ==== HLO lowering ====

def _flash_mha_varlen_fwd_hlo_lowering(q,k,v, seqlens_q, seqlens_k, seqused_k=None, *,
                                       max_seqlen_q: int, max_seqlen_k: int, has_seqused_k: bool,
                                       softmax_scale: float, is_causal: bool, window_size_left: int, window_size_right: int):
    q_dtype = dtypes.canonicalize_dtype(q.dtype)
    k_dtype = dtypes.canonicalize_dtype(k.dtype)
    v_dtype = dtypes.canonicalize_dtype(v.dtype)
    [totalq, h, d] = q.shape
    b = seqlens_q.shape[0] - 1
    assert q_dtype == k_dtype and q_dtype == v_dtype
    assert q_dtype in [jnp.bfloat16, jnp.float16]
    assert b >= 1

    if softmax_scale is None:
        softmax_scale = 1.0 / math.sqrt(d)

    if d <= 64:
        block_n = 256
    elif d <= 128:
        block_n = 128
    else:
        block_n = 64
    num_n_blocks = max(1, (max_seqlen_k + block_n - 1) // block_n)
    num_m_blocks = max(1, (max_seqlen_q + 64 - 1) // 64)
    sm_count = 114 # H100
    num_splits = num_splits_heuristic(b * h * num_m_blocks, sm_count, num_n_blocks, 128)
    lseaccum_shape = (num_splits, b, h, max_seqlen_q)
    oaccum_shape = (num_splits, b, max_seqlen_q, h, round_multiple(d, 32))

    dpad = (8 - (d % 8)) % 8
    if dpad > 0:
        q = jnp.pad(q, ((0, 0), (0, 0), (0, dpad)), mode='constant', constant_values=0)
        k = jnp.pad(k, ((0, 0), (0, 0), (0, dpad)), mode='constant', constant_values=0)
        v = jnp.pad(v, ((0, 0), (0, 0), (0, dpad)), mode='constant', constant_values=0)

    out_shape = [totalq, h, d+dpad]
    lse_shape = [b, h, max_seqlen_q]

    out_types = [jax.ShapeDtypeStruct(out_shape, q_dtype), 
                    jax.ShapeDtypeStruct(lse_shape, jnp.float32),
                    jax.ShapeDtypeStruct(oaccum_shape, jnp.float32),
                    jax.ShapeDtypeStruct(lseaccum_shape, jnp.float32)]


    out, lse = jax.ffi.ffi_call(
        "flash_mha_varlen_fwd", 
        result_shape_dtypes=out_types,
        has_side_effect=False,
        input_layouts=[None]*(5 + (seqused_k is not None)), # default row major
        output_layouts=[None]*4,
        )(q, k, v, seqlens_q, seqlens_k, *[seqused_k] if seqused_k is not None else [],
        max_seqlen_q=mlir.i32_attr(max_seqlen_q),
        max_seqlen_k=mlir.i32_attr(max_seqlen_k),
        softmax_scale=softmax_scale,
        zero_tensors=False,
        is_causal=is_causal,
        window_size_left=window_size_left,
        window_size_right=window_size_right)[:2]
    
    if dpad > 0:
        out = out[:,:,:d]

    return out, lse

def _flash_mha_varlen_fwd_hlo_lowering_mlir(ctx, *args, **keywords):
    return mlir.lower_fun(_flash_mha_varlen_fwd_hlo_lowering, multiple_results=True)(ctx, *args, **keywords)

mlir.register_lowering(
    _flash_mha_varlen_fwd_p,
    _flash_mha_varlen_fwd_hlo_lowering_mlir,  # type: ignore
    platform="gpu",
)

# ==== Abstract Evaluation ====

def _flash_mha_varlen_fwd_abstract(q, k, v, seqlens_q, seqlens_k, seqused_k=None, *,
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

# ==== VMap rules ====

def _flash_mha_varlen_fwd_batch(vector_arg_values, batch_axes, *, max_seqlen_q: int, max_seqlen_k: int, has_seqused_k: bool, **kwargs):
    # move mapping axes to the front
    vector_arg_values, batch_axes = zip(*[(jnp.moveaxis(x, b, 0), 0) if b is not None else (x, b) for x, b in zip(vector_arg_values, batch_axes)])
    q, k, v, seqlens_q, seqlens_k, *rest = vector_arg_values
    if has_seqused_k:
        seqused_k, = rest
    assert all(isinstance(b, int) or b is None for b in batch_axes)
    assert isinstance(has_seqused_k, bool)
    if batch_axes in ((0,0,0,0,0),(0,0,0,0,0,0)):
        b, sq, hq, dq = q.shape
        b, sk, hk, dk = k.shape
        assert dq == dk
        assert k.shape == v.shape
        assert seqlens_q.shape == seqlens_k.shape
        b, n_plus_1 = seqlens_q.shape
        n = n_plus_1 - 1
        new_q = q.reshape((b*sq, hq, dq))
        new_k = k.reshape((b*sk, hk, dk))
        new_v = v.reshape((b*sk, hk, dk))
        new_seqlens_q = (seqlens_q + (jnp.arange(b)[:,None]*sq)).reshape((b*n_plus_1,))
        new_seqlens_k = (seqlens_k + (jnp.arange(b)[:,None]*sk)).reshape((b*n_plus_1,))
        if has_seqused_k:
            assert seqused_k.shape == (b, n)
            new_seqused_k = jnp.pad(seqused_k, ((0,0),(0,1))).reshape((b*n_plus_1,))[:-1]
        else:
            new_seqused_k = None
        out, lse = flash_mha_varlen_fwd(new_q, new_k, new_v, new_seqlens_q, new_seqlens_k, new_seqused_k,
                                        max_seqlen_q=max_seqlen_q, max_seqlen_k=max_seqlen_k, **kwargs)
        # out: (b*sq) hq dq
        # lse: (b*n_plus_1-1) hq max_seqlen_q
        new_out = out.reshape((b, sq, hq, dq))
        new_lse = jnp.pad(lse, ((0,1),(0,0),(0,0))).reshape((b,n_plus_1,hq,max_seqlen_q))[:,:-1]
        return (new_out, new_lse), (0,0)
    elif batch_axes == (None, 0, 0, None, 0):
        # broadcasting q over k,v with different seqlens_k
        x = k.shape[0]
        q = einops.repeat(q, '... -> x ...', x=x)
        seqlens_q = einops.repeat(seqlens_q, '... -> x ...', x=x)
        out, lse = jax.vmap(partial(flash_mha_varlen_fwd, max_seqlen_q=max_seqlen_q, max_seqlen_k=max_seqlen_k, **kwargs))(q, k, v, seqlens_q, seqlens_k)
        return (out, lse), (0,0)
    elif batch_axes == (0, 0, 0, None, 0):
        # broadcast seqlens_q
        x = q.shape[0]
        seqlens_q = einops.repeat(seqlens_q, '... -> x ...', x=x)
        out, lse = jax.vmap(partial(flash_mha_varlen_fwd, max_seqlen_q=max_seqlen_q, max_seqlen_k=max_seqlen_k, **kwargs))(q, k, v, seqlens_q, seqlens_k)
        return (out, lse), (0,0)
    elif batch_axes == (0, 0, 0, None, None):
        # broadcast seqlens_q & seqlens_k over batches of q/k/v is the same as doing more heads
        x = q.shape[0]
        new_q = rearrange(q, 'x sq hq dq -> sq (x hq) dq')
        new_k = rearrange(k, 'x sk hk dk -> sk (x hk) dk')
        new_v = rearrange(v, 'x sk hk dk -> sk (x hk) dk')
        out, lse = flash_mha_varlen_fwd(new_q, new_k, new_v, seqlens_q, seqlens_k, max_seqlen_q=max_seqlen_q, max_seqlen_k=max_seqlen_k, **kwargs)
        new_out = rearrange(out, 'sq (x hq) dq -> x sq hq dq', x=x)
        new_lse = rearrange(lse, 'n (x hq) mseq -> x n hq mseq', x=x)
        return (new_out, new_lse), (0,0)
    else:
        raise NotImplementedError(f"flash_mha_varlen_fwd: unsupported vmap: {batch_axes}")
batching.primitive_batchers[_flash_mha_varlen_fwd_p] = _flash_mha_varlen_fwd_batch