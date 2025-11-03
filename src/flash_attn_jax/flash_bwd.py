from functools import partial, wraps
from typing import Optional, Sequence

import numpy as np
import jax
import jax.numpy as jnp
from jax import Array, core, dtypes
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

_flash_mha_bwd_p = Primitive("flash_mha_bwd")
_flash_mha_bwd_p.multiple_results = True
_flash_mha_bwd_p.def_impl(partial(xla.apply_primitive, _flash_mha_bwd_p))
jax._src.dispatch.prim_requires_devices_during_lowering.add(_flash_mha_bwd_p)

# ==== Frontend ====


def flash_mha_bwd(dout, q, k, v, o, lse, *,
                  softmax_scale: Optional[float] = None, is_causal: bool = False,
                  window_size: tuple = (-1, -1), deterministic: bool = False):
    d = q.shape[-1]
    if softmax_scale is None:
        softmax_scale = 1.0 / math.sqrt(d)
    kwargs = dict(
        softmax_scale=softmax_scale,
        is_causal=is_causal,
        window_size_left=window_size[0],
        window_size_right=window_size[1],
        deterministic=deterministic,
    )
    return tuple(_flash_mha_bwd_p.bind(dout, q, k, v, o, lse, **kwargs))

# ==== HLO lowering ====

def _flash_mha_bwd_lowering(dout, q, k, v, out, lse, *, softmax_scale: float, is_causal: bool, window_size_left: int, window_size_right: int, deterministic: bool):
    [n, lq, hq, d] = q.shape
    [_, lk, hk, _] = k.shape
    dtype = q.dtype
    
    dpad = (8 - d%8) % 8
    if dpad > 0:
        # We need padding. It's better to let xla's allocator handle it here than directly call cudaMalloc.
        q = jnp.pad(q, ((0,0),(0,0),(0,0),(0,dpad)), 'constant')
        k = jnp.pad(k, ((0,0),(0,0),(0,0),(0,dpad)), 'constant')
        v = jnp.pad(v, ((0,0),(0,0),(0,0),(0,dpad)), 'constant')
        out = jnp.pad(out, ((0,0),(0,0),(0,0),(0,dpad)), 'constant')
        dout = jnp.pad(dout, ((0,0),(0,0),(0,0),(0,dpad)), 'constant')

    # For MQA/GQA, hq != hk, but we pass a hq sized output tensor to the kernel and sum over it afterwards to reduce the size.
    # Calculate scratch array shapes
    lq_rounded = round_multiple(lq, 128)
    d_rounded = round_multiple(d+dpad, 32)
    softmax_d_shape = (n, hq, lq_rounded)
    
    # Calculate nsplits for deterministic mode
    sm_count = 114  # H100, should ideally get this from device query
    if deterministic:
        nsplits = max(1, (sm_count + n * hq - 1) // (n * hq))
        dq_accum_shape = (nsplits, n, lq_rounded, hq, d_rounded)
    else:
        dq_accum_shape = (n, lq_rounded, hq, d_rounded)
    
    rng_state_shape = (2,)
    
    out_types = [jax.ShapeDtypeStruct((n, lq, hq, d+dpad), dtype),  # dq
                jax.ShapeDtypeStruct((n, lk, hq, d+dpad), dtype),   # dk
                jax.ShapeDtypeStruct((n, lk, hq, d+dpad), dtype),   # dv
                jax.ShapeDtypeStruct(softmax_d_shape, jnp.float32),     # softmax_d
                jax.ShapeDtypeStruct(dq_accum_shape, jnp.float32),      # dq_accum
                jax.ShapeDtypeStruct(rng_state_shape, jnp.int64)]       # rng_state

    dq, dk, dv = jax.ffi.ffi_call(
        "flash_mha_bwd",
        result_shape_dtypes=out_types,
        has_side_effect=False,
        input_layouts=[None]*6, # default row major
        output_layouts=[None]*6,
        )(dout, q, k, v, out, lse,
          softmax_scale=softmax_scale,
        is_causal=is_causal,
        window_size_left=window_size_left,
        window_size_right=window_size_right,
        deterministic=deterministic)[:3]  # Only return first 3 outputs (dq, dk, dv)

    if hq != hk:
        assert hq > hk and hq % hk == 0
        m = hq // hk
        dk = einops.reduce(dk, 'n l (h m) d -> n l h d', reduction='sum', h=hk, m=m)
        dv = einops.reduce(dv, 'n l (h m) d -> n l h d', reduction='sum', h=hk, m=m)
    
    if dpad > 0:
        dq = dq[:,:,:,:d]
        dk = dk[:,:,:,:d]
        dv = dv[:,:,:,:d]

    return dq, dk, dv

def _flash_mha_bwd_lowering_mlir(ctx, dout, q, k, v, out, lse, **keywords):
    return mlir.lower_fun(_flash_mha_bwd_lowering, multiple_results=True)(ctx, dout, q, k, v, out, lse, **keywords)

mlir.register_lowering(
    _flash_mha_bwd_p,
    _flash_mha_bwd_lowering_mlir,  # type: ignore
    platform="gpu",
)

# ==== Abstract Evaluation ====

def _flash_mha_bwd_abstract(dout, q, k, v, o, lse, **keywords):
    q_dtype = dtypes.canonicalize_dtype(q.dtype)
    k_dtype = dtypes.canonicalize_dtype(k.dtype)
    v_dtype = dtypes.canonicalize_dtype(v.dtype)
    dq_shape = q.shape
    dk_shape = k.shape
    dv_shape = v.shape
    return (
        ShapedArray(dq_shape, q_dtype),
        ShapedArray(dk_shape, k_dtype),
        ShapedArray(dv_shape, v_dtype),
    )
_flash_mha_bwd_p.def_abstract_eval(_flash_mha_bwd_abstract)

# ==== VMap rules ====

def mha_bwd_batch(vector_arg_values: Sequence[Array], batch_axes, **kwargs):
    assert all(isinstance(b, int) or b is None for b in batch_axes)
    vector_arg_values, batch_axes = zip(*[(jnp.moveaxis(x, b, 0), 0) if b is not None else (x, b) for x, b in zip(vector_arg_values, batch_axes)])
    mapped = tuple(isinstance(b, int) for b in batch_axes)
    if mapped == (True, True, True, True, True, True):
        x = vector_arg_values[0].shape[0]
        do, q, k, v, o, lse = [einops.rearrange(val, 'x n ... -> (x n) ...') for val in vector_arg_values]
        dq, dk, dv = _flash_mha_bwd_p.bind(do, q, k, v, o, lse, **kwargs)
        dq = einops.rearrange(dq, '(n x) l h d -> x n l h d', x=x)
        dk = einops.rearrange(dk, '(n x) l h d -> x n l h d', x=x)
        dv = einops.rearrange(dv, '(n x) l h d -> x n l h d', x=x)
        return (dq,dk,dv), (0,0,0)
    elif mapped == (True, True, False, False, True, True):
        # Everything is mapped except k and v, which is a GQA backward
        x = vector_arg_values[0].shape[0]
        do, q, k, v, o, lse = vector_arg_values
        do = einops.rearrange(do, 'x n sq hq d -> n sq (hq x) d')
        q = einops.rearrange(q, 'x n sq hq d -> n sq (hq x) d')
        o = einops.rearrange(o, 'x n sq hq d -> n sq (hq x) d')
        lse = einops.rearrange(lse, 'x n hq sq -> n (hq x) sq')
        dq, dk, dv = _flash_mha_bwd_p.bind(do, q, k, v, o, lse, **kwargs)
        dq = einops.rearrange(dq, 'n l (h x) d -> x n l h d', x=x)
        return (dq,dk,dv), (0,None,None)
    else:
        raise NotImplementedError("MHA bwd only support vmapping over q or (q,k,v) for now, got batch axes " + str(batch_axes))

batching.primitive_batchers[_flash_mha_bwd_p] = mha_bwd_batch