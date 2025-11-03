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

from .varlen_bwd import flash_mha_varlen_bwd
from .varlen_fwd import flash_mha_varlen_fwd

@partial(jax.custom_vjp, nondiff_argnums=(6,))
def _flash_mha_varlen_vjp(q: jax.Array, k: jax.Array, v: jax.Array, seqlens_q: jax.Array, seqlens_k: jax.Array, seqused_k: jax.Array, config: dict):
    config = dict(config)  # make a copy
    return flash_mha_varlen_fwd(q,k,v, seqlens_q, seqlens_k, seqused_k, **config)[0]
def _flash_mha_varlen_vjp_fwd(q,k,v,seqlens_q, seqlens_k, seqused_k, config):
    config = dict(config)  # make a copy
    out, lse = flash_mha_varlen_fwd(q,k,v, seqlens_q, seqlens_k, seqused_k, **config)
    return out, (q,k,v,seqlens_q, seqlens_k, seqused_k, out,lse)
def _flash_mha_varlen_vjp_bwd(config, pack, dout):
    (q,k,v,seqlens_q, seqlens_k, seqused_k, out,lse) = pack
    if seqused_k is not None:
        # the bwd doesn't support seqused_k directly
        # instead, we use a mask to zero out K and V on the input, and DK and DV on the result
        starts = seqlens_k[:-1]
        ends = seqlens_k[1:]
        lens = ends - starts
        zero = jnp.zeros(q.shape[0], dtype=jnp.int32)
        ixl = jnp.arange(q.shape[0]) - jnp.cumsum(zero.at[ends].add(lens))
        limits = jnp.cumsum(zero.at[starts].add(seqused_k-jnp.concatenate([jnp.array([0]), seqused_k[:-1]])))
        mask = (ixl < limits)
        v = v * mask[:, None, None]
        k = k * mask[:, None, None]
    dq, dk, dv = flash_mha_varlen_bwd(dout, q, k, v, out, lse, seqlens_q, seqlens_k, **config)
    if seqused_k is not None:
        dk = dk * mask[:, None, None]
        dv = dv * mask[:, None, None]
    return (dq,dk,dv,None,None,None)
_flash_mha_varlen_vjp.defvjp(_flash_mha_varlen_vjp_fwd, _flash_mha_varlen_vjp_bwd)
    
def flash_mha_varlen(q, k, v, seqlens_q, seqlens_k=None, seqused_k=None, *,
                     max_seqlen_q: int = -1, max_seqlen_k: int = -1,
                     softmax_scale: Optional[float] = None, is_causal: bool = False,
                     window_size: tuple = (-1, -1)):
    if window_size != (-1, -1):
        assert seqused_k is None, "seqused_k is not supported with local attention."
    if seqlens_k is None:
        seqlens_k = seqlens_q
    config = dict(
        max_seqlen_q=max_seqlen_q,
        max_seqlen_k=max_seqlen_k,
        softmax_scale=softmax_scale,
        is_causal=is_causal,
        window_size_left=window_size[0],
        window_size_right=window_size[1],
    )
    return _flash_mha_varlen_vjp(q, k, v, seqlens_q, seqlens_k, seqused_k, config)