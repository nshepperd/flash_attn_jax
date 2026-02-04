from dataclasses import dataclass, asdict
from functools import partial, wraps
from typing import Optional

import numpy as np
import jax
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

@jax.tree_util.register_static
@dataclass
class FlashConfig:
    max_seqlen_q: int
    max_seqlen_k: int
    softmax_scale: Optional[float]
    is_causal: bool
    window_size_left: int
    window_size_right: int


@jax.custom_vjp
def _flash_mha_varlen_vjp(q: jax.Array, k: jax.Array, v: jax.Array, seqlens_q: jax.Array, seqlens_k: jax.Array, config: FlashConfig):
    return flash_mha_varlen_fwd(q,k,v, seqlens_q, seqlens_k, **asdict(config))[0]
def _flash_mha_varlen_vjp_fwd(q,k,v,seqlens_q, seqlens_k, config):
    out, lse = flash_mha_varlen_fwd(q,k,v, seqlens_q, seqlens_k, **asdict(config))
    return out, (q,k,v,seqlens_q, seqlens_k, out,lse, config)
def _flash_mha_varlen_vjp_bwd(pack, dout):
    (q,k,v,seqlens_q, seqlens_k, out,lse, config) = pack
    dq, dk, dv = flash_mha_varlen_bwd(dout, q, k, v, out, lse, seqlens_q, seqlens_k, **asdict(config))
    return (dq,dk,dv,None,None,None)
_flash_mha_varlen_vjp.defvjp(_flash_mha_varlen_vjp_fwd, _flash_mha_varlen_vjp_bwd)

def flash_mha_varlen(q, k, v, seqlens_q, seqlens_k=None, *,
                     max_seqlen_q: int = -1, max_seqlen_k: int = -1,
                     softmax_scale: Optional[float] = None, is_causal: bool = False,
                     window_size: tuple = (-1, -1)):
    if seqlens_k is None:
        seqlens_k = seqlens_q
    config = FlashConfig(
        max_seqlen_q=max_seqlen_q,
        max_seqlen_k=max_seqlen_k,
        softmax_scale=softmax_scale,
        is_causal=is_causal,
        window_size_left=window_size[0],
        window_size_right=window_size[1],
    )
    return _flash_mha_varlen_vjp(q, k, v, seqlens_q, seqlens_k, config)