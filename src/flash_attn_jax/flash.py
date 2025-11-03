from functools import partial, wraps

import numpy as np
import jax
import jax.numpy as jnp
from jax import core, dtypes
from jax.core import ShapedArray
from jax.interpreters import batching
from jax.interpreters import mlir
from jax.interpreters import xla
from jax.extend.core import Primitive

from einops import rearrange
import einops
import math

from .flash_fwd import flash_mha_fwd
from .flash_bwd import flash_mha_bwd



# ==== VJP Rule ====

def custom_vjp(cls, nondiff_argnums=()):
    f = jax.custom_vjp(cls.base, nondiff_argnums=nondiff_argnums)
    f.defvjp(cls.fwd, cls.bwd)
    return f

# Apparently we need nondiff_argnums so that config doesn't get turned
# into Tensors. They get placed at the front of the argument list in
# bwd.
@partial(custom_vjp, nondiff_argnums=(3,))
class _flash_mha_vjp:
    @staticmethod
    def base(q,k,v,config):
        return flash_mha_fwd(q,k,v, **config)[0]
    @staticmethod
    def fwd(q,k,v,config):
        out, lse = flash_mha_fwd(q,k,v, **config)
        return out, (q,k,v,out,lse)
    @staticmethod
    def bwd(config, pack, dout):
        (q,k,v,out,lse) = pack
        dq, dk, dv = flash_mha_bwd(dout, q, k, v, out, lse, **config)
        return (dq,dk,dv)

# ==== Frontend ====

def flash_mha(q,k,v,softmax_scale=None, is_causal=False, window_size=(-1,-1)):
    """Flash attention.

    softmax_scale defaults to 1/sqrt(d) and must be a python float if
    provided (ie. can't be a tensor or a tracer).

    """
    [nq, sq, hq, dq] = q.shape
    [nk, sk, hk, dk] = k.shape
    [nv, sv, hv, dv] = v.shape
    assert nq == nk == nv
    assert hk == hv
    assert nq % nk == 0 # Can be larger than nk if GQA
    assert dq == dk == dv # Don't support head size mismatch
    assert sk == sv
    assert q.dtype == k.dtype == v.dtype
    assert q.dtype in [jnp.bfloat16, jnp.float16]

    window_size_left, window_size_right = window_size
    return _flash_mha_vjp(q,k,v,dict(softmax_scale=softmax_scale, is_causal=is_causal, window_size_left=window_size_left, window_size_right=window_size_right))
