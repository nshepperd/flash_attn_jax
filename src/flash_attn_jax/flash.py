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

from .flash_sharding import _flash_mha_fwd_hlo_sharded, _flash_mha_bwd_hlo_sharded

# ==== Register primitives ====

# We do this with two sets of primitives, so that we can implement vjp
# and vmap for the outer "logical" mha primitives without worrying
# about sharding or padding, which will be handled when they are
# lowered to hlo, using the physical "hlo" primitives, which directly
# lower to XLA CustomCall.
_flash_mha_fwd_p = Primitive("flash_mha_fwd")
_flash_mha_fwd_p.multiple_results = True
_flash_mha_fwd_p.def_impl(partial(xla.apply_primitive, _flash_mha_fwd_p))

_flash_mha_bwd_p = Primitive("flash_mha_bwd")
_flash_mha_bwd_p.multiple_results = True
_flash_mha_bwd_p.def_impl(partial(xla.apply_primitive, _flash_mha_bwd_p))

try:
    # JAX 0.4.24 and above requires this because of custom partitioning.
    import jax._src.dispatch
    jax._src.dispatch.prim_requires_devices_during_lowering.add(_flash_mha_bwd_p)
    jax._src.dispatch.prim_requires_devices_during_lowering.add(_flash_mha_fwd_p)
except Exception as e:
    pass

# ==== Primitive frontends ====

def _flash_mha_fwd(q,k,v, softmax_scale, is_causal, window_size):
    return tuple(_flash_mha_fwd_p.bind(q,k,v, softmax_scale=softmax_scale, is_causal=is_causal, window_size=window_size))

def _flash_mha_bwd(dout, q, k, v, out, lse, softmax_scale, is_causal, window_size):
    return tuple(_flash_mha_bwd_p.bind(dout, q, k, v, out, lse, softmax_scale=softmax_scale, is_causal=is_causal, window_size=window_size))

# ==== HLO lowering ====

mlir.register_lowering(
    _flash_mha_fwd_p,
    mlir.lower_fun(_flash_mha_fwd_hlo_sharded),
    platform="gpu",
)

mlir.register_lowering(
    _flash_mha_bwd_p,
    mlir.lower_fun(_flash_mha_bwd_hlo_sharded),
    platform="gpu",
)

# ==== Abstract evaluation rules ====

def _flash_mha_fwd_abstract(q, k, v, softmax_scale=None, is_causal=None, window_size=None):
    q_dtype = dtypes.canonicalize_dtype(q.dtype)
    k_dtype = dtypes.canonicalize_dtype(k.dtype)
    v_dtype = dtypes.canonicalize_dtype(v.dtype)
    [n, l, h, d] = q.shape
    assert q_dtype == k_dtype and q_dtype == v_dtype
    assert q_dtype in [jnp.bfloat16, jnp.float16]
    return (
        ShapedArray(q.shape, q_dtype),
        ShapedArray([n, h, l], jnp.float32)
    )
_flash_mha_fwd_p.def_abstract_eval(_flash_mha_fwd_abstract)


def _flash_mha_bwd_abstract(dout, q, k, v, out, lse, softmax_scale=None, is_causal=None, window_size=None):
    dout_dtype = dtypes.canonicalize_dtype(dout.dtype)
    q_dtype = dtypes.canonicalize_dtype(q.dtype)
    k_dtype = dtypes.canonicalize_dtype(k.dtype)
    v_dtype = dtypes.canonicalize_dtype(v.dtype)
    out_dtype = dtypes.canonicalize_dtype(out.dtype)
    lse_dtype = dtypes.canonicalize_dtype(lse.dtype)
    [n, lq, hq, d] = q.shape
    assert len(set([dout_dtype, q_dtype, k_dtype, v_dtype, out_dtype])) == 1
    assert q_dtype in [jnp.bfloat16, jnp.float16]
    return (
        ShapedArray(q.shape, q_dtype),
        ShapedArray(k.shape, k_dtype),
        ShapedArray(v.shape, v_dtype),
    )
_flash_mha_bwd_p.def_abstract_eval(_flash_mha_bwd_abstract)

# ==== VMap rules ====

def mha_fwd_batch(vector_arg_values, batch_axes, **kwargs):
  assert all(isinstance(b, int) or b is None for b in batch_axes)
  mapped = tuple(isinstance(b, int) for b in batch_axes)
  if mapped == (True, True, True):
    x = vector_arg_values[0].shape[batch_axes[0]]
    def squish(val, axis):
        dims = ['n', 'l', 'h', 'd']
        dims.insert(axis, 'x')
        dims = ' '.join(dims)
        return einops.rearrange(val, f'{dims} -> (x n) l h d')
    def unsquish(val):
        return einops.rearrange(val, f'(x n) ... -> x n ...', x=x)
    [q, k, v] = [squish(x, axis) for x, axis in zip(vector_arg_values, batch_axes)]
    out, lse = _flash_mha_fwd_p.bind(q, k, v, **kwargs)
    return (unsquish(out), unsquish(lse)), (0,0)
  elif mapped == (True, False, False):
    # This is just a GQA!
    x = vector_arg_values[0].shape[batch_axes[0]]
    def squish(val, axis):
        if axis is None:
            return val
        dims = ['n', 'l', 'h', 'd']
        dims.insert(axis, 'x')
        dims = ' '.join(dims)
        return einops.rearrange(val, f'{dims} -> n l (h x) d')
    def unsquish(val):
        return einops.rearrange(val, 'n l (h x) d -> x n l h d', x=x)
    [q, k, v] = [squish(x, axis) for x, axis in zip(vector_arg_values, batch_axes)]
    out, lse = _flash_mha_fwd_p.bind(q, k, v, **kwargs)
    out = einops.rearrange(out, 'n l (h x) d -> x n l h d', x=x)
    lse = einops.rearrange(lse, 'n (h x) l -> x n h l', x=x)
    return (out, lse), (0,0)
  else:
    raise NotImplementedError("MHA fwd only support vmapping over q or (q,k,v) for now, got batch axes " + str(batch_axes))

def mha_bwd_batch(vector_arg_values, batch_axes, **kwargs):
    assert all(isinstance(b, int) or b is None for b in batch_axes)
    mapped = tuple(isinstance(b, int) for b in batch_axes)
    if mapped == (True, True, True, True, True, True):
        x = vector_arg_values[0].shape[batch_axes[0]]
        def squish(val, axis):
            if len(val.shape) == 5:
                # q/k/v/o
                dims = ['n', 'l', 'h', 'd']
                dims.insert(axis, 'x')
                dims = ' '.join(dims)
                return einops.rearrange(val, f'{dims} -> (x n) l h d')
            elif len(val.shape) == 4:
                # lse
                dims = ['n', 'h', 'l']
                dims.insert(axis, 'x')
                dims = ' '.join(dims)
                return einops.rearrange(val, f'{dims} -> (x n) h l')
        do, q, k, v, o, lse = [squish(x, axis) for x, axis in zip(vector_arg_values, batch_axes)]
        dq, dk, dv = _flash_mha_bwd_p.bind(do, q, k, v, o, lse, **kwargs)
        dq = einops.rearrange(dq, '(n x) l h d -> x n l h d', x=x)
        dk = einops.rearrange(dk, '(n x) l h d -> x n l h d', x=x)
        dv = einops.rearrange(dv, '(n x) l h d -> x n l h d', x=x)
        return (dq,dk,dv), (0,0,0)
    elif mapped == (True, True, False, False, True, True):
        # Everything is mapped except k and v, which is a GQA backward
        x = vector_arg_values[0].shape[batch_axes[0]]
        def squish(val, axis):
            if len(val.shape) == 5:
                # q/k/v/o
                dims = ['n', 'l', 'h', 'd']
                dims.insert(axis, 'x')
                dims = ' '.join(dims)
                return einops.rearrange(val, f'{dims} -> n l (h x) d')
            elif len(val.shape) == 4:
                # lse
                dims = ['n', 'h', 'l']
                dims.insert(axis, 'x')
                dims = ' '.join(dims)
                return einops.rearrange(val, f'{dims} -> n (h x) l')
        do = squish(vector_arg_values[0], batch_axes[0])
        q = squish(vector_arg_values[1], batch_axes[1])
        k = vector_arg_values[2]
        v = vector_arg_values[3]
        o = squish(vector_arg_values[4], batch_axes[4])
        lse = squish(vector_arg_values[5], batch_axes[5])
        dq, dk, dv = _flash_mha_bwd_p.bind(do, q, k, v, o, lse, **kwargs)
        dq = einops.rearrange(dq, 'n l (h x) d -> x n l h d', x=x)
        return (dq,dk,dv), (0,None,None)
    else:
        raise NotImplementedError("MHA bwd only support vmapping over q or (q,k,v) for now, got batch axes " + str(batch_axes))

batching.primitive_batchers[_flash_mha_fwd_p] = mha_fwd_batch
batching.primitive_batchers[_flash_mha_bwd_p] = mha_bwd_batch

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
        return _flash_mha_fwd(q,k,v, **config)[0]
    @staticmethod
    def fwd(q,k,v,config):
        out, lse = _flash_mha_fwd(q,k,v, **config)
        return out, (q,k,v,out,lse)
    @staticmethod
    def bwd(config, pack, dout):
        (q,k,v,out,lse) = pack
        dq, dk, dv = _flash_mha_bwd(dout, q, k, v, out, lse, **config)
        return (dq,dk,dv)

# ==== Frontend ====

def flash_mha(q,k,v,softmax_scale=None, is_causal=False, window_size=(-1,-1)):
    """Flash attention.

    softmax_scale defaults to 1/sqrt(d) and must be a python float if
    provided (ie. can't be a tensor or a tracer).0

    """
    assert len(q.shape) == 4
    assert len(k.shape) == 4
    assert len(v.shape) == 4

    if softmax_scale is None:
        softmax_scale = 1/math.sqrt(q.shape[-1])
    assert type(softmax_scale) is float
    o = _flash_mha_vjp(q,k,v,dict(softmax_scale=softmax_scale, is_causal=is_causal, window_size=window_size))
    return o
