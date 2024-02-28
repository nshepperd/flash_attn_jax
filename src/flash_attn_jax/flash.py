from functools import partial, wraps

import numpy as np
import jax
import jax.numpy as jnp
from jax import core, dtypes
from jax.core import ShapedArray
from jax.interpreters import batching
from jax.interpreters import mlir
from jax.interpreters import xla
from jax.interpreters.mlir import ir
from jax.lib import xla_client
from jaxlib.hlo_helpers import custom_call
from jax.experimental.custom_partitioning import custom_partitioning

from jax.sharding import PartitionSpec as P
from jax.sharding import Mesh
from jax.sharding import NamedSharding
from jax.sharding import PositionalSharding

from einops import rearrange
import math

from .flash_sharding import _flash_mha_fwd_hlo_sharded, _flash_mha_bwd_hlo_sharded

# ==== Register primitives ====

# We do this with two sets of primitives, so that we can implement vjp
# and vmap for the outer "logical" mha primitives without worrying
# about sharding or padding, which will be handled when they are
# lowered to hlo, using the physical "hlo" primitives, which directly
# lower to XLA CustomCall.
_flash_mha_fwd_p = core.Primitive("flash_mha_fwd")
_flash_mha_fwd_p.multiple_results = True
_flash_mha_fwd_p.def_impl(partial(xla.apply_primitive, _flash_mha_fwd_p))

_flash_mha_bwd_p = core.Primitive("flash_mha_bwd")
_flash_mha_bwd_p.multiple_results = True
_flash_mha_bwd_p.def_impl(partial(xla.apply_primitive, _flash_mha_bwd_p))

try:
    # JAX 0.4.24 and above requires this.
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
        ShapedArray(q.shape, q_dtype, named_shape=q.named_shape),
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
        ShapedArray(q.shape, q_dtype, named_shape=q.named_shape),
        ShapedArray(k.shape, k_dtype, named_shape=k.named_shape),
        ShapedArray(v.shape, v_dtype, named_shape=v.named_shape),
    )
_flash_mha_bwd_p.def_abstract_eval(_flash_mha_bwd_abstract)

# ==== VMap rules ====

def mha_fwd_batch(vector_arg_values, batch_axes, **kwargs):
  assert tuple(batch_axes) == (0,0,0), "Only support vmapping mha over axis 0 for now,"
  [q, k, v] = vector_arg_values
  [b, n, l, h, d] = q.shape
  [b, n, lk, hk, d] = k.shape
  assert [b, n, lk, hk, d] == list(v.shape)
  out, lse = _flash_mha_fwd_p.bind(q.reshape([b*n,l,h,d]),
                                   k.reshape([b*n,lk,hk,d]),
                                   v.reshape([b*n,lk,hk,d]),
                                   **kwargs)
  return (out.reshape([b,n,*out.shape[1:]]), lse.reshape([b,n,*lse.shape[1:]])), (0,0)

def mha_bwd_batch(vector_arg_values, batch_axes, **kwargs):
  assert tuple(batch_axes) == (0,0,0,0,0,0), "Only support vmapping mha over axis 0 for now,"
  dout, q, k, v, out, lse = vector_arg_values
  b = dout.shape[batch_axes[0]]
  def join(*args):
      return [rearrange(a, 'b n ... -> (b n) ...') for a in args]
  def unjoin(*args):
      return [rearrange(a, '(b n) ... -> b n ...', b=b) for a in args]
  dq, dk, dv = _flash_mha_bwd_p.bind(*join(dout,q,k,v,out,lse),
                                     **kwargs)
  return tuple(unjoin(dq,dk,dv)), (0,0,0)

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
    def base(q,k,v,config):
        return _flash_mha_fwd(q,k,v, **config)[0]
    def fwd(q,k,v,config):
        out, lse = _flash_mha_fwd(q,k,v, **config)
        return out, (q,k,v,out,lse)
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
