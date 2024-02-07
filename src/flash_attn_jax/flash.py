from functools import partial

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
from einops import rearrange
import math

import flash_attn_jax.flash_api as flash_api

# ==== Register primitives ====

_flash_mha_fwd_p = core.Primitive("flash_mha_fwd")
_flash_mha_fwd_p.multiple_results = True
_flash_mha_fwd_p.def_impl(partial(xla.apply_primitive, _flash_mha_fwd_p))

_flash_mha_bwd_p = core.Primitive("flash_mha_bwd")
_flash_mha_bwd_p.multiple_results = True
_flash_mha_bwd_p.def_impl(partial(xla.apply_primitive, _flash_mha_bwd_p))


def flash_mha_fwd(q, k, v, softmax_scale, is_causal, window_size):
    d = q.shape[-1]
    assert len(q.shape) == 4
    assert d == k.shape[-1]
    assert d == v.shape[-1]
    if d % 8 != 0:
        # We need padding.
        padding = [(0,0),(0,0),(0,0),(0, 8 - d%8)]
        q = jnp.pad(q, padding)
        k = jnp.pad(k, padding)
        v = jnp.pad(v, padding)
    out, lse = _flash_mha_fwd_p.bind(q, k, v, softmax_scale=softmax_scale, d_og=d, is_causal=is_causal, window_size=window_size)
    if d % 8 != 0:
        out = out[..., :d]
    return out, lse

def flash_mha_bwd(dout, q, k, v, out, lse, softmax_scale, is_causal, window_size):
    d = q.shape[-1]
    assert len(q.shape) == 4
    assert d == k.shape[-1]
    assert d == v.shape[-1]
    if d % 8 != 0:
        # We need padding.
        padding = [(0,0),(0,0),(0,0),(0, 8 - d%8)]
        q = jnp.pad(q, padding)
        k = jnp.pad(k, padding)
        v = jnp.pad(v, padding)
        out = jnp.pad(out, padding)
        dout = jnp.pad(dout, padding)
    dq, dk, dv = _flash_mha_bwd_p.bind(dout, q, k, v, out, lse, softmax_scale=softmax_scale, d_og=d, is_causal=is_causal, window_size=window_size)
    if d % 8 != 0:
        return dq[...,:d], dk[...,:d], dv[...,:d]
    return dq, dk, dv

# ==== CUDA lowerings ====

# Register functions defined in gpu_ops as custom call target for GPUs
for _name, _value in flash_api.get_registrations().items():
    xla_client.register_custom_call_target(_name, _value, platform="gpu")

def default_layouts(*shapes):
    def row_major(shape):
        return range(len(shape)-1, -1, -1)
    return [row_major(shape) for shape in shapes]

def _flash_mha_fwd_cuda_lowering(ctx, q, k, v, softmax_scale=None, d_og=None, is_causal=False, window_size=None):
    # print(type(q), dir(q), q.type)
    q_type = ir.RankedTensorType(q.type)
    q_shape = q_type.shape
    k_type = ir.RankedTensorType(k.type)
    k_shape = k_type.shape
    v_type = ir.RankedTensorType(v.type)
    v_shape = v_type.shape

    assert q_type.element_type == k_type.element_type
    assert q_type.element_type == v_type.element_type
    out_element_type = q_type.element_type
    assert type(out_element_type) in [ir.F16Type, ir.BF16Type]
    [n, l, h, d] = q_shape
    [nk, lk, hk, dk] = k_shape

    out_types = [ir.RankedTensorType.get([n, l, h, d], out_element_type),
                 ir.RankedTensorType.get([n, h, l], ir.F32Type.get(ctx.module_context.context))]

    assert k_shape == v_shape
    assert [n, d] == [nk, dk]

    opaque = flash_api.make_flash_mha_fwd_args(
        0.0, # p_dropout
        softmax_scale,
        is_causal, # is_causal
        window_size[0], # window_size_left
        window_size[1], # window_size_right
        False, # return_softmax
        n, l, h, d_og or d,
        lk, hk,
        flash_api.BF16 if type(out_element_type) == ir.BF16Type else flash_api.FP16,
        0)
    out = custom_call(
        b"flash_mha_fwd",
        result_types=out_types,
        operands=[q, k, v],
        backend_config=opaque,
        operand_layouts=default_layouts(q_shape, k_shape, v_shape),
        result_layouts=default_layouts(*[o.shape for o in out_types]),
    ).results
    return out

mlir.register_lowering(
    _flash_mha_fwd_p,
    _flash_mha_fwd_cuda_lowering,
    platform="gpu",
)

def _flash_mha_bwd_cuda_lowering(ctx, dout, q, k, v, out, lse, softmax_scale=None, d_og=None, is_causal=None, window_size=None):
    dout_type = ir.RankedTensorType(dout.type).element_type
    q_type = ir.RankedTensorType(q.type).element_type
    k_type = ir.RankedTensorType(k.type).element_type
    v_type = ir.RankedTensorType(v.type).element_type
    out_type = ir.RankedTensorType(out.type).element_type
    lse_type = ir.RankedTensorType(lse.type).element_type

    assert type(q_type) in [ir.F16Type, ir.BF16Type]
    assert q_type == dout_type
    assert q_type == k_type
    assert q_type == v_type
    assert q_type == out_type
    assert type(lse_type) in [ir.F32Type]

    dout_shape = ir.RankedTensorType(dout.type).shape
    q_shape = ir.RankedTensorType(q.type).shape
    k_shape = ir.RankedTensorType(k.type).shape
    v_shape = ir.RankedTensorType(v.type).shape
    out_shape = ir.RankedTensorType(out.type).shape
    lse_shape = ir.RankedTensorType(lse.type).shape
    [n, lq, hq, d] = q_shape
    [_, lk, hk, _] = k_shape

    # print(type(dout_shape))

    assert (list(map(list, [dout_shape, q_shape, k_shape, v_shape, out_shape, lse_shape])) ==
            [[n, lq, hq, d], [n, lq, hq, d], [n, lk, hk, d], [n, lk, hk, d],
             [n, lq, hq, d], [n, hq, lq]])

    out_types = [ir.RankedTensorType.get(q_shape, q_type),
                 ir.RankedTensorType.get(k_shape, k_type),
                 ir.RankedTensorType.get(v_shape, v_type)]

    opaque = flash_api.make_flash_mha_bwd_args(
        0.0, # p_dropout
        softmax_scale,
        is_causal, # is_causal
        window_size[0], # window_size_left
        window_size[1], # window_size_right
        False, # deterministic
        n, lq, hq, d_og or d,
        lk, hk,
        flash_api.BF16 if type(q_type) == ir.BF16Type else flash_api.FP16,
        0)
    out = custom_call(
        b"flash_mha_bwd",
        result_types=out_types,
        operands=[dout, q, k, v, out, lse],
        backend_config=opaque,
        operand_layouts=default_layouts(dout_shape, q_shape, k_shape, v_shape, out_shape, lse_shape),
        result_layouts=default_layouts(*[o.shape for o in out_types]),
    ).results
    return out

mlir.register_lowering(
    _flash_mha_bwd_p,
    _flash_mha_bwd_cuda_lowering,
    platform="gpu",
)

# ==== Abstract evaluation rules ====

def _flash_mha_fwd_abstract(q, k, v, softmax_scale=None, d_og=None, is_causal=None, window_size=None):
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


def _flash_mha_bwd_abstract(dout, q, k, v, out, lse, softmax_scale=None, d_og=None, is_causal=None, window_size=None):
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
  out, lse = flash_mha_fwd(q.reshape([b*n,l,h,d]),
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
  dq, dk, dv = flash_mha_fwd(*join(dout,q,k,v,out,lse),
                             **kwargs)
  return tuple(unjoin(dq,dk,dv)), (0,0,0)

batching.primitive_batchers[_flash_mha_fwd_p] = mha_fwd_batch
batching.primitive_batchers[_flash_mha_bwd_p] = mha_bwd_batch

# ==== VJP Rule ====

def custom_vjp(cls, nondiff_argnums=()):
    f = jax.custom_vjp(cls.base, nondiff_argnums=nondiff_argnums)
    f.defvjp(cls.fwd, cls.bwd)
    return f

# Apparently we need nondiff_argnums so that softmax_scale doesn't get
# turned into a Tracer, which we can't use as a static parameter. It
# gets placed at the front of the argument list in bwd.
@partial(custom_vjp, nondiff_argnums=(3,))
class _flash_mha_vjp:
    def base(q,k,v,config):
        return flash_mha_fwd(q,k,v, **config)[0]
    def fwd(q,k,v,config):
        out, lse = flash_mha_fwd(q,k,v, **config)
        return out, (q,k,v,out,lse)
    def bwd(config, pack, dout):
        (q,k,v,out,lse) = pack
        dq, dk, dv = flash_mha_bwd(dout, q, k, v, out, lse, **config)
        return (dq,dk,dv)

# ==== Frontend ====

def flash_mha(q,k,v,softmax_scale=None, is_causal=False, window_size=(-1,-1)):
    """Flash attention.

    softmax_scale defaults to 1/sqrt(d) and must be a python float if
    provided (ie. can't be a tensor or a tracer).0

    """
    if softmax_scale is None:
        softmax_scale = 1/math.sqrt(q.shape[-1])
    assert type(softmax_scale) is float
    o = _flash_mha_vjp(q,k,v,dict(softmax_scale=softmax_scale, is_causal=is_causal, window_size=window_size))
    return o
