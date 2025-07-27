from functools import partial, wraps

import numpy as np
import jax
import jax.numpy as jnp
from jax import core, dtypes
from jax.core import ShapedArray
from jax.interpreters import mlir
from jax.interpreters import xla
from jax.interpreters.mlir import ir

from jax.extend.core import Primitive

from einops import rearrange
import einops
import math

import flash_attn_jax_lib.flash_api as flash_api

# jax.ffi.ffi_call()

# ==== Register primitives ====

_flash_mha_fwd_hlo_p = Primitive("flash_mha_fwd_hlo")
_flash_mha_fwd_hlo_p.multiple_results = True
_flash_mha_fwd_hlo_p.def_impl(partial(xla.apply_primitive, _flash_mha_fwd_hlo_p))

_flash_mha_bwd_hlo_p = Primitive("flash_mha_bwd_hlo")
_flash_mha_bwd_hlo_p.multiple_results = True
_flash_mha_bwd_hlo_p.def_impl(partial(xla.apply_primitive, _flash_mha_bwd_hlo_p))

# ==== Primitive wrapper ====

def _flash_mha_fwd_hlo(q, k, v, softmax_scale, is_causal, window_size):
    out, lse = _flash_mha_fwd_hlo_p.bind(q, k, v, softmax_scale=softmax_scale, is_causal=is_causal, window_size=window_size)
    return out, lse

def _flash_mha_bwd_hlo(dout, q, k, v, out, lse, softmax_scale, is_causal, window_size):
    dq, dk, dv = _flash_mha_bwd_hlo_p.bind(dout, q, k, v, out, lse, softmax_scale=softmax_scale, is_causal=is_causal, window_size=window_size)
    return dq, dk, dv

# ==== HLO lowerings ====

# Register functions defined in gpu_ops as custom call target for GPUs
for _name, _value in flash_api.get_registrations().items():
    jax.ffi.register_ffi_target(_name, _value, platform="gpu", api_version=0)

def default_layouts(*shapes):
    def row_major(shape):
        return range(len(shape)-1, -1, -1)
    return [row_major(shape) for shape in shapes]

def value_layouts(*values):
    return default_layouts(*[ir.RankedTensorType(x.type).shape for x in values])

def ir_type_to_dtype(ty):
    for dtype in [np.dtype('bfloat16'), np.dtype('float16'), np.dtype('float32')]:
        if ty == mlir.dtype_to_ir_type(dtype):
            return dtype

def _flash_mha_fwd_hlo_lowering(ctx, q, k, v, softmax_scale=None, is_causal=False, window_size=None):
    q_type = ir.RankedTensorType(q.type)
    q_shape = q_type.shape
    k_type = ir.RankedTensorType(k.type)
    k_shape = k_type.shape
    v_type = ir.RankedTensorType(v.type)
    v_shape = v_type.shape

    assert q_type.element_type == k_type.element_type, "Q and K must have the same dtype"
    assert q_type.element_type == v_type.element_type, "Q and V must have the same dtype"
    element_type = q_type.element_type
    assert type(element_type) in [ir.F16Type, ir.BF16Type], "Only support fp16 and bf16 data type"
    [n, l, h, d] = q_shape
    [nk, lk, hk, dk] = k_shape
    assert k_shape == v_shape, "K and V must have the same shape"
    assert [n, d] == [nk, dk], "Q and K must have the same batch size and head size"
    assert isinstance(window_size, (tuple, list))

    opaque = flash_api.make_flash_mha_fwd_args(
        0.0, # p_dropout
        softmax_scale,
        is_causal, # is_causal
        window_size[0], # window_size_left
        window_size[1], # window_size_right
        False, # return_softmax
        n, l, h, d,
        lk, hk,
        flash_api.BF16 if type(element_type) == ir.BF16Type else flash_api.FP16,
        0)

    def fwd(q, k, v):
        dpad = (8 - d%8) % 8
        if dpad > 0:
            # We need padding. It's better to let xla's allocator handle it here than directly call cudaMalloc.
            q = jnp.pad(q, ((0,0),(0,0),(0,0),(0,dpad)), 'constant')
            k = jnp.pad(k, ((0,0),(0,0),(0,0),(0,dpad)), 'constant')
            v = jnp.pad(v, ((0,0),(0,0),(0,0),(0,dpad)), 'constant')
        
        # q_shape = [n, l, h, d+dpad]
        # k_shape = [n, lk, hk, d+dpad]
        # v_shape = [n, lk, hk, d+dpad]
        o_shape = [n, l, h, d+dpad]
        lse_shape = [n, h, l]
        
        jax_dtype = jnp.bfloat16 if type(element_type) == ir.BF16Type else jnp.float16
        out_types = [jax.ShapeDtypeStruct(o_shape, jax_dtype), jax.ShapeDtypeStruct(lse_shape, jnp.float32)]

        o, lse = jax.ffi.ffi_call(
            "flash_mha_fwd", 
            result_shape_dtypes=out_types,
            has_side_effect=False,
            legacy_backend_config=opaque,
            input_layouts=[None, None, None], # default row major
            output_layouts=[None, None],
            custom_call_api_version=1
            )(q, k, v)

        if dpad > 0:
            o = o[:,:,:,:d]
        return o, lse
    return mlir.lower_fun(fwd, multiple_results=True)(ctx, q, k, v)

mlir.register_lowering(
    _flash_mha_fwd_hlo_p,
    _flash_mha_fwd_hlo_lowering,
    platform="gpu",
)

def _flash_mha_bwd_hlo_lowering(ctx, dout, q, k, v, out, lse, softmax_scale=None, is_causal=None, window_size=None):
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
    dtype = q_type

    dout_shape = ir.RankedTensorType(dout.type).shape
    q_shape = ir.RankedTensorType(q.type).shape
    k_shape = ir.RankedTensorType(k.type).shape
    v_shape = ir.RankedTensorType(v.type).shape
    out_shape = ir.RankedTensorType(out.type).shape
    lse_shape = ir.RankedTensorType(lse.type).shape
    [n, lq, hq, d] = q_shape
    [nk, lk, hk, dk] = k_shape
    assert n == nk
    assert d == dk
    assert isinstance(window_size, (tuple, list))

    assert (list(map(list, [dout_shape, q_shape, k_shape, v_shape, out_shape, lse_shape])) ==
            [[n, lq, hq, d], [n, lq, hq, d], [n, lk, hk, d], [n, lk, hk, d],
             [n, lq, hq, d], [n, hq, lq]])


    opaque = flash_api.make_flash_mha_bwd_args(
        0.0, # p_dropout
        softmax_scale,
        is_causal, # is_causal
        window_size[0], # window_size_left
        window_size[1], # window_size_right
        False, # deterministic
        n, lq, hq, d,
        lk, hk,
        flash_api.BF16 if type(q_type) == ir.BF16Type else flash_api.FP16,
        0)

    def fwd(dout, q, k, v, out, lse):
        dpad = (8 - d%8) % 8
        if dpad > 0:
            # We need padding. It's better to let xla's allocator handle it here than directly call cudaMalloc.
            q = jnp.pad(q, ((0,0),(0,0),(0,0),(0,dpad)), 'constant')
            k = jnp.pad(k, ((0,0),(0,0),(0,0),(0,dpad)), 'constant')
            v = jnp.pad(v, ((0,0),(0,0),(0,0),(0,dpad)), 'constant')
            out = jnp.pad(out, ((0,0),(0,0),(0,0),(0,dpad)), 'constant')
            dout = jnp.pad(dout, ((0,0),(0,0),(0,0),(0,dpad)), 'constant')

        # For MQA/GQA, hq != hk, but we pass a hq sized output tensor to the kernel and sum over it afterwards to reduce the size.
        jax_dtype = jnp.bfloat16 if type(dtype) == ir.BF16Type else jnp.float16
        out_types = [jax.ShapeDtypeStruct((n, lq, hq, d+dpad), jax_dtype),
                    jax.ShapeDtypeStruct((n, lk, hq, d+dpad), jax_dtype),
                    jax.ShapeDtypeStruct((n, lk, hq, d+dpad), jax_dtype)]

        dq, dk, dv = jax.ffi.ffi_call(
            "flash_mha_bwd", 
            result_shape_dtypes=out_types,
            has_side_effect=False,
            legacy_backend_config=opaque,
            input_layouts=[None]*6, # default row major
            output_layouts=[None]*3,
            custom_call_api_version=1
            )(dout, q, k, v, out, lse)

        if hq != hk:
            assert hq > hk and hq % hk == 0
            m = hq // hk
            dk = einops.reduce(dk, 'n l (h m) d -> n l h d', reduction='sum', h=hk)
            dv = einops.reduce(dv, 'n l (h m) d -> n l h d', reduction='sum', h=hk)
        
        if dpad > 0:
            dq = dq[:,:,:,:d]
            dk = dk[:,:,:,:d]
            dv = dv[:,:,:,:d]

        return dq, dk, dv
    
    return mlir.lower_fun(fwd, multiple_results=True)(ctx, dout, q, k, v, out, lse)

mlir.register_lowering(
    _flash_mha_bwd_hlo_p,
    _flash_mha_bwd_hlo_lowering,
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
_flash_mha_fwd_hlo_p.def_abstract_eval(_flash_mha_fwd_abstract)


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
_flash_mha_bwd_hlo_p.def_abstract_eval(_flash_mha_bwd_abstract)
