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
from jax.experimental.custom_partitioning import custom_partitioning

from jax.sharding import PartitionSpec as P
from jax.sharding import Mesh
from jax.sharding import NamedSharding
from jax.sharding import PositionalSharding

from einops import rearrange
import einops
import math

import flash_attn_jax_lib.flash_api as flash_api

# ==== Register primitives ====

_flash_mha_fwd_hlo_p = core.Primitive("flash_mha_fwd_hlo")
_flash_mha_fwd_hlo_p.multiple_results = True
_flash_mha_fwd_hlo_p.def_impl(partial(xla.apply_primitive, _flash_mha_fwd_hlo_p))

_flash_mha_bwd_hlo_p = core.Primitive("flash_mha_bwd_hlo")
_flash_mha_bwd_hlo_p.multiple_results = True
_flash_mha_bwd_hlo_p.def_impl(partial(xla.apply_primitive, _flash_mha_bwd_hlo_p))

_custom_call_p = core.Primitive("custom_call")
_custom_call_p.multiple_results = True
_custom_call_p.def_impl(partial(xla.apply_primitive, _custom_call_p))

# ==== Primitive wrapper ====

def _flash_mha_fwd_hlo(q, k, v, softmax_scale, is_causal, window_size):
    out, lse = _flash_mha_fwd_hlo_p.bind(q, k, v, softmax_scale=softmax_scale, is_causal=is_causal, window_size=window_size)
    return out, lse

def _flash_mha_bwd_hlo(dout, q, k, v, out, lse, softmax_scale, is_causal, window_size):
    dq, dk, dv = _flash_mha_bwd_hlo_p.bind(dout, q, k, v, out, lse, softmax_scale=softmax_scale, is_causal=is_causal, window_size=window_size)
    return dq, dk, dv

def custom_call(*args, call_target_name, result_types, backend_config, operand_layouts, result_layouts):
    return _custom_call_p.bind(*args, call_target_name=call_target_name, result_types=result_types, backend_config=backend_config, operand_layouts=operand_layouts, result_layouts=result_layouts)

# ==== HLO lowerings ====

# Register functions defined in gpu_ops as custom call target for GPUs
for _name, _value in flash_api.get_registrations().items():
    xla_client.register_custom_call_target(_name, _value, platform="gpu")

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

    assert q_type.element_type == k_type.element_type
    assert q_type.element_type == v_type.element_type
    element_type = q_type.element_type
    assert type(element_type) in [ir.F16Type, ir.BF16Type]
    [n, l, h, d] = q_shape
    [nk, lk, hk, dk] = k_shape


    assert k_shape == v_shape
    assert [n, d] == [nk, dk]

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

    lse_type = ir.RankedTensorType.get([n, h, l], ir.F32Type.get(ctx.module_context.context))

    if d % 8 != 0:
        # We need padding. It's better to let xla's allocator handle it here than directly call cudaMalloc.
        dpad = 8 - d%8

        z = np.array(0.0, dtype=ir_type_to_dtype(element_type))
        z = mlir.ir_constant(z)
        q_padded = mlir.hlo.PadOp(q,z,[0,0,0,0],[0,0,0,dpad],[0,0,0,0]).result
        k_padded = mlir.hlo.PadOp(k,z,[0,0,0,0],[0,0,0,dpad],[0,0,0,0]).result
        v_padded = mlir.hlo.PadOp(v,z,[0,0,0,0],[0,0,0,dpad],[0,0,0,0]).result

        q_shape = [n, l, h, d+dpad]
        k_shape = [n, lk, hk, d+dpad]
        v_shape = [n, lk, hk, d+dpad]
        o_shape = [n, l, h, d+dpad]

        out_types = [ir.RankedTensorType.get(o_shape, element_type), lse_type]

        (o, lse) = mlir.custom_call(
            b"flash_mha_fwd",
            result_types=out_types,
            operands=[q_padded, k_padded, v_padded],
            backend_config=opaque,
            operand_layouts=default_layouts(q_shape, k_shape, v_shape),
            result_layouts=default_layouts(*[o.shape for o in out_types]),
        ).results

        o = mlir.hlo.SliceOp(o, [0,0,0,0], (n, l, h, d), [1,1,1,1]).result
        return (o,lse)
    else:
        out_types = [ir.RankedTensorType.get([n, l, h, d], element_type), lse_type]
        out = mlir.custom_call(
            b"flash_mha_fwd",
            result_types=out_types,
            operands=[q, k, v],
            backend_config=opaque,
            operand_layouts=default_layouts(q_shape, k_shape, v_shape),
            result_layouts=default_layouts(*[o.shape for o in out_types]),
        ).results
        return out

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
        out_types = [ir.RankedTensorType.get([n, lq, hq, d+dpad], dtype),
                    ir.RankedTensorType.get([n, lk, hq, d+dpad], dtype),
                    ir.RankedTensorType.get([n, lk, hq, d+dpad], dtype)]
        out_layouts = default_layouts([n, lq, hq, d+dpad], [n, lk, hq, d+dpad], [n, lk, hq, d+dpad])

        dq, dk, dv = custom_call(
            dout, q, k, v, out, lse,
            call_target_name=b"flash_mha_bwd",
            operand_layouts=default_layouts(dout.shape, q.shape, k.shape, v.shape, out.shape, lse.shape),
            backend_config=opaque,
            result_types=out_types,
            result_layouts=out_layouts,
        )

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
        ShapedArray(q.shape, q_dtype, named_shape=q.named_shape),
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
        ShapedArray(q.shape, q_dtype, named_shape=q.named_shape),
        ShapedArray(k.shape, k_dtype, named_shape=k.named_shape),
        ShapedArray(v.shape, v_dtype, named_shape=v.named_shape),
    )
_flash_mha_bwd_hlo_p.def_abstract_eval(_flash_mha_bwd_abstract)

# ==== Custom Call ====

def _custom_call_abstract_eval(*args, call_target_name, result_types, backend_config, operand_layouts, result_layouts):
    def convert(ty):
        ty = ir.RankedTensorType(ty)
        shape = tuple(ty.shape)
        dtype = ir_type_to_dtype(ty.element_type)
        return ShapedArray(shape, dtype)
    out_types = [convert(o) for o in result_types]
    return tuple(out_types)

_custom_call_p.def_abstract_eval(_custom_call_abstract_eval)

def _custom_call_hlo_lowering(ctx, *args, call_target_name, result_types, backend_config, operand_layouts, result_layouts):
    out = mlir.custom_call(
            call_target_name,
            operands=args,
            result_types=result_types,
            backend_config=backend_config,
            operand_layouts=operand_layouts,
            result_layouts=result_layouts,
        ).results
    return out

mlir.register_lowering(
    _custom_call_p,
    _custom_call_hlo_lowering
)
