from functools import partial, wraps
from typing import List, Optional, Sequence

import numpy as np
import jax
import jax.numpy as jnp
from jax import Array, core, dtypes
from jax.core import ShapedArray
from jax.experimental.custom_partitioning import (
    ArrayMapping,
    CompoundFactor,
    SdyShardingRule,
    custom_partitioning,
)
from jax.interpreters import batching
from jax.interpreters import mlir
from jax.interpreters.mlir import ir
from jax.interpreters import xla
from jax.extend.core import Primitive
from jax.sharding import Mesh, NamedSharding
from jax.sharding import PartitionSpec as P
import jax._src.dispatch

from einops import rearrange
import einops
import math

from flash_attn_jax.util import round_multiple, array_mapping

# ==== Register primitives ====

_flash_mha_bwd_p = Primitive("flash_mha_bwd")
_flash_mha_bwd_p.multiple_results = True
_flash_mha_bwd_p.def_impl(partial(xla.apply_primitive, _flash_mha_bwd_p))
jax._src.dispatch.prim_requires_devices_during_lowering.add(_flash_mha_bwd_p)

# ==== Frontend ====


def flash_mha_bwd(dout, q, k, v, o, lse, *,
                  softmax_scale: Optional[float] = None, is_causal: bool = False,
                  window_size_left: int = -1, window_size_right: int = -1):
    kwargs = dict(
        softmax_scale=softmax_scale,
        is_causal=is_causal,
        window_size_left=window_size_left,
        window_size_right=window_size_right,
    )
    return tuple(_flash_mha_bwd_p.bind(dout, q, k, v, o, lse, **kwargs))

# ==== HLO lowering ====

def _flash_mha_bwd_lowering(dout, q, k, v, out, lse, *, softmax_scale: Optional[float], is_causal: bool, window_size_left: int, window_size_right: int):
    [n, lq, hq, d] = q.shape
    [_, lk, hk, _] = k.shape
    dtype = q.dtype
    
    if softmax_scale is None:
        softmax_scale = 1.0 / math.sqrt(d)

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
    if False: # deterministic mode
        nsplits = max(1, (sm_count + n * hq - 1) // (n * hq))
        dq_accum_shape = (nsplits, n, lq_rounded, hq, d_rounded)
    else:
        dq_accum_shape = (n, lq_rounded, hq, d_rounded)
    
    out_types = [jax.ShapeDtypeStruct((n, lq, hq, d+dpad), dtype),  # dq
                jax.ShapeDtypeStruct((n, lk, hq, d+dpad), dtype),   # dk
                jax.ShapeDtypeStruct((n, lk, hq, d+dpad), dtype),   # dv
                jax.ShapeDtypeStruct(softmax_d_shape, jnp.float32),     # softmax_d
                jax.ShapeDtypeStruct(dq_accum_shape, jnp.float32)]       # rng_state

    dq, dk, dv = jax.ffi.ffi_call(
        "flash_mha_bwd",
        result_shape_dtypes=out_types,
        has_side_effect=False,
        input_layouts=[None]*6, # default row major
        output_layouts=[None]*5,
        )(dout, q, k, v, out, lse,
          softmax_scale=softmax_scale,
        is_causal=is_causal,
        window_size_left=window_size_left,
        window_size_right=window_size_right,
        deterministic=False)[:3]  # Only return first 3 outputs (dq, dk, dv)

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
    return mlir.lower_fun(
        partial(_flash_mha_bwd_lowering_sharded, **keywords), multiple_results=True
    )(ctx, dout, q, k, v, out, lse)

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
        dq, dk, dv = flash_mha_bwd(do, q, k, v, o, lse, **kwargs)
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
        dq, dk, dv = flash_mha_bwd(do, q, k, v, o, lse, **kwargs)
        dq = einops.rearrange(dq, 'n l (h x) d -> x n l h d', x=x)
        return (dq,dk,dv), (0,None,None)
    else:
        raise NotImplementedError("MHA bwd only support vmapping over q or (q,k,v) for now, got batch axes " + str(batch_axes))

batching.primitive_batchers[_flash_mha_bwd_p] = mha_bwd_batch

# ==== Sharding ====

@partial(custom_partitioning, static_argnums=(6, 7, 8, 9))
def _flash_mha_bwd_lowering_sharded(
    dout,
    q,
    k,
    v,
    out,
    lse,
    softmax_scale: Optional[float],
    is_causal: bool,
    window_size_left: int,
    window_size_right: int,
):
    return _flash_mha_bwd_lowering(
        dout,
        q,
        k,
        v,
        out,
        lse,
        softmax_scale=softmax_scale,
        is_causal=is_causal,
        window_size_left=window_size_left,
        window_size_right=window_size_right,
    )


def is_replicated(sharding):
    if isinstance(sharding, NamedSharding):
        return sharding.is_fully_replicated
    raise ValueError(f"Unsupported sharding type: {type(sharding)}")


def partition_bwd(softmax_scale, is_causal, window_size_left, window_size_right,
                  mesh: Mesh,
                  arg_shapes: List[jax.ShapeDtypeStruct],
                  result_shape: List[jax.ShapeDtypeStruct]):
    result_shardings = tuple([x.sharding for x in result_shape])
    arg_shardings = tuple([x.sharding for x in arg_shapes])

    # arg_shardings: dout, q, k, v, out, lse
    dout_sharding = arg_shardings[0]
    q_sharding = arg_shardings[1]
    k_sharding = arg_shardings[2]
    v_sharding = arg_shardings[3]
    out_sharding = arg_shardings[4]
    lse_sharding = arg_shardings[5]

    # For simplicity, require all q-shaped tensors to have the same sharding
    assert dout_sharding == q_sharding == out_sharding, "dout, q, out must share the same sharding."
    assert k_sharding == v_sharding, "k, v must share the same sharding."

    if is_replicated(q_sharding):
        result_shardings = (
            NamedSharding(mesh, P()),  # dq
            NamedSharding(mesh, P()),  # dk
            NamedSharding(mesh, P()),  # dv
        )
    elif isinstance(q_sharding, NamedSharding):
        [n, s, h, d] = q_sharding.spec
        assert d is None, "Sharding across `d` won't be efficient, so it's not supported."
        assert s is None, "No ring attention yet"
        # dq has same shape as q: [n, sq, hq, d]
        # dk, dv have same shape as k, v: [n, sk, hk, d]
        # For GQA, hq > hk, and the sharding on h dimension needs adjustment
        [nk, sk, hk, dk] = k_sharding.spec
        result_shardings = (
            q_sharding,  # dq: [n, sq, hq, d]
            k_sharding,  # dk: [n, sk, hk, d]
            k_sharding,  # dv: [n, sk, hk, d]
        )
        # lse has shape [n, hq, sq], sharding should be P(n, h, s)
        lse_sharding = NamedSharding(mesh, P(n, h, s))
        arg_shardings = (q_sharding, q_sharding, k_sharding, k_sharding, q_sharding, lse_sharding)

    def bwd(dout, q, k, v, out, lse):
        return _flash_mha_bwd_lowering(
            dout, q, k, v, out, lse,
            softmax_scale=softmax_scale,
            is_causal=is_causal,
            window_size_left=window_size_left,
            window_size_right=window_size_right,
        )

    return mesh, bwd, result_shardings, arg_shardings


def sharding_rule_bwd(
    softmax_scale: Optional[float],
    is_causal: bool,
    window_size_left: int,
    window_size_right: int,
    mesh: Mesh,
    arg_shapes: List[ir.RankedTensorType],
    result_shape: List[ir.RankedTensorType],
):
    # arg_shapes: dout, q, k, v, out, lse
    # dout/q/out: [n, sq, hq, d], k/v: [n, sk, hk, d], lse: [n, hq, sq]
    # result_shapes: dq: [n, sq, hq, d], dk/dv: [n, sk, hk, d]
    q_shape = arg_shapes[1]
    k_shape = arg_shapes[2]
    group_size = q_shape.shape[-2] // k_shape.shape[-2]
    if group_size > 1:
        return SdyShardingRule(
            operand_mappings=(
                array_mapping("N S (H G) D"),   # dout
                array_mapping("N S (H G) D"),   # q
                array_mapping("N T H D"),       # k
                array_mapping("N T H D"),       # v
                array_mapping("N S (H G) D"),   # out
                array_mapping("N (H G) S"),     # lse
            ),
            result_mappings=(
                array_mapping("N S (H G) D"),   # dq
                array_mapping("N T H D"),       # dk
                array_mapping("N T H D"),       # dv
            ),
            G=group_size,
        )
    else:
        return SdyShardingRule(
            operand_mappings=(
                array_mapping("N S H D"),   # dout
                array_mapping("N S H D"),   # q
                array_mapping("N T H D"),   # k
                array_mapping("N T H D"),   # v
                array_mapping("N S H D"),   # out
                array_mapping("N H S"),     # lse
            ),
            result_mappings=(
                array_mapping("N S H D"),   # dq
                array_mapping("N T H D"),   # dk
                array_mapping("N T H D"),   # dv
            ),
        )


_flash_mha_bwd_lowering_sharded.def_partition(
    infer_sharding_from_operands=None,
    propagate_user_sharding=None,
    partition=partition_bwd,
    sharding_rule=sharding_rule_bwd,
)