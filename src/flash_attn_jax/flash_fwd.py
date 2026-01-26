import math
import re
from functools import partial, wraps
from typing import List, Optional, Sequence

import einops
import jax
import jax._src.dispatch
import jax.numpy as jnp
import numpy as np
from einops import rearrange
from jax import Array, core, dtypes
from jax.core import ShapedArray
from jax.experimental.custom_partitioning import (
    ArrayMapping,
    CompoundFactor,
    SdyShardingRule,
    custom_partitioning,
)

from jax.extend.core import Primitive
from jax.interpreters import batching, mlir, xla
from jax.interpreters.mlir import ir
from jax.sharding import Mesh, NamedSharding
from jax.sharding import PartitionSpec as P

from flash_attn_jax.ring_attention import ring_fwd
from flash_attn_jax.util import num_splits_heuristic, round_multiple, array_mapping

# ==== Register primitives ====

_flash_mha_fwd_p = Primitive("flash_mha_fwd")
_flash_mha_fwd_p.multiple_results = True
_flash_mha_fwd_p.def_impl(partial(xla.apply_primitive, _flash_mha_fwd_p))
jax._src.dispatch.prim_requires_devices_during_lowering.add(_flash_mha_fwd_p)

# ==== Frontend ====

def flash_mha_fwd(q, k, v, *,
                  softmax_scale: Optional[float] = None, 
                  is_causal: bool = False,
                  window_size_left: int,
                  window_size_right: int):
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
    
    kwargs = dict(
        softmax_scale=softmax_scale,
        is_causal=is_causal,
        window_size_left=window_size_left,
        window_size_right=window_size_right,
    )
    return tuple(_flash_mha_fwd_p.bind(q, k, v, **kwargs))

# ==== HLO lowering ====

def _flash_mha_fwd_lowering(q, k, v, *, softmax_scale: float | None, is_causal: bool, window_size_left: int, window_size_right: int):
    #         // This needs to match with run_mha_fwd_splitkv_dispatch
    # const int block_n = head_size <= 64 ? 256 : (head_size <= 128 ? 128 : 64);
    # const int num_n_blocks = (max_seqlen_k + block_n - 1) / block_n;
    # // Technically kBlockM = 64 only for the splitKV kernels, not the standard kernel.
    # // In any case we don't expect seqlen_q to be larger than 64 for inference.
    # const int num_m_blocks = (max_seqlen_q + 64 - 1) / 64;
    [n, lq, hq, d] = q.shape
    [_, lk, hk, _] = k.shape
    dtype = q.dtype

    if softmax_scale is None:
        softmax_scale = 1.0 / math.sqrt(d)

    if d <= 64:
        block_n = 256
    elif d <= 128:
        block_n = 128
    else:
        block_n = 64
    num_n_blocks = max(1, (lk + block_n - 1) // block_n)
    num_m_blocks = max(1, (lq + 64 - 1) // 64)
    sm_count = 114 # H100
    num_splits = num_splits_heuristic(n * hq * num_m_blocks, sm_count, num_n_blocks, 128)
    lseaccum_shape = (num_splits, n, hq, lq)
    oaccum_shape = (num_splits, n, lq, hq, round_multiple(d, 32))

    dpad = (8 - d%8) % 8
    if dpad > 0:
        # We need padding. It's better to let xla's allocator handle it here than directly call cudaMalloc.
        q = jnp.pad(q, ((0,0),(0,0),(0,0),(0,dpad)), 'constant')
        k = jnp.pad(k, ((0,0),(0,0),(0,0),(0,dpad)), 'constant')
        v = jnp.pad(v, ((0,0),(0,0),(0,0),(0,dpad)), 'constant')
    
    o_shape = [n, lq, hq, d+dpad]
    lse_shape = [n, hq, lq]

    out_types = [jax.ShapeDtypeStruct(o_shape, dtype), 
                    jax.ShapeDtypeStruct(lse_shape, jnp.float32),
                    jax.ShapeDtypeStruct(oaccum_shape, jnp.float32),
                    jax.ShapeDtypeStruct(lseaccum_shape, jnp.float32),
                    ]

    o, lse = jax.ffi.ffi_call(
        "flash_mha_fwd",
        result_shape_dtypes=out_types,
        has_side_effect=False,
        input_layouts=[None, None, None], # default row major
        output_layouts=[None, None, None, None],
        )(q, k, v, softmax_scale=softmax_scale,
        is_causal=is_causal,
        window_size_left=window_size_left,
        window_size_right=window_size_right,
        )[:2]

    if dpad > 0:
        o = o[:,:,:,:d]
    return o, lse


def _flash_mha_fwd_lowering_mlir(ctx, q, k, v, **keywords):
    return mlir.lower_fun(
        partial(_flash_mha_fwd_lowering_sharded, **keywords), multiple_results=True
    )(ctx, q, k, v)


mlir.register_lowering(
    _flash_mha_fwd_p,
    _flash_mha_fwd_lowering_mlir,  # type: ignore
    platform="gpu",
)

# ==== Abstract Evaluation ====

def _flash_mha_fwd_abstract(q, k, v, **keywords):
    q_dtype = dtypes.canonicalize_dtype(q.dtype)
    [n, sq, hq, d] = q.shape
    out_shape = [n, sq, hq, d]
    lse_shape = [n, hq, sq]    
    return (
        ShapedArray(out_shape, q_dtype),
        ShapedArray(lse_shape, jnp.float32)
    )
_flash_mha_fwd_p.def_abstract_eval(_flash_mha_fwd_abstract)

# ==== VMap rules ====

def mha_fwd_batch(vector_arg_values: Sequence[Array], batch_axes, **kwargs):
  assert all(isinstance(b, int) or b is None for b in batch_axes)
  vector_arg_values, batch_axes = zip(*[(jnp.moveaxis(x, b, 0), 0) if b is not None else (x, b) for x, b in zip(vector_arg_values, batch_axes)])
  mapped = tuple(isinstance(b, int) for b in batch_axes)
  q, k, v = vector_arg_values
  if mapped == (True, True, True):
    x, n, sq, hq, d = q.shape
    x, n, sk, hk, d = k.shape
    out, lse = flash_mha_fwd(q.reshape((x*n, sq, hq, d)), 
                            k.reshape((x*n, sk, hk, d)), 
                            v.reshape((x*n, sk, hk, d)), 
                            **kwargs)
    out = out.reshape((x, n, sq, hq, d))
    lse = lse.reshape((x, n, hq, sq))
    return (out, lse), (0,0)
  elif mapped == (True, False, False):
    # This is just a GQA!
    x, n, sq, hq, d = q.shape
    n, sk, hk, d = k.shape
    q = einops.rearrange(q, 'x n sq hq d -> n sq (hq x) d')
    out, lse = flash_mha_fwd(q, k, v, **kwargs)
    out = einops.rearrange(out, 'n l (h x) d -> x n l h d', x=x)
    lse = einops.rearrange(lse, 'n (h x) l -> x n h l', x=x)
    return (out, lse), (0,0)
  else:
    raise NotImplementedError("MHA fwd only support vmapping over q or (q,k,v) for now, got batch axes " + str(batch_axes))
batching.primitive_batchers[_flash_mha_fwd_p] = mha_fwd_batch

# ==== Sharding ====
@partial(custom_partitioning, static_argnums=(3, 4, 5, 6))
def _flash_mha_fwd_lowering_sharded(
    q,
    k,
    v,
    softmax_scale: Optional[float],
    is_causal: bool,
    window_size_left: int,
    window_size_right: int,
):
    return _flash_mha_fwd_lowering(
        q,
        k,
        v,
        softmax_scale=softmax_scale,
        is_causal=is_causal,
        window_size_left=window_size_left,
        window_size_right=window_size_right,
    )


def is_replicated(sharding):
    if isinstance(sharding, NamedSharding):
        return sharding.is_fully_replicated
    raise ValueError(f"Unsupported sharding type: {type(sharding)}")

def partition_fwd(softmax_scale, is_causal, window_size_left, window_size_right,
                  mesh: Mesh,
                  arg_shapes: List[jax.ShapeDtypeStruct],
                  result_shape: List[jax.ShapeDtypeStruct]):
    result_shardings = tuple([x.sharding for x in result_shape])
    arg_shardings = tuple([x.sharding for x in arg_shapes])

    q_sharding = arg_shardings[0]
    k_sharding = arg_shardings[1]
    v_sharding = arg_shardings[2]
    assert q_sharding == k_sharding and q_sharding == v_sharding, "Only support q, k, v sharing the same sharding."
    if is_replicated(q_sharding):
        result_shardings = (NamedSharding(mesh, P()), NamedSharding(mesh, P()))
    elif isinstance(q_sharding, NamedSharding):
        [n,s,h,d] = q_sharding.spec
        assert d is None, "Sharding across `d` won't be efficient, so it's not supported."
        assert s is None, "No ring attention yet"
        result_shardings = q_sharding, NamedSharding(mesh, P(n,h,s))
        arg_shardings = q_sharding, q_sharding, q_sharding
    def fwd(q,k,v):
        return _flash_mha_fwd_lowering(q,k,v, softmax_scale=softmax_scale, is_causal=is_causal, window_size_left=window_size_left, window_size_right=window_size_right)
    return mesh, fwd, result_shardings, arg_shardings

def sharding_rule_fwd(
    softmax_scale: Optional[float],
    is_causal: bool,
    window_size_left: int,
    window_size_right: int,
    mesh: Mesh,
    arg_shapes: List[ir.RankedTensorType],
    result_shape: List[ir.RankedTensorType],
):
    q_shape, k_shape, v_shape = arg_shapes
    group_size = q_shape.shape[-2] // k_shape.shape[-2]
    if group_size > 1:
        return SdyShardingRule(
            operand_mappings=(
                array_mapping("N S (H G) D"),
                array_mapping("N T H D"),
                array_mapping("N T H D"),
            ),
            result_mappings=(array_mapping("N S (H G) D"), array_mapping("N (H G) S")),
            G=group_size,
        )
    else:
        return SdyShardingRule(
            operand_mappings=(
                array_mapping("N S H D"),
                array_mapping("N T H D"),
                array_mapping("N T H D"),
            ),
            result_mappings=(array_mapping("N S H D"), array_mapping("N H S")),
        )

_flash_mha_fwd_lowering_sharded.def_partition(
    infer_sharding_from_operands=None,
    propagate_user_sharding=None,
    partition=partition_fwd,
    sharding_rule = sharding_rule_fwd,
    )