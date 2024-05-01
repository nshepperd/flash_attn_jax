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

from .flash_hlo import _flash_mha_fwd_hlo, _flash_mha_bwd_hlo
from .ring_attention import ring_fwd, ring_bwd

# ==== Sharding ====

_flash_mha_fwd_hlo_sharded = custom_partitioning(_flash_mha_fwd_hlo, static_argnums=(3,4,5))
_flash_mha_bwd_hlo_sharded = custom_partitioning(_flash_mha_bwd_hlo, static_argnums=(6,7,8))

from jax._src.ad_checkpoint import _optimization_barrier

def is_replicated(sharding):
    return (isinstance(sharding, PositionalSharding) and sharding.shape == (1,)) or (isinstance(sharding, NamedSharding) and len(sharding.spec) == 0)

def partition_fwd(softmax_scale, is_causal, window_size, mesh, arg_shapes, result_shape):
    result_shardings = jax.tree_map(lambda x: x.sharding, result_shape)
    arg_shardings = jax.tree_map(lambda x: x.sharding, arg_shapes)

    q_sharding = arg_shardings[0]
    k_sharding = arg_shardings[1]
    v_sharding = arg_shardings[2]
    assert q_sharding == k_sharding and q_sharding == v_sharding, "Only support q, k, v sharing the same sharding."
    if is_replicated(q_sharding):
        result_sharding = (q_sharding, q_sharding)
    elif isinstance(q_sharding, PositionalSharding):
        (n,l,h,d) = q_sharding.shape
        assert d == 1, "Sharding across `d` won't be efficient, so it's not supported."
        assert l == 1, "For ring attention, use `with Mesh(...) as mesh` and NamedSharding."
        result_shardings = q_sharding, q_sharding.reshape((n,h,1)) # n h l
        arg_shardings = q_sharding, q_sharding, q_sharding
    elif isinstance(q_sharding, NamedSharding):
        mesh = q_sharding.mesh
        [n,l,h,d] = q_sharding.spec
        assert d == None, "Sharding across `d` won't be efficient, so it's not supported."
        if l != None:
            # assert not is_causal and window_size == (-1,-1), "Ring attention doesn't support causal or local masking yet."
            assert window_size == (-1,-1), "Ring attention doesn't support local masking yet."
            result_shardings = q_sharding, NamedSharding(mesh, P(n,h,l))
            arg_shardings = q_sharding, q_sharding, q_sharding
            axis_name = l
            axis_size = mesh.shape[axis_name]
            # ring attention
            return mesh, partial(ring_fwd, softmax_scale=softmax_scale, is_causal=is_causal, axis_name=axis_name, axis_size=axis_size, mha_fwd=_flash_mha_fwd_hlo), result_shardings, arg_shardings
        else:
            result_shardings = q_sharding, NamedSharding(mesh, P(n,h,l))
            arg_shardings = q_sharding, q_sharding, q_sharding
    def fwd(q,k,v):
        return _flash_mha_fwd_hlo(q,k,v, softmax_scale=softmax_scale, is_causal=is_causal, window_size=window_size)
    return mesh, fwd, result_shardings, arg_shardings

def infer_sharding_fwd(softmax_scale, is_causal, window_size, mesh, arg_shapes, result_shape):
    arg_shardings = jax.tree_map(lambda x: x.sharding, arg_shapes)
    q_sharding = arg_shardings[0]
    k_sharding = arg_shardings[1]
    v_sharding = arg_shardings[2]
    assert q_sharding == k_sharding and q_sharding == v_sharding, "Only support q, k, v sharing the same sharding."
    if is_replicated(q_sharding):
        result_sharding = (q_sharding, q_sharding)
    elif isinstance(q_sharding, PositionalSharding):
        [n,l,h,d] = q_sharding.shape
        result_sharding = (q_sharding, # [n,l,h,d]
                           q_sharding.replicate(3).reshape(n,l,h).transpose((0,2,1)) # [n,h,l]
                           )
    elif isinstance(q_sharding, NamedSharding):
        [n,l,h,d] = q_sharding.spec
        result_sharding = (q_sharding,
                           NamedSharding(q_sharding.mesh, P(n,h,l)))
    else:
        raise ValueError("Unsupported sharding type.", type(q_sharding))
    return result_sharding

_flash_mha_fwd_hlo_sharded.def_partition(
    infer_sharding_from_operands=infer_sharding_fwd,
    partition=partition_fwd)

def infer_sharding_bwd(softmax_scale, is_causal, window_size, mesh, arg_shapes, result_shape):
    # args: dout, q, k, v, out, lse
    # outs: dq, dk, dv
    # i think generally we want the output sharding for dq,dk,dv to be the same as q,k,v?
    arg_shardings = jax.tree_map(lambda x: x.sharding, arg_shapes)
    q_sharding = arg_shardings[1]
    k_sharding = arg_shardings[2]
    v_sharding = arg_shardings[3]
    return q_sharding, k_sharding, v_sharding

def partition_bwd(softmax_scale, is_causal, window_size, mesh, arg_shapes, result_shape):
    result_shardings = jax.tree_map(lambda x: x.sharding, result_shape)
    arg_shardings = jax.tree_map(lambda x: x.sharding, arg_shapes)

    do_sharding = arg_shardings[0]
    q_sharding = arg_shardings[1]
    k_sharding = arg_shardings[2]
    v_sharding = arg_shardings[3]
    o_sharding = arg_shardings[4]
    lse_sharding = arg_shardings[5]
    assert q_sharding == k_sharding and q_sharding == v_sharding, "Only support q, k, v sharing the same sharding."
    if is_replicated(q_sharding):
        result_shardings = (q_sharding,)*3
    elif isinstance(q_sharding, PositionalSharding):
        assert q_sharding == k_sharding, "Expect q and k sharding to match"
        assert q_sharding == v_sharding, "Expect q and v sharding to match"
        [n, l, h, d] = q_sharding.shape
        assert d == 1, "Sharding across `d` won't be efficient, so it's not supported."
        assert l == 1, "For ring attention, use `with Mesh(...) as mesh` and NamedSharding."
        lse_sharding = q_sharding.reshape(n,h,1) # n h l
        result_shardings = (q_sharding,)*3
        arg_shardings = (q_sharding,)*5 + (lse_sharding,)
    elif isinstance(q_sharding, NamedSharding):
        mesh = q_sharding.mesh
        [n,l,h,d] = q_sharding.spec
        assert d == None, "Sharding across `d` won't be efficient, so it's not supported."
        if l != None:
            # assert not is_causal and window_size == (-1,-1), "Ring attention doesn't support causal or local masking yet."
            assert window_size == (-1,-1), "Ring attention doesn't support local masking yet."
            result_shardings = q_sharding, q_sharding, q_sharding
            lse_sharding = NamedSharding(mesh, P(n,h,l))
            arg_shardings = (q_sharding,)*5 + (lse_sharding,)
            axis_name = l
            axis_size = mesh.shape[axis_name]
            # ring attention
            return mesh, partial(ring_bwd, softmax_scale=softmax_scale, is_causal=is_causal, axis_name=axis_name, axis_size=axis_size, mha_bwd=_flash_mha_bwd_hlo), result_shardings, arg_shardings
        else:
            result_shardings = q_sharding, q_sharding, q_sharding
            lse_sharding = NamedSharding(mesh, P(n,h,l))
            arg_shardings = (q_sharding,)*5 + (lse_sharding,)
    def fwd(*args):
        return _flash_mha_bwd_hlo(*args, softmax_scale=softmax_scale, is_causal=is_causal, window_size=window_size)
    return mesh, fwd, result_shardings, arg_shardings

_flash_mha_bwd_hlo_sharded.def_partition(
    infer_sharding_from_operands=infer_sharding_bwd,
    partition=partition_bwd)
