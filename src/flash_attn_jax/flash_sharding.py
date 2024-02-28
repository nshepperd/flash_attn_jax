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

# ==== Sharding ====

_flash_mha_fwd_hlo_sharded = custom_partitioning(_flash_mha_fwd_hlo, static_argnums=(3,4,5))
_flash_mha_bwd_hlo_sharded = custom_partitioning(_flash_mha_bwd_hlo, static_argnums=(6,7,8))

def partition_fwd(softmax_scale, is_causal, window_size, mesh, arg_shapes, result_shape):
    result_shardings = jax.tree_map(lambda x: x.sharding, result_shape)
    arg_shardings = jax.tree_map(lambda x: x.sharding, arg_shapes)

    q_sharding = arg_shardings[0]
    if isinstance(q_sharding, PositionalSharding):
        if not is_causal and window_size == (-1,-1):
            # We can handle Q that's sharded across the L dimension
            # without replicating Q by executing it as a cross
            # attention:
            #
            #  q : n [L/devices] h d
            #  kv : n L h d
            #  -> o : n [L/devices] h d
            #
            # TODO: We could handle q sharded across L even with
            # causal/local if we could communicate the slice offset
            # (of q in kv) to the c++ driver. But it's unclear how to
            # do that since the HLO has to be identical (SPMD).
            q_sharding = q_sharding.replicate(3)
            kv_sharding = q_sharding.replicate(1)
            (n,l,h,d) = q_sharding.shape
            result_shardings = q_sharding, q_sharding.reshape((n,l,h)).transpose(0,2,1) # n h l
            arg_shardings = q_sharding, kv_sharding, kv_sharding
        else:
            # We need to replicate d always.
            q_sharding = q_sharding.replicate((1,3))
            (n,l,h,d) = q_sharding.shape # l=1, d=1
            result_shardings = q_sharding, q_sharding.reshape((n,l,h)).transpose(0,2,1)
            arg_shardings = q_sharding, q_sharding, q_sharding
    elif isinstance(q_sharding, NamedSharding):
        mesh = q_sharding.mesh
        [n,l,h,d] = q_sharding.spec
        if not is_causal and window_size == (-1,-1):
            q_sharding = NamedSharding(mesh, P(n,l,h,None))
            kv_sharding = NamedSharding(mesh, P(n,None,h,None))
            lse_sharding = NamedSharding(mesh, P(n,h,l))
        else:
            q_sharding = NamedSharding(mesh, P(n,None,h,None))
            kv_sharding = q_sharding
            lse_sharding = NamedSharding(mesh, P(n,h,None))
        result_sharding = (q_sharding, lse_sharding)
        arg_shardings = (q_sharding, kv_sharding, kv_sharding)
    def fwd(q,k,v):
        return _flash_mha_fwd_hlo(q,k,v, softmax_scale=softmax_scale, is_causal=is_causal, window_size=window_size)
    return mesh, fwd, result_shardings, arg_shardings

def infer_sharding_fwd(softmax_scale, is_causal, window_size, mesh, arg_shapes, result_shape):
    arg_shardings = jax.tree_map(lambda x: x.sharding, arg_shapes)
    q_sharding = arg_shardings[0]
    if isinstance(q_sharding, PositionalSharding):
        [n,l,h,d] = q_sharding.shape
        result_sharding = (q_sharding, # [n,l,h,d]
                           q_sharding.replicate(3).reshape(n,l,h).transpose((0,2,1)) # [n,h,l]
                           )
    elif isinstance(q_sharding, NamedSharding):
        [n,l,h,d] = q_sharding.spec
        result_sharding = (q_sharding,
                           NamedSharding(q_sharding.mesh, P(n,h,l)))
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
    if isinstance(q_sharding, PositionalSharding):
        do_sharding = q_sharding.replicate((1,3))
        [n, l, h, d] = do_sharding.shape
        lse_sharding = do_sharding.reshape(n,l,h).transpose(0,2,1) # n h l
        result_shardings = (do_sharding,)*3
        arg_shardings = (do_sharding,)*5 + (lse_sharding,)
    elif isinstance(q_sharding, NamedSharding):
        mesh = q_sharding.mesh
        [n,l,h,d] = q_sharding.spec
        do_sharding = NamedSharding(mesh, P(n,None,h,None))
        lse_sharding = NamedSharding(mesh, P(n,h,None))
        result_shardings = (do_sharding,)*3
    def fwd(*args):
        return _flash_mha_bwd_hlo(*args, softmax_scale=softmax_scale, is_causal=is_causal, window_size=window_size)
    return mesh, fwd, result_shardings, arg_shardings

_flash_mha_bwd_hlo_sharded.def_partition(
    infer_sharding_from_operands=infer_sharding_bwd,
    partition=partition_bwd)
