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

from jax._src.ad_checkpoint import _optimization_barrier

def partition_fwd(softmax_scale, is_causal, window_size, mesh, arg_shapes, result_shape):
    result_shardings = jax.tree_map(lambda x: x.sharding, result_shape)
    arg_shardings = jax.tree_map(lambda x: x.sharding, arg_shapes)

    q_sharding = arg_shardings[0]
    if isinstance(q_sharding, PositionalSharding):
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
            return mesh, partial(ring_fwd, softmax_scale, is_causal, axis_name, axis_size), result_shardings, arg_shardings
        else:
            result_shardings = q_sharding, NamedSharding(mesh, P(n,h,l))
            arg_shardings = q_sharding, q_sharding, q_sharding
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
            return mesh, partial(ring_bwd, softmax_scale, is_causal, axis_name, axis_size), result_shardings, arg_shardings
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

# ==== Ring Forward ====

def ring_fwd(softmax_scale, is_causal, axis_name, axis_size, q,k,v):
    [n,l,h,d] = q.shape

    q_ix = jax.lax.axis_index(axis_name)
    k_ix = jax.lax.axis_index(axis_name)

    o = jnp.zeros([n,l,h,d], jnp.float32)
    lse = jnp.full([n,h,l], float('-inf'), jnp.float32)

    # scan :: (c -> a -> (c, b)) -> c -> [a] -> (c, [b])
    def f(c, a):
        (k, v, o, lse, k_ix) = c

        o1, lse1 = o, lse
        if is_causal:
            o2, lse2 = jax.lax.switch((k_ix < q_ix).astype(jnp.int32) + (k_ix <= q_ix).astype(jnp.int32),
                                    [
                                        lambda q,k,v: (jnp.zeros([n,l,h,d], q.dtype), jnp.full([n,h,l], float('-inf'), jnp.float32)),
                                        lambda q,k,v: _flash_mha_fwd_hlo(q,k,v, softmax_scale=softmax_scale, is_causal=True, window_size=(-1,-1)),
                                        lambda q,k,v: _flash_mha_fwd_hlo(q,k,v, softmax_scale=softmax_scale, is_causal=False, window_size=(-1,-1)),
                                    ], q, k, v)
        else:
            o2, lse2 = _flash_mha_fwd_hlo(q,k,v, softmax_scale=softmax_scale, is_causal=False, window_size=(-1,-1))
        o2 = o2.astype(jnp.float32)

        mx = jnp.maximum(lse1,lse2)
        mn = jnp.minimum(lse1,lse2)
        lse = jnp.log1p(jnp.exp(mn-mx)) + mx

        o = (o1 * rearrange(jnp.exp(lse1 - lse), 'n h l -> n l h 1') +
             o2 * rearrange(jnp.exp(lse2 - lse), 'n h l -> n l h 1'))
        
        k2 = jax.lax.ppermute(k, axis_name, [(i, (i+1)%axis_size) for i in range(axis_size)])
        v2 = jax.lax.ppermute(v, axis_name, [(i, (i+1)%axis_size) for i in range(axis_size)])
        k_ix = jax.lax.ppermute(k_ix, axis_name, [(i, (i+1)%axis_size) for i in range(axis_size)])

        return ((k2, v2, o, lse, k_ix), None)
    acc = (k,v,o,lse,k_ix)
    # We sadly have to manually unroll this because scan breaks the axis context preventing us from using ppermute (unroll=axis_size doesn't help either).
    # Optimization barrier prevents instruction reordering so that ppermute and flash_mha execute concurrently.
    for _ in range(axis_size):
        acc, _ = f(acc, None)
        acc = _optimization_barrier(acc)
    (_,_,o,lse,_) = acc
    # (_,_,o,lse), _ = jax.lax.scan(f,init,None,axis_size)
    return o.astype(q.dtype), lse

# ==== Ring Backward ===

# This doesn't seem like the most efficient way to do this, kind of wasting compute by calculating every dq,dk,dv twice.
# Should we send the accumulator for dk,dv cross-device instead? Relying on the fact that after a full cycle, they return to the starting device.
def ring_bwd(softmax_scale, is_causal, axis_name, axis_size, do,q,k,v,o,lse):
    [n,l,h,d] = q.shape

    ix = jax.lax.axis_index(axis_name)

    dq = jnp.zeros([n,l,h,d], jnp.float32)
    dk = jnp.zeros([n,l,h,d], jnp.float32)
    dv = jnp.zeros([n,l,h,d], jnp.float32)

    # scan :: (c -> a -> (c, b)) -> c -> [a] -> (c, [b])
    def f(acc, a):
        (do2,q2,k2,v2,o2,lse2,ix2, dq,dk,dv) = acc

        cmp = (ix2 < ix).astype(jnp.int32) + (ix2 <= ix).astype(jnp.int32) 
        # 0: ix < ix2
        # 1: ix = ix2
        # 2: ix > ix2
        if is_causal:
            dqa = jax.lax.switch(cmp, [
                                lambda q,k,v: jnp.zeros([n,l,h,d], q.dtype),
                                lambda q,k,v: _flash_mha_bwd_hlo(do,q,k2,v2,o,lse, softmax_scale=softmax_scale, is_causal=True, window_size=(-1,-1))[0],
                                lambda q,k,v: _flash_mha_bwd_hlo(do,q,k2,v2,o,lse, softmax_scale=softmax_scale, is_causal=False, window_size=(-1,-1))[0],
                            ], q, k, v)
            dka,dva = jax.lax.switch(cmp, [
                                lambda q,k,v: _flash_mha_bwd_hlo(do2,q2,k,v,o2,lse2, softmax_scale=softmax_scale, is_causal=False, window_size=(-1,-1))[1:],
                                lambda q,k,v: _flash_mha_bwd_hlo(do2,q2,k,v,o2,lse2, softmax_scale=softmax_scale, is_causal=True, window_size=(-1,-1))[1:],
                                lambda q,k,v: (jnp.zeros([n,l,h,d], q.dtype),jnp.zeros([n,l,h,d], q.dtype)),
                            ], q, k, v)
        else:
            dqa,_,_ = _flash_mha_bwd_hlo(do,q,k2,v2,o,lse, softmax_scale=softmax_scale, is_causal=False, window_size=(-1,-1))
            _,dka,dva = _flash_mha_bwd_hlo(do2,q2,k,v,o2,lse2, softmax_scale=softmax_scale, is_causal=False, window_size=(-1,-1))

        dq += dqa
        dk += dka
        dv += dva

        (do2,q2,k2,v2,o2,lse2,ix2) = jax.lax.ppermute((do2,q2,k2,v2,o2,lse2,ix2), axis_name, [(i, (i+1)%axis_size) for i in range(axis_size)])

        return ((do2,q2,k2,v2,o2,lse2,ix2, dq,dk,dv), None)
    acc = (do,q,k,v,o,lse,ix,dq,dk,dv)
    # Unrolled as above.
    for _ in range(axis_size):
        acc, _ = f(acc, None)
        acc = _optimization_barrier(acc)
    (do2,q2,k2,v2,o2,lse2,ix2, dq,dk,dv) = acc
    return dq.astype(q.dtype),dk.astype(q.dtype),dv.astype(q.dtype)