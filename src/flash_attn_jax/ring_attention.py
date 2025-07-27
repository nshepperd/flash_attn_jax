from functools import partial, wraps

import numpy as np
import jax
import jax.numpy as jnp

from einops import rearrange
import math

# ==== Ring Forward ====

def ring_fwd(q,k,v, axis_name, axis_size, mha_fwd, softmax_scale=None, is_causal=False):
    [n,l,h,d] = q.shape
    if softmax_scale is None:
        softmax_scale = 1/math.sqrt(d)

    q_ix = jax.lax.axis_index(axis_name)
    k_ix = jax.lax.axis_index(axis_name)

    o = jnp.zeros([n,l,h,d], jnp.float32)
    lse = jnp.full([n,h,l], float('-inf'), jnp.float32)

    # scan :: (c -> a -> (c, b)) -> c -> [a] -> (c, [b])
    def f(c, a):
        (k, v, o, lse, k_ix) = c

        o1, lse1 = o, lse
        if is_causal:
            cmp = (k_ix < q_ix).astype(jnp.int32) + (k_ix <= q_ix).astype(jnp.int32) 
            o2, lse2 = jax.lax.switch(cmp,
                                    [
                                        lambda: (jnp.zeros([n,l,h,d], q.dtype), jnp.full([n,h,l], float('-inf'), jnp.float32)),
                                        lambda: mha_fwd(q,k,v, softmax_scale=softmax_scale, is_causal=True, window_size=(-1,-1)),
                                        lambda: mha_fwd(q,k,v, softmax_scale=softmax_scale, is_causal=False, window_size=(-1,-1)),
                                    ])
        else:
            o2, lse2 = mha_fwd(q,k,v, softmax_scale=softmax_scale, is_causal=False, window_size=(-1,-1))
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
    # Manually unroll this until https://github.com/google/jax/pull/20884 is merged.
    # Optimization barrier prevents instruction reordering across loop iters, so that
    # ppermute and flash_mha execute concurrently (though this is unreliable).
    # for _ in range(axis_size):
    #     acc, _ = f(acc, None)
    #     # acc = _optimization_barrier(acc)
    # (_,_,o,lse,_) = acc
    (_,_,o,lse,_), _ = jax.lax.scan(f,acc,None,axis_size)
    return o.astype(q.dtype), lse

# ==== Ring Backward ===

def ring_bwd(do,q,k,v,o,lse, axis_name, axis_size, mha_bwd, softmax_scale=None, is_causal=False):
    [n,l,h,d] = q.shape
    [n,lk,hk,d] = k.shape
    if softmax_scale is None:
        softmax_scale = 1/math.sqrt(d)

    ix = jax.lax.axis_index(axis_name)

    dq = jnp.zeros([n,l,h,d], jnp.float32)
    dk = jnp.zeros([n,lk,hk,d], jnp.float32)
    dv = jnp.zeros([n,lk,hk,d], jnp.float32)

    # scan :: (c -> a -> (c, b)) -> c -> [a] -> (c, [b])
    def f(acc, _):
        (k2,v2,dk2,dv2,ix2, dq) = acc

        cmp = (ix2 < ix).astype(jnp.int32) + (ix2 <= ix).astype(jnp.int32) 
        # 0: ix < ix2
        # 1: ix = ix2
        # 2: ix > ix2
        def skip():
            return (jnp.zeros(q.shape, q.dtype), jnp.zeros(k.shape, k.dtype), jnp.zeros(v.shape, v.dtype))
        def causal():
            return mha_bwd(do,q,k2,v2,o,lse, softmax_scale=softmax_scale, is_causal=True, window_size=(-1,-1))
        def non_causal():
            return mha_bwd(do,q,k2,v2,o,lse, softmax_scale=softmax_scale, is_causal=False, window_size=(-1,-1))

        (dk2_,dv2_) = jax.lax.ppermute((dk2,dv2), axis_name, [(i, (i+1)%axis_size) for i in range(axis_size)])
        (k2_,v2_,ix2_) = jax.lax.ppermute((k2,v2,ix2), axis_name, [(i, (i+1)%axis_size) for i in range(axis_size)])
        
        if is_causal:
            (dqa, dka, dva) = jax.lax.switch(cmp, [skip, causal, non_causal])
        else:
            (dqa, dka, dva) = non_causal()

        # Send/receive of dk/dv retires here (because the following depends on it).
        if is_causal:
            (dq, dk2_, dv2_) = jax.lax.switch(cmp, [
                lambda: (dq, dk2_, dv2_),
                lambda: (dq+dqa, dk2_+dka, dv2_+dva),
                lambda: (dq+dqa, dk2_+dka, dv2_+dva)
            ])
        else:
            dq, dk2_, dv2_ = (dq+dqa, dk2_+dka, dv2_+dva)
        
        return ((k2_,v2_,dk2_,dv2_,ix2_, dq), None)
    acc = (k,v,dk,dv,ix, dq)
    # See above (#20884).
    # for _ in range(axis_size):
    #     acc, _ = f(acc, None)
    #     # acc = _optimization_barrier(acc)
    acc, _ = jax.lax.scan(f,acc,None,axis_size)
    (k,v,dk,dv,ix2, dq) = acc
    (dk,dv) = jax.lax.ppermute((dk,dv), axis_name, [(i, (i+1)%axis_size) for i in range(axis_size)])
    return dq.astype(q.dtype),dk.astype(q.dtype),dv.astype(q.dtype)