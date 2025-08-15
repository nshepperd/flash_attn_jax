import glob
import sys, os

import jax
import jax.numpy as jnp
import numpy as np
from functools import partial
import einops

def make_mask(R, C, is_causal, window_size):
    q_idx = jnp.arange(R)[:, None]-R
    k_idx = jnp.arange(C)[None, :]-C
    mask = jnp.ones([R,C], dtype=jnp.int32)
    if is_causal:
        mask &= q_idx >= k_idx
    if window_size[0] != -1:
        mask &= k_idx >= q_idx - window_size[0] #jnp.triu(mask, -window_size[0])
    if window_size[1] != -1:
        # mask = jnp.tril(mask, window_size[1])
        mask &= k_idx <= q_idx + window_size[1]
    return mask

def ref_mha(q,k,v, is_causal=False, window_size=(-1,-1), softmax_scale=None):
    return ref_fwd(q,k,v, is_causal=is_causal, window_size=window_size, softmax_scale=softmax_scale)[0]

def ref_fwd(q,k,v, is_causal=False, window_size=(-1,-1), softmax_scale=None):
    [n, l, h, d] = q.shape
    [n, lk, hk, d] = k.shape
    if softmax_scale is None:
        softmax_scale = 1/np.sqrt(d)
    mask = make_mask(l,lk,is_causal,window_size)
    if h != hk:
        assert h > hk and h % hk == 0
        q = einops.rearrange(q, 'n L (h x) d -> n L h x d', h=hk)
        S = jnp.einsum('nlhxd,nLhd->nhxlL',q,k) * softmax_scale
        S = jnp.where(mask, S, float('-inf'))
        lse = jax.nn.logsumexp(S, axis=-1) #nhxl
        P = jnp.exp(S - lse[...,None]) # n h l L
        o = jnp.einsum('nhxlL,nLhd->nlhxd',P,v)
        o = einops.rearrange(o, 'n l h x d -> n l (h x) d')
        lse = einops.rearrange(lse, 'n h x l -> n (h x) l')
        return o.astype(q.dtype), lse.astype(jnp.float32)
    else:
        S = jnp.einsum('nlhd,nLhd->nhlL',q,k)
        S = jnp.where(mask, S, float('-inf'))
        lse = jax.nn.logsumexp(S*softmax_scale, axis=-1) #nhl
        P = jax.nn.softmax(S*softmax_scale, axis=-1) #jnp.exp(att - lse[...,None])
        o = jnp.einsum('nhlL,nLhd->nlhd',P,v)
        return o.astype(q.dtype), lse.astype(jnp.float32)

def ref_bwd(do,q,k,v,o,lse, is_causal=False, window_size=(-1,-1), softmax_scale=None):
    [n, l, h, d] = q.shape
    [n, lk, hk, d] = k.shape
    if softmax_scale is None:
        softmax_scale = 1/np.sqrt(d)
    mask = make_mask(l,lk,is_causal,window_size)
    if h != hk:
        assert h > hk and h % hk == 0
        q = einops.rearrange(q, 'n l (h x) d -> n l h x d', h=hk)
        lse = einops.rearrange(lse, 'n (h x) l -> n h x l', h=hk)
        S = jnp.einsum('nlhxd,nLhd->nhxlL',q,k) * softmax_scale
        D = einops.reduce(do * o, 'n l (h x) d -> n h x l', reduction='sum', h=hk)
        do = einops.rearrange(do, 'n l (h x) d -> n l h x d', h=hk)
        S = jnp.where(mask, S, float('-inf'))
        P = jnp.exp(S - lse[...,None]) # n h x l L
        dP = jnp.einsum('nlhxd,nLhd->nhxlL',do,v)
        dv = jnp.einsum('nlhxd,nhxlL->nLhd',do,P)
        dS = P * (dP - D[...,None])
        dq = softmax_scale*jnp.einsum('nLhd,nhxlL->nlhxd',k,dS)
        dk = softmax_scale*jnp.einsum('nlhxd,nhxlL->nLhd',q,dS)
        dq = einops.rearrange(dq, 'n l h x d -> n l (h x) d')
        return dq.astype(q.dtype),dk.astype(q.dtype),dv.astype(q.dtype)
    else:
        S = jnp.einsum('nlhd,nLhd->nhlL',q,k)*softmax_scale
        D = einops.reduce(do * o, 'n l h d -> n h l', reduction='sum')
        S = jnp.where(mask, S, float('-inf'))
        P = jnp.exp(S - lse[...,None]) # n h l L
        dP = jnp.einsum('nlhd,nLhd->nhlL',do,v)
        dv = jnp.einsum('nlhd,nhlL->nLhd',do,P)
        dS = P * (dP - D[...,None])
        dq = softmax_scale*jnp.einsum('nLhd,nhlL->nlhd',k,dS)
        dk = softmax_scale*jnp.einsum('nlhd,nhlL->nLhd',q,dS)
        return dq.astype(q.dtype),dk.astype(q.dtype),dv.astype(q.dtype)