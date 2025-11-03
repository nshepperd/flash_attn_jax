import glob
import os
import sys

sys.path.insert(0,'./src')

import math
from functools import partial

import einops
import jax
import jax.numpy as jnp
import numpy as np
import pytest
from jax.tree_util import tree_map

from flash_attn_jax import flash_mha
from flash_attn_jax.varlen import flash_mha_varlen

from .ref_mha import ref_mha

jax.config.update("jax_default_matmul_precision", "highest")

def pretty(tensor):
    shape = tensor.shape
    mx = jnp.max(tensor)
    mn = jnp.min(tensor)
    mean = jnp.mean(tensor)
    std = jnp.std(tensor)
    return f'[{shape}: {mn:.3g} | {mean:.3g}±{std:.3g} | {mx:.3g}]'

def check(ref_out, jax_out, out, margin=4):
    def check1(ref_out, jax_out, out):
        atol = margin * jnp.max(jnp.abs(jax_out - ref_out)).item()
        rtol = 1e-3
        np.testing.assert_allclose(out, ref_out, rtol=rtol, atol=atol)
        # assert jnp.max(jnp.abs(out - ref_out)).item() <= margin * jnp.max(jnp.abs(jax_out - ref_out)).item(), (pretty(jnp.abs(out - ref_out)), 'vs', pretty(jnp.abs(jax_out - ref_out)))
    tree_map(check1, ref_out, jax_out, out)


def ref_mha_varlen(q, k, v, seqlens_q, seqlens_k, seqused_k=None, *, is_causal=False, window_size=(-1,-1), **kwargs):
    [sq, hq, cq] = q.shape
    [sk, hk, ck] = k.shape
    assert k.shape == v.shape
    assert cq == ck
    if hq != hk:
        assert hq > hk and hq % hk == 0
        m = hq // hk
        q = einops.rearrange(q, 'sq (h m) c -> m sq h c', m=m)
        out = jax.vmap(lambda q_: ref_mha_varlen(q_, k, v, seqlens_q, seqlens_k, seqused_k, is_causal=is_causal, window_size=window_size, **kwargs), in_axes=0)(q)
        out = einops.rearrange(out, 'm sq h c -> sq (h m) c')
        return out
    [b_plus_1] = seqlens_q.shape
    coord_q = jnp.broadcast_to(jnp.arange(sq)[:,None], (sq, sk))
    coord_k = jnp.broadcast_to(jnp.arange(sk), (sq, sk))
    q_starts, q_ends = seqlens_q[:-1], seqlens_q[1:]
    k_starts, k_ends = seqlens_k[:-1], seqlens_k[1:]
    if seqused_k is not None:
        k_ends = jnp.minimum(k_ends, k_starts + seqused_k)
    mask_q = jax.vmap(
        lambda q_start, q_end, k_start, k_end: (
            (coord_q >= q_start)
            & (coord_q < q_end)
            & (coord_k >= k_start)
            & (coord_k < k_end)
        )
    )(q_starts, q_ends, k_starts, k_ends)
    mask_q = mask_q.any(axis=0)
    if is_causal:
        mask_q = jnp.logical_and(mask_q, coord_k <= coord_q)
    if window_size[0] >= 0:
        mask_q = jnp.logical_and(mask_q, coord_k >= (coord_q - window_size[0]))
    if window_size[1] >= 0:
        mask_q = jnp.logical_and(mask_q, coord_k <= (coord_q + window_size[1]))
    
    attn = jnp.einsum('qhd,khd->hqk', q.astype(jnp.float32), k.astype(jnp.float32)) / math.sqrt(cq)
    attn = jnp.where(mask_q[None, :, :], attn, float('-inf'))
    attn = jax.nn.softmax(attn, axis=-1)
    out = jnp.einsum('hqk,khd->qhd', attn, v.astype(jnp.float32))
    return out.astype(q.dtype)

@pytest.mark.parametrize("seqused_k_limit", [None, 4])
@pytest.mark.parametrize("dtype", [jnp.float16, jnp.bfloat16])
@pytest.mark.parametrize("local", ['local',''])
@pytest.mark.parametrize("causal", ['causal',''])
@pytest.mark.parametrize("d", [59, 32])
@pytest.mark.parametrize("h", [1, 4])
@pytest.mark.parametrize("m", [1, 2]) # for MQA/GQA
def test_varlen_flash_fwd(m, h, d, causal, local, dtype, seqused_k_limit):
    window_size = (3,3) if local else (-1,-1)
    lens = [1, 2, 0, 6, 10]
    b = len(lens)
    total_seqlen = sum(lens)

    if seqused_k_limit is not None and (causal or local):
        return # skip causal/local tests with seqused_k_limit

    fenceposts = jnp.cumsum(jnp.array([0] + lens), dtype=jnp.int32)

    q = jax.random.normal(jax.random.PRNGKey(0), [total_seqlen, h*m, d], dtype=jnp.float32)
    k = jax.random.normal(jax.random.PRNGKey(1), [total_seqlen, h, d], dtype=jnp.float32)
    v = jax.random.normal(jax.random.PRNGKey(2), [total_seqlen, h, d], dtype=jnp.float32)

    seqused_k = None
    if seqused_k_limit is not None:
        seqused_k = jnp.array([min(l, seqused_k_limit) for l in lens])

    def ref(q,k,v):
        out = ref_mha_varlen(q,k,v, seqlens_q = fenceposts, seqlens_k = fenceposts, 
                           seqused_k=seqused_k,
                           max_seqlen_q=max(lens), max_seqlen_k=max(lens),
                            is_causal=bool(causal), window_size=window_size)
        # out = jnp.zeros([total_seqlen, h*m, d], dtype=jnp.float32)
        # for i in range(b):
        #     bq = q[None,fenceposts[i]:fenceposts[i+1]]
        #     bk = k[None,fenceposts[i]:fenceposts[i+1]]
        #     bv = v[None,fenceposts[i]:fenceposts[i+1]]
        #     if seqused_k_limit is not None:
        #         bk = bk[:, :seqused_k_limit, :]
        #         bv = bv[:, :seqused_k_limit, :]
        #     out = out.at[None,fenceposts[i]:fenceposts[i+1]].set(ref_mha(bq, bk, bv, is_causal=bool(causal), window_size=window_size))
        return out

    ref_out = ref(q,k,v)
    q = q.astype(dtype)
    k = k.astype(dtype)
    v = v.astype(dtype)
    jax_out = ref(q,k,v)

    out = flash_mha_varlen(q,k,v, seqlens_q = fenceposts, seqlens_k = fenceposts, 
                           seqused_k=seqused_k,
                           max_seqlen_q=max(lens), max_seqlen_k=max(lens),
                            is_causal=bool(causal), window_size=window_size)
    check(ref_out, jax_out, out)
    

@pytest.mark.parametrize("seqused_k_limit", [None, 4])
@pytest.mark.parametrize("dtype", [jnp.float16, jnp.bfloat16])
@pytest.mark.parametrize("local", ['local',''])
@pytest.mark.parametrize("causal", ['causal',''])
@pytest.mark.parametrize("d", [59, 32])
@pytest.mark.parametrize("h", [1, 4])
@pytest.mark.parametrize("m", [1, 2]) # for MQA/GQA
def test_varlen_flash_bwd(m, h, d, causal, local, dtype, seqused_k_limit):
    window_size = (3,3) if local else (-1,-1)
    lens = [1, 2, 0, 6, 10]
    b = len(lens)
    total_seqlen = sum(lens)
    if seqused_k_limit is not None and (causal or local):
        return # skip causal/local tests with seqused_k_limit
    fenceposts = jnp.cumsum(jnp.array([0] + lens), dtype=jnp.int32)
    q = jax.random.normal(jax.random.PRNGKey(0), [total_seqlen, h*m, d], dtype=jnp.float32)
    k = jax.random.normal(jax.random.PRNGKey(1), [total_seqlen, h, d], dtype=jnp.float32)
    v = jax.random.normal(jax.random.PRNGKey(2), [total_seqlen, h, d], dtype=jnp.float32)

    seqused_k = None
    if seqused_k_limit is not None:
        seqused_k = jnp.array([min(l, seqused_k_limit) for l in lens])
    
    def ref(qkv, dtype=jnp.float32):
        q,k,v = tree_map(lambda x: x.astype(dtype), qkv)
        o = ref_mha_varlen(q, k, v, seqlens_q = fenceposts, seqlens_k = fenceposts, seqused_k=seqused_k,
                            max_seqlen_q=max(lens), max_seqlen_k=max(lens),
                            is_causal=bool(causal), window_size=window_size)
        return o.sum() * (1.0 / math.sqrt(total_seqlen * h * d * m))
        # q,k,v = tree_map(lambda x: x.astype(dtype), qkv)
        # out = jnp.zeros([total_seqlen, h*m, d], dtype=dtype)
        # for i in range(b):
        #     bq = q[None,fenceposts[i]:fenceposts[i+1]]
        #     bk = k[None,fenceposts[i]:fenceposts[i+1]]
        #     bv = v[None,fenceposts[i]:fenceposts[i+1]]
        #     if seqused_k_limit is not None:
        #         bk = bk[:, :seqused_k_limit, :]
        #         bv = bv[:, :seqused_k_limit, :]
        #     out = out.at[None,fenceposts[i]:fenceposts[i+1]].set(ref_mha(bq, bk, bv, is_causal=bool(causal), window_size=window_size))
        # return out.sum() * (1.0 / math.sqrt(total_seqlen * h * d * m))
    
    def fwd(qkv, dtype):
        q,k,v = tree_map(lambda x: x.astype(dtype), qkv)
        o = flash_mha_varlen(q, k, v, seqlens_q = fenceposts, seqlens_k = fenceposts, seqused_k=seqused_k,
                            max_seqlen_q=max(lens), max_seqlen_k=max(lens),
                            is_causal=bool(causal), window_size=window_size)
        return o.sum() * (1.0 / math.sqrt(total_seqlen * h * d * m))
    
    ref_grad = jax.grad(ref)((q,k,v), dtype=jnp.float32)
    ref_grad_dtype = jax.grad(ref)((q,k,v), dtype=dtype)
    mha_grad = jax.grad(fwd)((q,k,v), dtype=dtype)
    check(ref_grad, ref_grad_dtype, mha_grad)

def vmap_unrolled(f):
    def wrapped(*args, **kwargs):
        outs = []
        for items in zip(*args):
            outs.append(f(*items, **kwargs))
        return jnp.stack(outs)# for o in zip(*outs)
    return wrapped


@pytest.mark.parametrize("seqused_k_limit", [None, 4])
@pytest.mark.parametrize("dtype", [jnp.float16, jnp.bfloat16])
@pytest.mark.parametrize("local", ['local',''])
@pytest.mark.parametrize("causal", ['causal',''])
@pytest.mark.parametrize("d", [59, 32])
@pytest.mark.parametrize("h", [1, 4])
@pytest.mark.parametrize("m", [1, 2]) # for MQA/GQA
def test_varlen_flash_vmap(m, h, d, causal, local, dtype, seqused_k_limit):
    window_size = (3,3) if local else (-1,-1)
    lens = [1, 2, 0, 6, 10]
    b = len(lens)
    total_seqlen = sum(lens)

    if (causal or local) and seqused_k_limit is not None:
        pytest.skip()

    N = 4
    fenceposts = jnp.broadcast_to(jnp.cumsum(jnp.array([0] + lens), dtype=jnp.int32), (N, b+1))
    q = jax.random.normal(jax.random.PRNGKey(0), [N, total_seqlen, h*m, d], dtype=dtype)
    k = jax.random.normal(jax.random.PRNGKey(1), [N, total_seqlen, h, d], dtype=dtype)
    v = jax.random.normal(jax.random.PRNGKey(2), [N, total_seqlen, h, d], dtype=dtype)
    seqused_k = None
    if seqused_k_limit is not None:
        seqused_k = jnp.array([min(l, seqused_k_limit) for l in lens])
        seqused_k = jnp.broadcast_to(seqused_k, (N, b))
    def fwd_fn(q,k,v,fenceposts,seqused_k=None):
        return flash_mha_varlen(q,k,v, seqlens_q = fenceposts, seqlens_k = fenceposts, 
                           seqused_k=seqused_k,
                           max_seqlen_q=max(lens), max_seqlen_k=max(lens),
                            is_causal=bool(causal), window_size=window_size)
    def ref_fn(q,k,v,fenceposts,seqused_k=None):
        return ref_mha_varlen(q,k,v, seqlens_q = fenceposts, seqlens_k = fenceposts, 
                           seqused_k=seqused_k,
                           max_seqlen_q=max(lens), max_seqlen_k=max(lens),
                            is_causal=bool(causal), window_size=window_size)
    
    # if seqused_k is not None:
    #     out_ref = vmap_unrolled(fwd_fn)(q,k,v,fenceposts, seqused_k)
    # else:
    #     out_ref = vmap_unrolled(fwd_fn)(q,k,v,fenceposts)
    atol = 5e-3 if dtype == jnp.float16 else 4e-2
    rtol = 3e-3 if dtype == jnp.float16 else 3e-2
    out_ref = jax.vmap(ref_fn)(q,k,v,fenceposts, seqused_k)
    out_vmap = jax.vmap(fwd_fn)(q,k,v,fenceposts, seqused_k)
    np.testing.assert_allclose(out_vmap, out_ref, atol=atol, rtol=rtol)

    # if seqused_k is not None:
    #     grad_ref = jax.grad(lambda q,k,v: jnp.square(vmap_unrolled(fwd_fn)(q,k,v,fenceposts, seqused_k)).sum())(q,k,v)
    # else:
    #     grad_ref = jax.grad(lambda q,k,v: jnp.square(vmap_unrolled(fwd_fn)(q,k,v,fenceposts)).sum())(q,k,v)
    grad_ref = jax.grad(lambda qkv: jnp.square(jax.vmap(ref_fn)(*qkv,fenceposts, seqused_k)).sum())((q,k,v))
    grad_vmap = jax.grad(lambda qkv: jnp.square(jax.vmap(fwd_fn)(*qkv,fenceposts, seqused_k)).sum())((q,k,v))
    
    np.testing.assert_allclose(grad_vmap[0], grad_ref[0], atol=atol, rtol=rtol)
    np.testing.assert_allclose(grad_vmap[1], grad_ref[1], atol=atol, rtol=rtol)
    np.testing.assert_allclose(grad_vmap[2], grad_ref[2], atol=atol, rtol=rtol)


@pytest.mark.parametrize("dtype", [jnp.float16, jnp.bfloat16])
@pytest.mark.parametrize("local", ['local',''])
@pytest.mark.parametrize("causal", ['causal',''])
@pytest.mark.parametrize("d", [59, 32])
@pytest.mark.parametrize("h", [1, 4])
def test_varlen_flash_vmapk(h, d, causal, local, dtype):
    window_size = (3,3) if local else (-1,-1)
    lens = [1, 2, 0, 6, 10]
    b = len(lens)
    total_seqlen = sum(lens)

    N = 4
    seqlens_q = jnp.broadcast_to(jnp.cumsum(jnp.array([0] + lens), dtype=jnp.int32), (b+1,))
    seqlens_k = jnp.broadcast_to(jnp.cumsum(jnp.array([0] + lens), dtype=jnp.int32), (N, b+1))
    q = jax.random.normal(jax.random.PRNGKey(0), [total_seqlen, h, d], dtype=dtype)
    k = jax.random.normal(jax.random.PRNGKey(1), [N, total_seqlen, h, d], dtype=dtype)
    v = jax.random.normal(jax.random.PRNGKey(2), [N, total_seqlen, h, d], dtype=dtype)
    @partial(jax.vmap, in_axes=(None,0,0,None,0))
    def fwd_fn(q, k, v, seqlens_q, seqlens_k):
        return flash_mha_varlen(q,k,v, seqlens_q = seqlens_q, seqlens_k = seqlens_k, 
                           max_seqlen_q=max(lens), max_seqlen_k=max(lens),
                            is_causal=bool(causal), window_size=window_size)
    @partial(jax.vmap, in_axes=(None,0,0,None,0))
    def ref_fn(q, k, v, seqlens_q, seqlens_k):
        return ref_mha_varlen(q,k,v, seqlens_q = seqlens_q, seqlens_k = seqlens_k, 
                           max_seqlen_q=max(lens), max_seqlen_k=max(lens),
                            is_causal=bool(causal), window_size=window_size)
    
    atol = 3e-3 if dtype == jnp.float16 else 3e-2
    rtol = 3e-3 if dtype == jnp.float16 else 3e-2

    out_ref = ref_fn(q,k,v, seqlens_q, seqlens_k)
    out_vmap = fwd_fn(q,k,v, seqlens_q, seqlens_k)
    np.testing.assert_allclose(out_vmap, out_ref, atol=atol, rtol=rtol)


    grad_ref = jax.grad(lambda qkv: ref_fn(*qkv, seqlens_q, seqlens_k).sum())((q,k,v))
    grad_vmap = jax.grad(lambda qkv: fwd_fn(*qkv, seqlens_q, seqlens_k).sum())((q,k,v))
    
    np.testing.assert_allclose(grad_vmap[0], grad_ref[0], atol=atol, rtol=rtol)
    np.testing.assert_allclose(grad_vmap[1], grad_ref[1], atol=atol, rtol=rtol)
    np.testing.assert_allclose(grad_vmap[2], grad_ref[2], atol=atol, rtol=rtol)

if __name__ == '__main__':
    print(flash_mha_varlen(jnp.zeros([4,1,64],dtype=jnp.float16), 
                                jnp.zeros([4,1,64],dtype=jnp.float16), 
                                jnp.zeros([4,1,64],dtype=jnp.float16),
                                jnp.array([0,2,4]),
                                jnp.array([0,2,4]),
                                seqused_k=None,
                                max_seqlen_q=4,
                                max_seqlen_k=4,
                                softmax_scale=0.5,
                                is_causal=False,
                                window_size=(-1,-1)))
