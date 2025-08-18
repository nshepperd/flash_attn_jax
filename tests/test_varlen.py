import sys, glob, os
os.environ["XLA_FLAGS"] = '--xla_force_host_platform_device_count=2'
if glob.glob('build/lib.linux-*'):
    sys.path.insert(0, glob.glob('build/lib.linux-*')[0])
sys.path.insert(0,'./src')

from functools import partial
import pytest
import jax
import jax.numpy as jnp
from jax.tree_util import tree_map
import numpy as np
import math
import einops
jax.config.update("jax_default_matmul_precision", "highest")

from flash_attn_jax import flash_mha
from flash_attn_jax.varlen import flash_mha_varlen
from .ref_mha import ref_mha

def pretty(tensor):
    shape = tensor.shape
    mx = jnp.max(tensor)
    mn = jnp.min(tensor)
    mean = jnp.mean(tensor)
    std = jnp.std(tensor)
    return f'[{shape}: {mn:.3g} | {mean:.3g}±{std:.3g} | {mx:.3g}]'

# Smart idea from Tri Dao's repo: compare both impl to a float32
# reference impl, and call it a pass if the absolute error isn't
# more than 3x worse with flash attention.
def check(ref_out, jax_out, out, margin=4):
    def check1(ref_out, jax_out, out):
        assert jnp.max(jnp.abs(out - ref_out)).item() <= margin * jnp.max(jnp.abs(jax_out - ref_out)).item(), (pretty(jnp.abs(out - ref_out)), 'vs', pretty(jnp.abs(jax_out - ref_out)))
    tree_map(check1, ref_out, jax_out, out)

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

    def ref(q,k,v):
        out = jnp.zeros([total_seqlen, h*m, d], dtype=jnp.float32)
        for i in range(b):
            bq = q[None,fenceposts[i]:fenceposts[i+1]]
            bk = k[None,fenceposts[i]:fenceposts[i+1]]
            bv = v[None,fenceposts[i]:fenceposts[i+1]]
            if seqused_k_limit is not None:
                bk = bk[:, :seqused_k_limit, :]
                bv = bv[:, :seqused_k_limit, :]
            out = out.at[None,fenceposts[i]:fenceposts[i+1]].set(ref_mha(bq, bk, bv, is_causal=bool(causal), window_size=window_size))
        return out

    ref_out = ref(q,k,v)
    q = q.astype(dtype)
    k = k.astype(dtype)
    v = v.astype(dtype)
    jax_out = ref(q,k,v)

    seqused_k = None
    if seqused_k_limit is not None:
        seqused_k = jnp.array([min(l, seqused_k_limit) for l in lens])
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

    def ref(qkv, dtype=jnp.float32):
        q,k,v = tree_map(lambda x: x.astype(dtype), qkv)
        out = jnp.zeros([total_seqlen, h*m, d], dtype=dtype)
        for i in range(b):
            bq = q[None,fenceposts[i]:fenceposts[i+1]]
            bk = k[None,fenceposts[i]:fenceposts[i+1]]
            bv = v[None,fenceposts[i]:fenceposts[i+1]]
            if seqused_k_limit is not None:
                bk = bk[:, :seqused_k_limit, :]
                bv = bv[:, :seqused_k_limit, :]
            out = out.at[None,fenceposts[i]:fenceposts[i+1]].set(ref_mha(bq, bk, bv, is_causal=bool(causal), window_size=window_size))
        return out.sum() * (1.0 / math.sqrt(total_seqlen * h * d * m))
    
    seqused_k = None
    if seqused_k_limit is not None:
        seqused_k = jnp.array([min(l, seqused_k_limit) for l in lens])

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
