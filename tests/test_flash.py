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

from flash_attn_jax import flash_mha
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
def check(ref_out, jax_out, out):
    def check1(ref_out, jax_out, out):
        assert jnp.max(jnp.abs(out - ref_out)).item() <= 3 * jnp.max(jnp.abs(jax_out - ref_out)).item(), (pretty(jnp.abs(out - ref_out)), 'vs', pretty(jnp.abs(jax_out - ref_out)))
    tree_map(check1, ref_out, jax_out, out)

@pytest.mark.parametrize("dtype", [jnp.float16, jnp.bfloat16])
@pytest.mark.parametrize("local", ['local',''])
@pytest.mark.parametrize("causal", ['causal',''])
@pytest.mark.parametrize("d", [59, 32])
@pytest.mark.parametrize("h", [1, 4])
@pytest.mark.parametrize("seqlen", [97, 128])
@pytest.mark.parametrize("n", [1])
@pytest.mark.parametrize("m", [1, 2]) # for MQA/GQA
def test_flash_fwd(n, seqlen, h, d, m, causal, local, dtype):
    window_size = (3,3) if local else (-1,-1)

    q = jax.random.normal(jax.random.PRNGKey(0), [n, seqlen, h*m, d], dtype=jnp.float32)
    k = jax.random.normal(jax.random.PRNGKey(1), [n, seqlen, h, d], dtype=jnp.float32)
    v = jax.random.normal(jax.random.PRNGKey(2), [n, seqlen, h, d], dtype=jnp.float32)
    ref_out = ref_mha(q,k,v, is_causal=bool(causal), window_size=window_size)
    q = q.astype(dtype)
    k = k.astype(dtype)
    v = v.astype(dtype)
    jax_out = ref_mha(q,k,v, is_causal=bool(causal), window_size=window_size)
    out = flash_mha(q,k,v, is_causal=bool(causal), window_size=window_size)
    check(ref_out, jax_out, out)

@pytest.mark.parametrize("dtype", [jnp.float16, jnp.bfloat16])
@pytest.mark.parametrize("local", ['local',''])
@pytest.mark.parametrize("causal", ['causal',''])
@pytest.mark.parametrize("d", [59, 32, 40])
@pytest.mark.parametrize("h", [1, 4, 8])
@pytest.mark.parametrize("seqlen", [97, 128])
@pytest.mark.parametrize("n", [1])
@pytest.mark.parametrize("m", [1, 2]) # for MQA/GQA
def test_flash_bwd(n, seqlen, h, d, m, causal, local, dtype):
    window_size = (3,3) if local else (-1,-1)
    A = 1.0 / math.sqrt(n * seqlen * h * d)

    @jax.grad
    def ref(qkv):
        return ref_mha(*qkv, is_causal=bool(causal), window_size=window_size).sum() * A

    @jax.jit
    @jax.grad
    def flash(qkv):
        return flash_mha(*qkv, is_causal=bool(causal), window_size=window_size).sum() * A
    q = jax.random.normal(jax.random.PRNGKey(0), [n, seqlen, h*m, d], dtype=jnp.float32)
    k = jax.random.normal(jax.random.PRNGKey(1), [n, seqlen, h, d], dtype=jnp.float32)
    v = jax.random.normal(jax.random.PRNGKey(2), [n, seqlen, h, d], dtype=jnp.float32)
    ref_out = ref((q,k,v))
    q = q.astype(dtype)
    k = k.astype(dtype)
    v = v.astype(dtype)
    jax_out = ref((q,k,v))
    # print(flash.lower((q,k,v)).compile().as_text())
    out = flash((q,k,v))
    check(ref_out, jax_out, out)

@pytest.mark.parametrize("dtype", [jnp.float16, jnp.bfloat16])
@pytest.mark.parametrize("local", ['local',''])
@pytest.mark.parametrize("causal", ['causal',''])
@pytest.mark.parametrize("d", [59, 32])
@pytest.mark.parametrize("h", [1, 4])
@pytest.mark.parametrize("seqlen", [97, 128])
@pytest.mark.parametrize("n", [1])
def test_flash_fwd_vmap(n, seqlen, h, d, causal, local, dtype):
    window_size = (3,3) if local else (-1,-1)

    x = 4
    q = jax.random.normal(jax.random.PRNGKey(0), [x, n, seqlen, h, d], dtype=jnp.float32)
    k = jax.random.normal(jax.random.PRNGKey(1), [x, n, seqlen, h, d], dtype=jnp.float32)
    v = jax.random.normal(jax.random.PRNGKey(2), [x, n, seqlen, h, d], dtype=jnp.float32)

    def ref(q,k,v):
        return ref_mha(q,k,v, is_causal=bool(causal), window_size=window_size)
    def flash(q,k,v):
        return flash_mha(q,k,v, is_causal=bool(causal), window_size=window_size)

    ref_out = jax.vmap(ref)(q,k,v)
    q = q.astype(dtype)
    k = k.astype(dtype)
    v = v.astype(dtype)
    f16_out = jax.vmap(ref)(q,k,v)

    out = jax.vmap(flash)(q,k,v)
    check(ref_out, f16_out, out)

@pytest.mark.parametrize("dtype", [jnp.float16, jnp.bfloat16])
@pytest.mark.parametrize("local", ['local',''])
@pytest.mark.parametrize("causal", ['causal',''])
@pytest.mark.parametrize("d", [59, 32])
@pytest.mark.parametrize("h", [1, 4])
@pytest.mark.parametrize("seqlen", [97, 128])
@pytest.mark.parametrize("n", [1])
def test_flash_fwd_vmapq(n, seqlen, h, d, causal, local, dtype):
    window_size = (3,3) if local else (-1,-1)

    x = 4
    q = jax.random.normal(jax.random.PRNGKey(0), [x, n, seqlen, h, d], dtype=jnp.float32)
    k = jax.random.normal(jax.random.PRNGKey(1), [n, seqlen, h, d], dtype=jnp.float32)
    v = jax.random.normal(jax.random.PRNGKey(2), [n, seqlen, h, d], dtype=jnp.float32)

    def ref(q,k,v):
        return ref_mha(q,k,v, is_causal=bool(causal), window_size=window_size)
    def flash(q,k,v):
        return flash_mha(q,k,v, is_causal=bool(causal), window_size=window_size)

    ref_out = jax.vmap(ref, in_axes=(0,None,None))(q,k,v)
    q = q.astype(dtype)
    k = k.astype(dtype)
    v = v.astype(dtype)
    f16_out = jax.vmap(ref, in_axes=(0,None,None))(q,k,v)

    out = jax.vmap(flash, in_axes=(0,None,None))(q,k,v)
    check(ref_out, f16_out, out)

@pytest.mark.parametrize("dtype", [jnp.float16, jnp.bfloat16])
@pytest.mark.parametrize("local", ['local',''])
@pytest.mark.parametrize("causal", ['causal',''])
@pytest.mark.parametrize("d", [59, 32])
@pytest.mark.parametrize("h", [1, 4])
@pytest.mark.parametrize("seqlen", [97, 128])
@pytest.mark.parametrize("n", [1])
def test_flash_bwd_vmap(n, seqlen, h, d, causal, local, dtype):
    window_size = (3,3) if local else (-1,-1)

    x = 4
    q = jax.random.normal(jax.random.PRNGKey(0), [x, n, seqlen, h, d], dtype=jnp.float32)
    k = jax.random.normal(jax.random.PRNGKey(1), [x, n, seqlen, h, d], dtype=jnp.float32)
    v = jax.random.normal(jax.random.PRNGKey(2), [x, n, seqlen, h, d], dtype=jnp.float32)
    do = jax.random.normal(jax.random.PRNGKey(3), [x, n, seqlen, h, d], dtype=jnp.float32)

    def func(mha, q,k,v):
        @partial(jax.vmap, in_axes=(0,0,0))
        def fwd(q,k,v):
            return mha(q,k,v, is_causal=bool(causal), window_size=window_size)
        o, bwd = jax.vjp(fwd,q,k,v)
        return bwd(do)

    ref_out = func(ref_mha, q,k,v)
    q = q.astype(dtype)
    k = k.astype(dtype)
    v = v.astype(dtype)
    do = do.astype(dtype)
    f16_out = func(ref_mha, q,k,v)

    out = func(flash_mha, q,k,v)
    check(ref_out, f16_out, out)

@pytest.mark.parametrize("dtype", [jnp.float16, jnp.bfloat16])
@pytest.mark.parametrize("local", ['local',''])
@pytest.mark.parametrize("causal", ['causal',''])
@pytest.mark.parametrize("d", [59, 32])
@pytest.mark.parametrize("h", [1, 4])
@pytest.mark.parametrize("seqlen", [97, 128])
@pytest.mark.parametrize("n", [1])
def test_flash_bwd_vmapq(n, seqlen, h, d, causal, local, dtype):
    window_size = (3,3) if local else (-1,-1)

    x = 4
    q = jax.random.normal(jax.random.PRNGKey(0), [x, n, seqlen, h, d], dtype=jnp.float32)
    k = jax.random.normal(jax.random.PRNGKey(1), [n, seqlen, h, d], dtype=jnp.float32)
    v = jax.random.normal(jax.random.PRNGKey(2), [n, seqlen, h, d], dtype=jnp.float32)
    do = jax.random.normal(jax.random.PRNGKey(3), [x, n, seqlen, h, d], dtype=jnp.float32)

    def func(mha, q,k,v):
        @partial(jax.vmap, in_axes=(0,None,None))
        def fwd(q,k,v):
            return mha(q,k,v, is_causal=bool(causal), window_size=window_size)
        o, bwd = jax.vjp(fwd,q,k,v)
        return bwd(do)

    ref_out = func(ref_mha, q,k,v)
    q = q.astype(dtype)
    k = k.astype(dtype)
    v = v.astype(dtype)
    do = do.astype(dtype)
    f16_out = func(ref_mha, q,k,v)

    out = func(flash_mha, q,k,v)
    check(ref_out, f16_out, out)

if __name__ == '__main__':
    test_flash_bwd(1,4,1,32,4,False,False,jnp.float16)
