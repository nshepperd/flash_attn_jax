import sys, glob,  os
os.environ["XLA_FLAGS"] = '--xla_force_host_platform_device_count=2'
if glob.glob('build/lib.linux-*'):
    sys.path.insert(0, glob.glob('build/lib.linux-*')[0])
sys.path.insert(0,'./src')

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

def check(ref_out, jax_out, out):
    def check1(ref_out, jax_out, out):
        assert jnp.max(jnp.abs(out - ref_out)).item() <= 3 * jnp.max(jnp.abs(jax_out - ref_out)).item(), (pretty(jnp.abs(out - ref_out)), 'vs', pretty(jnp.abs(jax_out - ref_out)))
    tree_map(check1, ref_out, jax_out, out)

@pytest.mark.parametrize("dtype", [jnp.float16, jnp.bfloat16])
@pytest.mark.parametrize("d", [59, 32])
@pytest.mark.parametrize("h", [4])
@pytest.mark.parametrize("seqlen_q", [97, 128])
@pytest.mark.parametrize("seqlen_k", [32, 63])
@pytest.mark.parametrize("n", [1])
@pytest.mark.parametrize("m", [1, 2]) # for MQA/GQA
def test_cross_fwd(n, seqlen_q, seqlen_k, h, d, m, dtype):
    q = jax.random.normal(jax.random.PRNGKey(0), [n, seqlen_q, h*m, d], dtype=jnp.float32)
    k = jax.random.normal(jax.random.PRNGKey(1), [n, seqlen_k, h, d], dtype=jnp.float32)
    v = jax.random.normal(jax.random.PRNGKey(2), [n, seqlen_k, h, d], dtype=jnp.float32)
    ref_out = ref_mha(q,k,v)
    q = q.astype(dtype)
    k = k.astype(dtype)
    v = v.astype(dtype)
    jax_out = ref_mha(q,k,v)
    out = flash_mha(q,k,v)
    check(ref_out, jax_out, out)

@pytest.mark.parametrize("dtype", [jnp.float16, jnp.bfloat16])
@pytest.mark.parametrize("d", [59, 32])
@pytest.mark.parametrize("h", [4])
@pytest.mark.parametrize("seqlen_q", [97, 128])
@pytest.mark.parametrize("seqlen_k", [32, 63])
@pytest.mark.parametrize("n", [1])
@pytest.mark.parametrize("m", [1, 2]) # for MQA/GQA
def test_cross_bwd(n, seqlen_q, seqlen_k, h, d, m, dtype):
    @jax.grad
    def ref(qkv):
        return ref_mha(*qkv).sum()
    @jax.grad
    def flash(qkv):
        return flash_mha(*qkv).sum()
    q = jax.random.normal(jax.random.PRNGKey(0), [n, seqlen_q, h*m, d], dtype=jnp.float32)
    k = jax.random.normal(jax.random.PRNGKey(1), [n, seqlen_k, h, d], dtype=jnp.float32)
    v = jax.random.normal(jax.random.PRNGKey(2), [n, seqlen_k, h, d], dtype=jnp.float32)
    ref_out = ref((q,k,v))
    q = q.astype(dtype)
    k = k.astype(dtype)
    v = v.astype(dtype)
    jax_out = ref((q,k,v))
    out = flash((q,k,v))
    check(ref_out, jax_out, out)
