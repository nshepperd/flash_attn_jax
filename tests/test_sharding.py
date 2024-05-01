import glob
import sys, os

os.environ["XLA_FLAGS"] = '--xla_force_host_platform_device_count=2'
if glob.glob('build/lib.linux-*'):
    sys.path.insert(0, glob.glob('build/lib.linux-*')[0])
sys.path.insert(0,'./src')

import math
import einops
import jax
import jax.numpy as jnp
import numpy as np
import pytest
from jax.sharding import Mesh, NamedSharding
from jax.sharding import PartitionSpec as P
from jax.sharding import PositionalSharding
from jax.tree_util import tree_map
jax.config.update("jax_traceback_filtering", "off")

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

@pytest.mark.skipif(len(jax.local_devices()) < 2, reason='Requires >1 gpu device')
@pytest.mark.parametrize("dtype", [jnp.float16, jnp.bfloat16])
@pytest.mark.parametrize("local", ['local',''])
@pytest.mark.parametrize("causal", ['causal',''])
@pytest.mark.parametrize("d", [32])
@pytest.mark.parametrize("h", [4])
@pytest.mark.parametrize("seqlen", [128])
@pytest.mark.parametrize("m", [1, 2]) # for MQA/GQA
def test_flash_fwd_sharded_hlo(seqlen, h, d, m, causal, local, dtype):
    window_size = (3,3) if local else (-1,-1)

    devices = jax.local_devices()[:4]
    n = len(devices)

    @jax.jit
    def flash(qkv):
        return flash_mha(*qkv, is_causal=bool(causal), window_size=window_size)

    def with_sharding(q_sharding, kv_sharding=None):
        if kv_sharding is None:
            kv_sharding = q_sharding
        q = jax.random.normal(jax.random.PRNGKey(0), [n, seqlen, m*h, d], dtype=dtype)
        k = jax.random.normal(jax.random.PRNGKey(1), [n, seqlen, h, d], dtype=dtype)
        v = jax.random.normal(jax.random.PRNGKey(2), [n, seqlen, h, d], dtype=dtype)
        q = jax.device_put(q, q_sharding)
        (k,v) = jax.device_put((k,v), kv_sharding)
        hlo = flash.lower((q,k,v)).compile().as_text()
        return hlo

    with Mesh(np.array(devices), axis_names=('x',)) as mesh:
        hlo = with_sharding(NamedSharding(mesh, P('x',None,None,None)))
        assert 'all-gather' not in hlo
        assert 'dynamic-slice' not in hlo

        hlo = with_sharding(NamedSharding(mesh, P(None,None,'x',None)))
        assert 'all-gather' not in hlo
        assert 'dynamic-slice' not in hlo

    if not local:
        with Mesh(np.array(devices), axis_names=('x',)) as mesh:
            sharding = NamedSharding(mesh, P(None,'x',None,None))
            hlo = with_sharding(sharding)
            # No resharding should occur, only manual collective-permute.
            assert 'all-gather' not in hlo
            assert 'dynamic-slice' not in hlo
            assert 'collective-permute' in hlo
            # Should always run concurrently, meaning custom-call is always between start and done.
            import re
            collectives = ''.join(re.findall(" collective-permute-start| collective-permute-done| custom-call", hlo))
            assert 'collective-permute-start collective-permute-done' not in collectives, hlo


@pytest.mark.skipif(len(jax.local_devices()) < 2, reason='Requires >1 gpu device')
@pytest.mark.parametrize("dtype", [jnp.float16, jnp.bfloat16])
@pytest.mark.parametrize("local", ['local',''])
@pytest.mark.parametrize("causal", ['causal',''])
@pytest.mark.parametrize("d", [32])
@pytest.mark.parametrize("h", [4])
@pytest.mark.parametrize("seqlen", [128])
@pytest.mark.parametrize("m", [1, 2]) # for MQA/GQA
def test_flash_bwd_sharded_hlo(seqlen, h, d, m, causal, local, dtype):
    window_size = (3,3) if local else (-1,-1)

    devices = jax.local_devices()[:4]
    n = len(devices)

    @jax.jit
    @jax.grad
    def flash(qkv):
        return (flash_mha(*qkv, is_causal=bool(causal), window_size=window_size)**2).sum()

    def with_sharding(sharding):
        q = jax.random.normal(jax.random.PRNGKey(0), [n, seqlen, m*h, d], dtype=dtype)
        k = jax.random.normal(jax.random.PRNGKey(1), [n, seqlen, h, d], dtype=dtype)
        v = jax.random.normal(jax.random.PRNGKey(2), [n, seqlen, h, d], dtype=dtype)
        (q,k,v) = jax.device_put((q,k,v), sharding)
        hlo = flash.lower((q,k,v)).compile().as_text()
        return hlo

    hlo = with_sharding(PositionalSharding(devices).reshape(n,1,1,1))
    assert 'all-gather' not in hlo
    assert 'dynamic-slice' not in hlo

    hlo = with_sharding(PositionalSharding(devices).reshape(1,1,n,1))
    assert 'all-gather' not in hlo
    assert 'dynamic-slice' not in hlo

    if not local:
        with Mesh(np.array(devices), axis_names=('x',)) as mesh:
            sharding = NamedSharding(mesh, P(None,'x',None,None))
            hlo = with_sharding(sharding)
            # No resharding should occur, only manual collective-permute.
            assert 'all-gather' not in hlo
            assert 'dynamic-slice' not in hlo
            assert 'collective-permute' in hlo
            # Should always run concurrently, meaning custom-call is always between start and done.
            import re
            collectives = ''.join(re.findall(" collective-permute-start| collective-permute-done| custom-call", hlo))
            assert 'collective-permute-start collective-permute-done' not in collectives, hlo

@pytest.mark.skipif(len(jax.local_devices()) < 2, reason='Requires >1 gpu device')
@pytest.mark.parametrize("dtype", [jnp.float16, jnp.bfloat16])
@pytest.mark.parametrize("local", ['local',''])
@pytest.mark.parametrize("causal", ['causal',''])
@pytest.mark.parametrize("d", [32])
@pytest.mark.parametrize("h", [4, 8])
@pytest.mark.parametrize("seqlen", [128])
@pytest.mark.parametrize("m", [1, 2]) # for MQA/GQA
def test_flash_fwd_sharded(seqlen, h, d, m, causal, local, dtype):
    window_size = (3,3) if local else (-1,-1)

    devices = jax.local_devices()
    n = len(devices)

    @jax.jit
    def ref(qkv):
        return ref_mha(*qkv, is_causal=bool(causal), window_size=window_size)
    @jax.jit
    def flash(qkv):
        return flash_mha(*qkv, is_causal=bool(causal), window_size=window_size)
    q = jax.random.normal(jax.random.PRNGKey(0), [n, seqlen, h*m, d], dtype=jnp.float32)
    k = jax.random.normal(jax.random.PRNGKey(1), [n, seqlen, h, d], dtype=jnp.float32)
    v = jax.random.normal(jax.random.PRNGKey(2), [n, seqlen, h, d], dtype=jnp.float32)

    ref_out = ref((q,k,v))
    q = q.astype(dtype)
    k = k.astype(dtype)
    v = v.astype(dtype)
    ref16_out = flash((q,k,v))

    def check_sharding(sharding,q,k,v):
        (q,k,v) = jax.device_put((q,k,v), sharding)
        out = flash((q,k,v))
        check(ref_out,ref16_out,out)

    check_sharding(PositionalSharding(devices).reshape(n,1,1,1),q,k,v)
    check_sharding(PositionalSharding(devices).reshape(1,1,n,1),q,k,v)

    if not local:
        # Ring attention
        with Mesh(np.array(devices), axis_names=('x',)) as mesh:
            sharding = NamedSharding(mesh, P(None,'x',None,None))
            check_sharding(sharding,q,k,v)


@pytest.mark.skipif(len(jax.local_devices()) < 2, reason='Requires >1 gpu device')
@pytest.mark.parametrize("dtype", [jnp.float16, jnp.bfloat16])
@pytest.mark.parametrize("local", ['local',''])
@pytest.mark.parametrize("causal", ['causal',''])
@pytest.mark.parametrize("d", [32])
@pytest.mark.parametrize("h", [4, 8])
@pytest.mark.parametrize("seqlen", [128])
@pytest.mark.parametrize("m", [1, 2]) # for MQA/GQA
def test_flash_bwd_sharded(seqlen, h, d, m, causal, local, dtype):
    window_size = (3,3) if local else (-1,-1)

    devices = jax.local_devices()
    n = len(devices)

    @jax.jit
    @jax.grad
    def ref(qkv):
        return ref_mha(*qkv, is_causal=bool(causal), window_size=window_size).sum()
    @jax.jit
    @jax.grad
    def flash(qkv):
        return flash_mha(*qkv, is_causal=bool(causal), window_size=window_size).sum()
    q = jax.random.normal(jax.random.PRNGKey(0), [n, seqlen, h*m, d], dtype=jnp.float32)
    k = jax.random.normal(jax.random.PRNGKey(1), [n, seqlen, h, d], dtype=jnp.float32)
    v = jax.random.normal(jax.random.PRNGKey(2), [n, seqlen, h, d], dtype=jnp.float32)

    ref_out = ref((q,k,v))
    q = q.astype(dtype)
    k = k.astype(dtype)
    v = v.astype(dtype)
    ref16_out = flash((q,k,v))

    def check_sharding(sharding,q,k,v):
        (q,k,v) = jax.device_put((q,k,v), sharding)
        out = flash((q,k,v))
        check(ref_out,ref16_out,out)

    check_sharding(PositionalSharding(devices).reshape(n,1,1,1),q,k,v)
    check_sharding(PositionalSharding(devices).reshape(1,1,n,1),q,k,v)

    if not local:
        # Ring attention
        with Mesh(np.array(devices), axis_names=('x',)) as mesh:
            sharding = NamedSharding(mesh, P(None,'x',None,None))
            check_sharding(sharding,q,k,v)

if __name__ == '__main__':
    test_flash_fwd_sharded_hlo(128,4,32,False,False,jnp.float16)
