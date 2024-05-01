import glob
import sys, os

os.environ["XLA_FLAGS"] = '--xla_force_host_platform_device_count=2'
if glob.glob('build/lib.linux-*'):
    sys.path.insert(0, glob.glob('build/lib.linux-*')[0])
sys.path.insert(0,'./src')

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from jax.sharding import Mesh, NamedSharding
from jax.sharding import PartitionSpec as P
from jax.sharding import PositionalSharding
from jax.tree_util import tree_map
from jax.experimental.shard_map import shard_map
from functools import partial
import einops

from flash_attn_jax.ring_attention import ring_fwd, ring_bwd

def make_mask(R, C, is_causal, window_size):
    mask = jnp.ones([R,C], dtype=jnp.int32)
    if is_causal:
        mask = jnp.tril(mask)
    if window_size[0] != -1:
        mask = jnp.triu(mask, -window_size[0])
    if window_size[1] != -1:
        mask = jnp.tril(mask, window_size[1])
    return mask

def ref_fwd(q,k,v, is_causal=False, window_size=(-1,-1), softmax_scale=None):
    if softmax_scale is None:
        softmax_scale = 1/np.sqrt(q.shape[-1])
    att = jnp.einsum('nlhd,nLhd->nhlL',q,k)*softmax_scale
    [_, _, l, L] = att.shape
    mask = make_mask(l,L,is_causal,window_size)
    att = jnp.where(mask, att, float('-inf'))
    lse = jax.nn.logsumexp(att, axis=-1) #nhl
    att = jnp.exp(att - lse[...,None])
    o = jnp.einsum('nhlL,nLhd->nlhd',att,v)
    return o, lse

def ref_bwd(do,q,k,v,o,lse, is_causal=False, window_size=(-1,-1), softmax_scale=None):
    if softmax_scale is None:
        softmax_scale = 1/np.sqrt(q.shape[-1])
    S = jnp.einsum('nlhd,nLhd->nhlL',q,k)*softmax_scale
    D = einops.reduce(do * o, 'n l h d -> n h l', reduction='sum')
    [_, _, l, L] = S.shape
    mask = make_mask(l,L,is_causal,window_size)
    S = jnp.where(mask, S, float('-inf'))
    P = jnp.exp(S - lse[...,None]) # n h l L
    dP = jnp.einsum('nlhd,nLhd->nhlL',do,v)
    dv = jnp.einsum('nlhd,nhlL->nLhd',do,P)
    dS = P * (dP - D[...,None])
    dq = softmax_scale*jnp.einsum('nLhd,nhlL->nlhd',k,dS)
    dk = softmax_scale*jnp.einsum('nlhd,nhlL->nLhd',q,dS)
    return dq,dk,dv


def test_ref_bwd():
    """Test that ref_bwd matches autodiff and is splittable."""
    n = 1
    seqlen = 2
    h = 1
    d = 4

    q = jax.random.normal(jax.random.PRNGKey(0), [n, seqlen, h, d], dtype=jnp.float32)
    k = jax.random.normal(jax.random.PRNGKey(1), [n, seqlen, h, d], dtype=jnp.float32)
    v = jax.random.normal(jax.random.PRNGKey(2), [n, seqlen, h, d], dtype=jnp.float32)
    do = jax.random.normal(jax.random.PRNGKey(3), [n, seqlen, h, d], dtype=jnp.float32)

    def b_ref(do,q,k,v):
        def fwd(q,k,v):
            return ref_fwd(q,k,v)[0]
        _, bwd = jax.vjp(fwd, q, k, v)
        return bwd(do)

    def b_bwd(do,q,k,v):
        o, lse = ref_fwd(q,k,v, is_causal=False, window_size=(-1,-1), softmax_scale=1/np.sqrt(q.shape[-1]))
        dq = jnp.zeros_like(q)
        for qi in range(seqlen):
            for ki in range(seqlen):
                dqa, dka, dva = ref_bwd(do[:,qi:qi+1],q[:,qi:qi+1],k[:,ki:ki+1],v[:,ki:ki+1],o[:,qi:qi+1],lse[:,:,qi:qi+1], is_causal=False, window_size=(-1,-1), softmax_scale=1/np.sqrt(q.shape[-1]))
                dq = dq.at[:,qi].add(dqa[:,0])
        return dq,

    dq_ref = b_ref(do,q,k,v)[0]
    dq_bwd = b_bwd(do,q,k,v)[0]
    assert jnp.allclose(dq_ref, dq_bwd, rtol=1e-2, atol=1e-3)

@pytest.mark.parametrize("causal", ['causal',''])
@pytest.mark.parametrize("d", [32])
@pytest.mark.parametrize("h", [4])
@pytest.mark.parametrize("seqlen", [128])
def test_ring_fwd(seqlen, h, d, causal):
    window_size = (-1,-1)

    devices = jax.devices(backend='cpu')
    n = len(devices)

    with Mesh(np.array(devices), axis_names=('x',)) as mesh:
        @jax.jit
        def ref(q,k,v):
            return ref_fwd(q,k,v, is_causal=bool(causal), window_size=window_size)[0]

        in_specs = (P(None,'x',None,None),)*3
        out_specs = P(None,'x',None,None)

        @jax.jit
        @partial(shard_map, mesh=mesh, in_specs=in_specs, out_specs=out_specs, check_rep=False)
        def ring(q,k,v):
            softmax_scale = 1/np.sqrt(q.shape[-1])
            return ring_fwd(q, k, v, softmax_scale=softmax_scale, is_causal=bool(causal), axis_name='x', axis_size=n, mha_fwd=ref_fwd)[0]

        q = jax.random.normal(jax.random.PRNGKey(0), [n, seqlen, h, d], dtype=jnp.float32)
        k = jax.random.normal(jax.random.PRNGKey(1), [n, seqlen, h, d], dtype=jnp.float32)
        v = jax.random.normal(jax.random.PRNGKey(2), [n, seqlen, h, d], dtype=jnp.float32)
        o_ref = ref(q,k,v)
        o_ring = ring(q,k,v)
        assert jnp.allclose(o_ref, o_ring, rtol=1e-2, atol=2e-3)


@pytest.mark.parametrize("causal", ['causal',''])
@pytest.mark.parametrize("d", [8])
@pytest.mark.parametrize("h", [1])
@pytest.mark.parametrize("seqlen", [2])
def test_ring_bwd(seqlen, h, d, causal):
    window_size = (-1,-1)

    devices = jax.devices(backend='cpu')
    n_device = len(devices)

    with Mesh(np.array(devices), axis_names=('x',)) as mesh:
        @jax.jit
        def ref(q,k,v,do):
            o, lse = ref_fwd(q,k,v, is_causal=bool(causal), window_size=window_size, softmax_scale=1/np.sqrt(q.shape[-1]))
            dq,dk,dv = ref_bwd(do,q,k,v,o,lse, is_causal=bool(causal), window_size=window_size, softmax_scale=1/np.sqrt(q.shape[-1]))
            return dq,dk,dv

        @jax.jit
        @partial(shard_map, mesh=mesh, in_specs=(P(None,'x',None,None),)*4, out_specs=(P(None,'x',None,None),)*3, check_rep=False)
        def ring(q,k,v,do):
            softmax_scale = 1/np.sqrt(q.shape[-1])
            o, lse = ring_fwd(q, k, v, softmax_scale=softmax_scale, is_causal=bool(causal), axis_name='x', axis_size=n_device, mha_fwd=ref_fwd)
            dq,dk,dv = ring_bwd(do,q,k,v,o,lse, softmax_scale=softmax_scale, is_causal=bool(causal), axis_name='x', axis_size=n_device, mha_bwd=ref_bwd)
            return dq,dk,dv

        q = jax.random.normal(jax.random.PRNGKey(0), [1, seqlen, h, d], dtype=jnp.float32)
        k = jax.random.normal(jax.random.PRNGKey(1), [1, seqlen, h, d], dtype=jnp.float32)
        v = jax.random.normal(jax.random.PRNGKey(2), [1, seqlen, h, d], dtype=jnp.float32)
        do = jax.random.normal(jax.random.PRNGKey(3), [1, seqlen, h, d], dtype=jnp.float32)
        o_ref = ref(q,k,v,do)
        o_ring = ring(q,k,v,do)
        for i in range(3):
            assert jnp.allclose(o_ref[i], o_ring[i], rtol=1e-2, atol=1e-3)

if __name__ == '__main__':
    test_ref()