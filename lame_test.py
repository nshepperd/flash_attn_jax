import sys
sys.path.append('build/lib.linux-x86_64-cpython-311')

from functools import partial
from functools import reduce

import jaxlib.mlir.ir
import jax
import jax.numpy as jnp
import jax._src.test_util as jtu
from jax import core, dtypes
from jax.interpreters import xla
from jax.lib import xla_client
from jax.interpreters import mlir
from jax.interpreters.mlir import ir
from jaxlib.hlo_helpers import custom_call

# from flash_attn_jax.flash import flash_mha_fwd, flash_mha_bwd
from flash_attn_jax import flash_mha


if __name__ == '__main__':
    import time
    import numpy as np

    @jax.jit
    def pure_mha(q,k,v, softmax_scale=None):
        if softmax_scale is None:
            softmax_scale = 1/np.sqrt(q.shape[-1])
        att = jnp.einsum('nlhd,nLhd->nhlL',q,k)
        att = jax.nn.softmax(att*softmax_scale, axis=-1)
        o = jnp.einsum('nhlL,nLhd->nlhd',att,v)
        return o

    # n l h d

    def pretty(tensor):
        shape = tensor.shape
        mx = jnp.max(tensor)
        mn = jnp.min(tensor)
        mean = jnp.mean(tensor)
        std = jnp.std(tensor)
        return f'[{shape}: {mn:.3g} | {mean:.3g}±{std:.3g} | {mx:.3g}]'

    # q = jax.random.normal(jax.random.PRNGKey(0), [2, 4096, 4, 32]).astype(jnp.float16)
    # k = jax.random.normal(jax.random.PRNGKey(1), [2, 4096, 4, 32]).astype(jnp.float16)
    # v = jax.random.normal(jax.random.PRNGKey(2), [2, 4096, 4, 32]).astype(jnp.float16)

    # @jax.jit
    # def fwd(q,k,v):
    #     return flash_mha(q,k,v)

    # from jax.sharding import PositionalSharding
    # from einops import rearrange

    # sharding = PositionalSharding([*jax.devices(), *jax.devices(backend='cpu')])

    # q = jax.device_put(q, sharding.reshape(2,1,1,1))
    # k = jax.device_put(k, sharding.reshape(2,1,1,1))
    # v = jax.device_put(v, sharding.reshape(2,1,1,1))
    # jax.debug.visualize_array_sharding(rearrange(q, 'n l h d -> n (l h d)'))
    # print(fwd.lower(q,k,v).compile().as_text())
    # # exit()

    print('==== forward ====')
    q = jax.random.normal(jax.random.PRNGKey(0), [32, 4096, 4, 32]).astype(jnp.float16)
    k = jax.random.normal(jax.random.PRNGKey(1), [32, 4096, 4, 32]).astype(jnp.float16)
    v = jax.random.normal(jax.random.PRNGKey(2), [32, 4096, 4, 32]).astype(jnp.float16)

    @jax.jit
    def fwd(q,k,v):
        o = flash_mha(q,k,v)
        for _ in range(32):
            o = flash_mha(q,k,o)
        return o

    @jax.jit
    def fwd_jax(q,k,v):
        ro = pure_mha(q,k,v)
        for _ in range(32):
            ro = pure_mha(q,k,ro)
        return ro

    o = fwd(q,k,v) #, softmax_scale=float(np.sqrt(1/32)))[0]
    start = time.time()
    o = fwd(q,k,v) #, softmax_scale=float(np.sqrt(1/32)))[0]
    print('flash:', time.time() - start, 'seconds')
    ro = fwd_jax(q,k,v)
    start = time.time()
    ro = fwd_jax(q,k,v)
    print('jax:', time.time() - start, 'seconds')
    print(pretty(jnp.abs(o - ro)), jnp.mean(jnp.abs(ro)))

    @jax.jit
    @jax.grad
    def grad_pure(inputs):
        q,k,v = inputs
        return pure_mha(q,k,v).sum()

    @jax.jit
    @jax.grad
    def grad_flash(inputs):
        q,k,v = inputs
        return flash_mha(q,k,v).sum()

    print('==== backward ====')
    q = jax.random.normal(jax.random.PRNGKey(0), [1, 4, 2, 32]).astype(jnp.float16)
    k = jax.random.normal(jax.random.PRNGKey(1), [1, 4, 2, 32]).astype(jnp.float16)
    v = jax.random.normal(jax.random.PRNGKey(2), [1, 4, 2, 32]).astype(jnp.float16)
    dq, dk, dv = grad_flash((q,k,v))
    rdq, rdk, rdv = grad_pure((q,k,v))
    # print(rdq, jnp.mean(jnp.abs(rdq)))
    print('q', pretty(jnp.abs(dq - rdq)), jnp.mean(jnp.abs(rdq)))
    print('k', pretty(jnp.abs(dk - rdk)), jnp.mean(jnp.abs(rdk)))
    print('v', pretty(jnp.abs(dv - rdv)), jnp.mean(jnp.abs(rdv)))
