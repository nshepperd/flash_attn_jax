import glob
import os
import sys

os.environ["XLA_FLAGS"] = '--xla_force_host_platform_device_count=2'

if glob.glob('build/lib.linux-*'):
    sys.path.append(glob.glob('build/lib.linux-*')[0])

from functools import partial, reduce

import jax
import jax.numpy as jnp
import jaxlib.mlir.ir
from einops import rearrange
from jax import core, dtypes
from jax.experimental import mesh_utils
from jax.interpreters import mlir, xla
from jax.interpreters.mlir import ir
from jax.lib import xla_client
from jax.sharding import Mesh, NamedSharding
from jax.sharding import PartitionSpec as P
from jax.sharding import PositionalSharding
from jaxlib.hlo_helpers import custom_call

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
        return o.astype(q.dtype)

    # n l h d

    def pretty(tensor):
        shape = tensor.shape
        mx = jnp.max(tensor)
        mn = jnp.min(tensor)
        mean = jnp.mean(tensor)
        std = jnp.std(tensor)
        return f'[{shape}: {mn:.3g} | {mean:.3g}±{std:.3g} | {mx:.3g}]'

    q = jax.random.normal(jax.random.PRNGKey(0), [2, 4096, 4, 32]).astype(jnp.float16)
    k = jax.random.normal(jax.random.PRNGKey(1), [2, 4096, 4, 32]).astype(jnp.float16)
    v = jax.random.normal(jax.random.PRNGKey(2), [2, 4096, 4, 32]).astype(jnp.float16)

    @jax.jit
    def fwd(q,k,v):
        return flash_mha(q,k,v)


    # devices = jax.devices(backend='cpu')
    # n_device = len(devices)
    # sharding = PositionalSharding(devices).reshape(-1,1,1,1)#.replicate()

    devices = jax.devices(backend='gpu')
    with Mesh(devices, axis_names=('x',)) as mesh:
        sharding = NamedSharding(mesh, P(None,None,'x',None))
        q = jax.device_put(q, sharding)
        k = jax.device_put(k, sharding)
        v = jax.device_put(v, sharding)
        # jax.debug.visualize_array_sharding(rearrange(q, 'n l h d -> n (l h d)'))
        print(fwd.lower(q,k,v).compile().as_text())

