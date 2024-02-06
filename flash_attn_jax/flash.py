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
import flash_attn_2_cuda_jax

_flash_mha_fwd_p = core.Primitive("flash_mha_fwd")
_flash_mha_fwd_p.multiple_results = True
_flash_mha_fwd_p.def_impl(partial(xla.apply_primitive, _flash_mha_fwd_p))

_flash_mha_bwd_p = core.Primitive("flash_mha_bwd")
_flash_mha_bwd_p.multiple_results = True
_flash_mha_bwd_p.def_impl(partial(xla.apply_primitive, _flash_mha_bwd_p))


def flash_mha_fwd(q, k, v, softmax_scale=1.0):
    return _flash_mha_fwd_p.bind(q, k, v, softmax_scale=softmax_scale)

def flash_mha_bwd(dout, q, k, v, out, lse, softmax_scale=1.0):
    return _flash_mha_bwd_p.bind(dout, q, k, v, out, lse, softmax_scale=softmax_scale)

# Register functions defined in gpu_ops as custom call target for GPUs
for _name, _value in flash_attn_2_cuda_jax.get_registrations().items():
    xla_client.register_custom_call_target(_name, _value, platform="gpu")

def default_layouts(*shapes):
    return [range(len(shape) - 1, -1, -1) for shape in shapes]

def _flash_mha_fwd_cuda_lowering(ctx, q, k, v, softmax_scale=1.0):
    print(type(q), dir(q), q.type)
    q_type = ir.RankedTensorType(q.type)
    q_shape = q_type.shape
    k_type = ir.RankedTensorType(k.type)
    k_shape = k_type.shape
    v_type = ir.RankedTensorType(v.type)
    v_shape = v_type.shape

    assert q_type.element_type == k_type.element_type
    assert q_type.element_type == v_type.element_type
    out_element_type = q_type.element_type
    assert type(out_element_type) in [jaxlib.mlir.ir.F16Type, jaxlib.mlir.ir.BF16Type]
    [n, l, h, d] = q_shape
    [nk, lk, hk, dk] = k_shape

    out_types = [ir.RankedTensorType.get([n, l, h, d], out_element_type),
                 ir.RankedTensorType.get([n, h, l], jaxlib.mlir.ir.F32Type.get(ctx.module_context.context))]

    assert k_shape == v_shape
    assert [n, d] == [nk, dk]

    opaque = flash_attn_2_cuda_jax.make_flash_mha_fwd_args(
        0.0, # p_dropout
        softmax_scale,
        False, # is_causal
        -1, # window_size_left
        -1, # window_size_right
        False, # return_softmax
        n, l, h, d,
        lk, hk,
        flash_attn_2_cuda_jax.BF16 if type(out_element_type) == jaxlib.mlir.ir.BF16Type else flash_attn_2_cuda_jax.FP16,
        0)
    out = custom_call(
        b"flash_mha_fwd",
        result_types=out_types,
        operands=[q, k, v],
        backend_config=opaque,
        operand_layouts=default_layouts(q_shape, k_shape, v_shape),
        result_layouts=default_layouts(*[o.shape for o in out_types]),
    ).results
    return out

mlir.register_lowering(
    _flash_mha_fwd_p,
    _flash_mha_fwd_cuda_lowering,
    platform="gpu",
)

def _flash_mha_bwd_cuda_lowering(ctx, dout, q, k, v, out, lse, softmax_scale=1.0):
    dout_type = ir.RankedTensorType(dout.type).element_type
    q_type = ir.RankedTensorType(q.type).element_type
    k_type = ir.RankedTensorType(k.type).element_type
    v_type = ir.RankedTensorType(v.type).element_type
    out_type = ir.RankedTensorType(out.type).element_type
    lse_type = ir.RankedTensorType(lse.type).element_type

    assert type(q_type) in [jaxlib.mlir.ir.F16Type, jaxlib.mlir.ir.BF16Type]
    assert q_type == dout_type
    assert q_type == k_type
    assert q_type == v_type
    assert q_type == out_type
    assert type(lse_type) in [jaxlib.mlir.ir.F32Type]

    dout_shape = ir.RankedTensorType(dout.type).shape
    q_shape = ir.RankedTensorType(q.type).shape
    k_shape = ir.RankedTensorType(k.type).shape
    v_shape = ir.RankedTensorType(v.type).shape
    out_shape = ir.RankedTensorType(out.type).shape
    lse_shape = ir.RankedTensorType(lse.type).shape
    [n, lq, hq, d] = q_shape
    [_, lk, hk, _] = k_shape

    print(type(dout_shape))

    assert (list(map(list, [dout_shape, q_shape, k_shape, v_shape, out_shape, lse_shape])) ==
            [[n, lq, hq, d], [n, lq, hq, d], [n, lk, hk, d], [n, lk, hk, d],
             [n, lq, hq, d], [n, hq, lq]])

    out_types = [ir.RankedTensorType.get(q_shape, q_type),
                 ir.RankedTensorType.get(k_shape, k_type),
                 ir.RankedTensorType.get(v_shape, v_type)]

    opaque = flash_attn_2_cuda_jax.make_flash_mha_bwd_args(
        0.0, # p_dropout
        softmax_scale,
        False, # is_causal
        -1, # window_size_left
        -1, # window_size_right
        False, # deterministic
        n, lq, hq, d,
        lk, hk,
        flash_attn_2_cuda_jax.BF16 if type(q_type) == jaxlib.mlir.ir.BF16Type else flash_attn_2_cuda_jax.FP16,
        0)
    out = custom_call(
        b"flash_mha_bwd",
        result_types=out_types,
        operands=[dout, q, k, v, out, lse],
        backend_config=opaque,
        operand_layouts=default_layouts(dout_shape, q_shape, k_shape, v_shape, out_shape, lse_shape),
        result_layouts=default_layouts(*[o.shape for o in out_types]),
    ).results
    return out

mlir.register_lowering(
    _flash_mha_bwd_p,
    _flash_mha_bwd_cuda_lowering,
    platform="gpu",
)

from jax.core import ShapedArray
def _flash_mha_fwd_abstract(q, k, v, softmax_scale=1.0):
    q_dtype = dtypes.canonicalize_dtype(q.dtype)
    k_dtype = dtypes.canonicalize_dtype(k.dtype)
    v_dtype = dtypes.canonicalize_dtype(v.dtype)
    [n, l, h, d] = q.shape
    assert q_dtype == k_dtype and q_dtype == v_dtype
    assert q_dtype in [jnp.bfloat16, jnp.float16]
    return (
        ShapedArray(q.shape, q_dtype, named_shape=q.named_shape),
        ShapedArray([n, h, l], jnp.float32)
    )

_flash_mha_fwd_p.def_abstract_eval(_flash_mha_fwd_abstract)


def _flash_mha_bwd_abstract(dout, q, k, v, out, lse, softmax_scale=1.0):
    dout_dtype = dtypes.canonicalize_dtype(dout.dtype)
    q_dtype = dtypes.canonicalize_dtype(q.dtype)
    k_dtype = dtypes.canonicalize_dtype(k.dtype)
    v_dtype = dtypes.canonicalize_dtype(v.dtype)
    out_dtype = dtypes.canonicalize_dtype(out.dtype)
    lse_dtype = dtypes.canonicalize_dtype(lse.dtype)
    [n, lq, hq, d] = q.shape
    assert len(set([dout_dtype, q_dtype, k_dtype, v_dtype, out_dtype])) == 1
    assert q_dtype in [jnp.bfloat16, jnp.float16]
    return (
        ShapedArray(q.shape, q_dtype, named_shape=q.named_shape),
        ShapedArray(k.shape, k_dtype, named_shape=k.named_shape),
        ShapedArray(v.shape, v_dtype, named_shape=v.named_shape),
    )
_flash_mha_bwd_p.def_abstract_eval(_flash_mha_bwd_abstract)

def mha_fwd_batch(vector_arg_values, batch_axes, **kwargs):
  assert tuple(batch_axes) == (0,0,0), "Only support vmapping mha over axis 0 for now,"
  [q, k, v] = vector_arg_values
  [b, n, l, h, d] = q.shape
  [b, n, lk, hk, d] = k.shape
  assert [b, n, lk, hk, d] == list(v.shape)
  out, lse = flash_mha_fwd(q.reshape([b*n,l,h,d]),
                           k.reshape([b*n,lk,hk,d]),
                           v.reshape([b*n,lk,hk,d]),
                           **kwargs)
  return (out.reshape([b,n,*out.shape[1:]]), lse.reshape([b,n,*lse.shape[1:]])), (0,0)

from einops import rearrange
def mha_bwd_batch(vector_arg_values, batch_axes, **kwargs):
  assert tuple(batch_axes) == (0,0,0,0,0,0), "Only support vmapping mha over axis 0 for now,"
  dout, q, k, v, out, lse = vector_arg_values
  b = dout.shape[batch_axes[0]]
  def join(*args):
      return [rearrange(a, 'b n ... -> (b n) ...') for a in args]
  def unjoin(*args):
      return [rearrange(a, '(b n) ... -> b n ...', b=b) for a in args]
  dq, dk, dv = flash_mha_fwd(*join(dout,q,k,v,out,lse),
                             **kwargs)
  return tuple(unjoin(dq,dk,dv)), (0,0,0)

from jax.interpreters import batching
batching.primitive_batchers[_flash_mha_fwd_p] = mha_fwd_batch
batching.primitive_batchers[_flash_mha_bwd_p] = mha_bwd_batch

if __name__ == '__main__':
    import time
    import numpy as np

    @jax.jit
    def pure_mha(q,k,v, softmax_scale=1.0):
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

    q = jax.random.normal(jax.random.PRNGKey(0), [2, 4096, 4, 32]).astype(jnp.float16)
    k = jax.random.normal(jax.random.PRNGKey(1), [2, 4096, 4, 32]).astype(jnp.float16)
    v = jax.random.normal(jax.random.PRNGKey(2), [2, 4096, 4, 32]).astype(jnp.float16)

    @jax.jit
    def fwd(q,k,v):
        return flash_mha_fwd(q,k,v, softmax_scale=float(np.sqrt(1/32)))[0]

    from jax.sharding import PositionalSharding
    from einops import rearrange

    sharding = PositionalSharding([*jax.devices(), *jax.devices(backend='cpu')])

    q = jax.device_put(q, sharding.reshape(2,1,1,1))
    k = jax.device_put(k, sharding.reshape(2,1,1,1))
    v = jax.device_put(v, sharding.reshape(2,1,1,1))
    jax.debug.visualize_array_sharding(rearrange(q, 'n l h d -> n (l h d)'))
    print(fwd.lower(q,k,v).compile().as_text())
    # exit()

    print('==== forward ====')
    q = jax.random.normal(jax.random.PRNGKey(0), [64, 4096, 4, 32]).astype(jnp.float16)
    k = jax.random.normal(jax.random.PRNGKey(1), [64, 4096, 4, 32]).astype(jnp.float16)
    v = jax.random.normal(jax.random.PRNGKey(2), [64, 4096, 4, 32]).astype(jnp.float16)
    @jax.jit
    def fwd(q,k,v):
        o = flash_mha_fwd(q,k,v, softmax_scale=float(np.sqrt(1/32)))[0]
        for _ in range(32):
            o = flash_mha_fwd(q,k,o, softmax_scale=float(np.sqrt(1/32)))[0]
        return o
    @jax.jit
    def fwd_jax(q,k,v):
        ro = pure_mha(q,k,v, softmax_scale=float(np.sqrt(1/32)))
        for _ in range(32):
            ro = pure_mha(q,k,ro, softmax_scale=float(np.sqrt(1/32)))
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

    @jax.grad
    def grad_pure(inputs, softmax_scale=1.0):
        q,k,v = inputs
        return pure_mha(q,k,v, softmax_scale=softmax_scale).sum()

    print('==== backward ====')
    q = jax.random.normal(jax.random.PRNGKey(0), [1, 4, 2, 32]).astype(jnp.float16)
    k = jax.random.normal(jax.random.PRNGKey(1), [1, 4, 2, 32]).astype(jnp.float16)
    v = jax.random.normal(jax.random.PRNGKey(2), [1, 4, 2, 32]).astype(jnp.float16)
    scale = float(np.sqrt(1/32))
    out, lse = flash_mha_fwd(q,k,v, softmax_scale=scale)
    dout = jnp.ones_like(out)
    dq, dk, dv = flash_mha_bwd(dout, q, k, v, out, lse, softmax_scale=scale)

    rdq, rdk, rdv = grad_pure((q,k,v), softmax_scale=scale)

    # print(rdq, jnp.mean(jnp.abs(rdq)))

    print('q', pretty(jnp.abs(dq - rdq)), jnp.mean(jnp.abs(rdq)))
    print('k', pretty(jnp.abs(dk - rdk)), jnp.mean(jnp.abs(rdk)))
    print('v', pretty(jnp.abs(dv - rdv)), jnp.mean(jnp.abs(rdv)))
