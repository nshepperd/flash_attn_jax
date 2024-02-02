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

_mha_fwd_p = core.Primitive("mha_fwd")
_mha_fwd_p.multiple_results = True
_mha_fwd_p.def_impl(partial(xla.apply_primitive, _mha_fwd_p))


def mha_fwd(q, k, v):
    return _mha_fwd_p.bind(q, k, v)



# Register functions defined in gpu_ops as custom call target for GPUs
for _name, _value in flash_attn_2_cuda_jax.get_registrations().items():
    xla_client.register_custom_call_target(_name, _value, platform="gpu")

def default_layouts(*shapes):
    return [range(len(shape) - 1, -1, -1) for shape in shapes]

def prod(xs):
    r = 1
    for x in xs:
        r *= x;
    return r

def _mha_fwd_cuda_lowering(ctx, q, k, v):
    q_type = ir.RankedTensorType(q.type)
    q_shape = q_type.shape
    k_type = ir.RankedTensorType(k.type)
    k_shape = k_type.shape
    v_type = ir.RankedTensorType(v.type)
    v_shape = v_type.shape

    assert q_type.element_type == k_type.element_type
    assert q_type.element_type == v_type.element_type
    out_element_type = q_type.element_type
    print(type(out_element_type), out_element_type)
    assert type(out_element_type) in [jaxlib.mlir.ir.F16Type, jaxlib.mlir.ir.BF16Type]
    [n, l, h, d] = q_shape
    [nk, lk, hk, dk] = k_shape

    print(type(ctx))

    out_types = [ir.RankedTensorType.get([n, l, h, d], out_element_type),
                 ir.RankedTensorType.get([n, h, l], jaxlib.mlir.ir.F32Type.get(ctx.module_context.context))]

    assert k_shape == v_shape
    assert [n, d] == [nk, dk]

    opaque = flash_attn_2_cuda_jax.make_mha_fwd_args(
        0.0, # p_dropout
        1.0, # softmax_scale
        False, # is_causal
        -1, # window_size_left
        -1, # window_size_right
        False, # return_softmax
        n, l, h, d,
        lk, hk,
        flash_attn_2_cuda_jax.BF16 if type(out_element_type) == jaxlib.mlir.ir.BF16Type else flash_attn_2_cuda_jax.FP16,
        0)
    out = custom_call(
        b"mha_fwd",
        result_types=out_types,
        operands=[q, k, v],
        backend_config=opaque,
        operand_layouts=default_layouts(q_shape, k_shape, v_shape),
        result_layouts=default_layouts(*[o.shape for o in out_types]),
    ).results
    return out


mlir.register_lowering(
    _mha_fwd_p,
    _mha_fwd_cuda_lowering,
    platform="gpu",
)

from jax.core import ShapedArray
def _mha_fwd_abstract(q, k, v):
    q_dtype = dtypes.canonicalize_dtype(k.dtype)
    k_dtype = dtypes.canonicalize_dtype(k.dtype)
    v_dtype = dtypes.canonicalize_dtype(v.dtype)
    [n, l, h, d] = q.shape
    assert q_dtype == k_dtype and q_dtype == v_dtype
    assert q_dtype in [jnp.bfloat16, jnp.float16]
    return (
        ShapedArray(q.shape, q_dtype, named_shape=q.named_shape),
        ShapedArray([n, h, l], jnp.float32)
    )

_mha_fwd_p.def_abstract_eval(_mha_fwd_abstract)


import time
key = jax.random.PRNGKey(0)
q = jax.random.normal(key, [64, 4096, 4, 32]).astype(jnp.float16)
k = jax.random.normal(key, [64, 4096, 4, 32]).astype(jnp.float16)
v = jax.random.normal(key, [64, 4096, 4, 32]).astype(jnp.float16)
# n l h d
start = time.time()
print(mha_fwd(q,k,v)[0].mean())
print(time.time() - start)
start = time.time()
att = jnp.einsum('nlhd,nLhd->nhlL',q,k)
att = jax.nn.softmax(att, axis=-1)
o = jnp.einsum('nhlL,nLhd->nlhd',att,v)
print(o.mean())
print(time.time() - start)
