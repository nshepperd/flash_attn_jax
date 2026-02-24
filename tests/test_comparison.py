"""Unit tests with strict numerics comparing to pytorch's flash attention."""

import numpy as np
import pytest
import hypothesis.strategies as st
import jax
import jax.numpy as jnp
from hypothesis import given, settings
from jax.tree_util import tree_map

try:
    import torch
    from flash_attn import flash_attn_func
    HAS_TORCH_FLASH = True
except ImportError:
    HAS_TORCH_FLASH = False

from flash_attn_jax import flash_mha

pytestmark = pytest.mark.skipif(not HAS_TORCH_FLASH, reason="torch or flash_attn not installed")


def jax_to_torch(x):
    """Convert a JAX array to a PyTorch CUDA tensor, preserving dtype."""
    if x.dtype == jnp.bfloat16:
        # numpy doesn't support bfloat16, round-trip through float32 (lossless)
        return torch.from_numpy(np.array(x, dtype=np.float32)).to(torch.bfloat16).cuda()
    return torch.from_numpy(np.array(x)).cuda()


def torch_to_jax(x):
    """Convert a PyTorch tensor to a JAX array, preserving dtype."""
    if x.dtype == torch.bfloat16:
        return jnp.array(x.float().cpu().numpy()).astype(jnp.bfloat16)
    return jnp.array(x.detach().cpu().numpy())


def check(jax_out, torch_out, rtol=1e-5, atol=1e-5):
    def check1(jax_out, torch_out):
        np.testing.assert_allclose(
            np.array(jax_out, dtype=np.float32),
            np.array(torch_out, dtype=np.float32),
            rtol=rtol, atol=atol,
        )
    tree_map(check1, jax_out, torch_out)


@settings(deadline=None)
@given(d=st.integers(min_value=1, max_value=128),
       h=st.integers(min_value=1, max_value=8),
       seqlen=st.integers(min_value=1, max_value=16384),
       n=st.integers(min_value=1, max_value=2),
       is_causal=st.booleans(),
       dtype=st.sampled_from([jnp.float16, jnp.bfloat16]))
def test_fwd(n, seqlen, h, d, dtype, is_causal):
    q = jax.random.normal(jax.random.PRNGKey(0), [n, seqlen, h, d], dtype=dtype)
    k = jax.random.normal(jax.random.PRNGKey(1), [n, seqlen, h, d], dtype=dtype)
    v = jax.random.normal(jax.random.PRNGKey(2), [n, seqlen, h, d], dtype=dtype)

    jax_out = flash_mha(q, k, v, is_causal=is_causal)

    with torch.no_grad():
        torch_out = flash_attn_func(jax_to_torch(q), jax_to_torch(k), jax_to_torch(v), causal=is_causal)

    check(jax_out, torch_to_jax(torch_out))


@pytest.mark.parametrize("m", [1,2]) 
@settings(deadline=None)
@given(d=st.integers(min_value=1, max_value=128),
       h=st.integers(min_value=1, max_value=8),
       seqlen_q=st.integers(min_value=1, max_value=16384),
       seqlen_k=st.integers(min_value=1, max_value=16384),
       n=st.integers(min_value=1, max_value=2),
       is_causal=st.booleans(),
       dtype=st.sampled_from([jnp.float16, jnp.bfloat16]))
def test_cross_fwd(n, seqlen_q, seqlen_k, h, d, m, dtype, is_causal):
    if m == 2 and seqlen_q == 1:
        pytest.xfail("seqlenq_ngroups_swapped not implemented yet")

    q = jax.random.normal(jax.random.PRNGKey(0), [n, seqlen_q, h*m, d], dtype=dtype)
    k = jax.random.normal(jax.random.PRNGKey(1), [n, seqlen_k, h, d], dtype=dtype)
    v = jax.random.normal(jax.random.PRNGKey(2), [n, seqlen_k, h, d], dtype=dtype)

    jax_out = flash_mha(q, k, v, is_causal=is_causal)

    with torch.no_grad():
        torch_out = flash_attn_func(jax_to_torch(q), jax_to_torch(k), jax_to_torch(v), causal=is_causal)

    check(jax_out, torch_to_jax(torch_out))


@settings(deadline=None)
@given(d=st.integers(min_value=1, max_value=128),
       h=st.integers(min_value=1, max_value=8),
       seqlen=st.integers(min_value=1, max_value=16384),
       n=st.integers(min_value=1, max_value=2),
       is_causal=st.booleans(),
       dtype=st.sampled_from([jnp.float16, jnp.bfloat16]))
def test_bwd(n, seqlen, h, d, dtype, is_causal):
    q = jax.random.normal(jax.random.PRNGKey(0), [n, seqlen, h, d], dtype=dtype)
    k = jax.random.normal(jax.random.PRNGKey(1), [n, seqlen, h, d], dtype=dtype)
    v = jax.random.normal(jax.random.PRNGKey(2), [n, seqlen, h, d], dtype=dtype)

    @jax.grad
    def jax_grads(qkv):
        return flash_mha(*qkv, is_causal=is_causal).sum()
    jax_dq, jax_dk, jax_dv = jax_grads((q, k, v))

    q_pt = jax_to_torch(q).requires_grad_(True)
    k_pt = jax_to_torch(k).requires_grad_(True)
    v_pt = jax_to_torch(v).requires_grad_(True)
    torch_out = flash_attn_func(q_pt, k_pt, v_pt, causal=is_causal)
    torch_out.sum().backward()

    check((jax_dq, jax_dk, jax_dv),
          (torch_to_jax(q_pt.grad), torch_to_jax(k_pt.grad), torch_to_jax(v_pt.grad)),
          rtol=1e-3, atol=1e-3)


@pytest.mark.parametrize("m", [1,2])
@settings(deadline=None)
@given(d=st.integers(min_value=1, max_value=128),
       h=st.integers(min_value=1, max_value=8),
       seqlen_q=st.integers(min_value=1, max_value=16384),
       seqlen_k=st.integers(min_value=1, max_value=16384),
       n=st.integers(min_value=1, max_value=2),
       is_causal=st.booleans(),
       dtype=st.sampled_from([jnp.float16, jnp.bfloat16]))
def test_cross_bwd(n, seqlen_q, seqlen_k, h, d, m, dtype, is_causal):
    q = jax.random.normal(jax.random.PRNGKey(0), [n, seqlen_q, h*m, d], dtype=dtype)
    k = jax.random.normal(jax.random.PRNGKey(1), [n, seqlen_k, h, d], dtype=dtype)
    v = jax.random.normal(jax.random.PRNGKey(2), [n, seqlen_k, h, d], dtype=dtype)

    @jax.grad
    def jax_grads(qkv):
        return flash_mha(*qkv, is_causal=is_causal).sum()
    jax_dq, jax_dk, jax_dv = jax_grads((q, k, v))

    q_pt = jax_to_torch(q).requires_grad_(True)
    k_pt = jax_to_torch(k).requires_grad_(True)
    v_pt = jax_to_torch(v).requires_grad_(True)
    torch_out = flash_attn_func(q_pt, k_pt, v_pt, causal=is_causal)
    torch_out.sum().backward()

    check((jax_dq, jax_dk, jax_dv),
          (torch_to_jax(q_pt.grad), torch_to_jax(k_pt.grad), torch_to_jax(v_pt.grad)),
          rtol=1e-3, atol=1e-3)


if __name__ == "__main__":
    test_cross_bwd.__wrapped_target(m=1,
           d=1,
           h=4,
           seqlen_q=83,
           seqlen_k=258,
           n=1,
           is_causal=False,
           dtype=jax.numpy.float16)