# FlashAttention JAX
This repository provides a jax binding to <https://github.com/Dao-AILab/flash-attention>. To avoid depending on pytorch, since torch and jax installations often conflict, this is a fork of the official repo.

Please see [Tri Dao's repo](https://github.com/Dao-AILab/flash-attention) for more information about flash attention.

FlashAttention and FlashAttention-2 are free to use and modify (see LICENSE).
Please cite (see below) and credit FlashAttention if you use it.

## Installation

Requirements:
- CUDA 11.8 and above.
- Linux. Same story as with the pytorch repo. I haven't tested compilation of the jax bindings on windows.
- JAX >=`0.4.24`. The custom sharding used for ring attention requires some somewhat advanced features.

To install: `pip install flash-attn-jax` will get the latest release from pypi. This gives you the cuda 12.3 build. If you want to use the cuda 11.8 build, you can install from the releases page (but according to jax's documentation, 11.8 will stop being supported for newer versions of jax).

### Installing from source

Flash attention takes a long time to compile unless you have a powerful machine. But if you want to compile from source, I use `cibuildwheel` to compile the releases. You could do the same. Something like (for python 3.12):

```sh
git clone https://github.com/nshepperd/flash-attn-jax
cd flash-attn-jax
cibuildwheel --only cp312-manylinux_x86_64 # I think cibuildwheel needs superuser privileges on some systems because of docker reasons?
```

This will create a wheel in the `wheelhouse` directory. You can then install it with `pip install wheelhouse/flash_attn_jax_0.2.0-cp312-cp312-manylinux_x86_64.whl`. Or you could use setup.py to build the wheel and install it. You need cuda toolkit installed in that case.

## Usage

Interface: `src/flash_attn_jax/flash.py`

```py
from flash_attn_jax import flash_mha

# flash_mha : [n, l, h, d] x [n, lk, hk, d] x [n, lk, hk, d] -> [n, l, h, d]
flash_mha(q,k,v,softmax_scale=None, is_causal=False, window_size=(-1,-1))
```

This supports multi-query and grouped-query attention (when hk != h). The `softmax_scale` is the multiplier for the softmax, defaulting to `1/sqrt(d)`. Set `window_size` to positive values for sliding window attention.

### Now Supports Ring Attention

Use jax.Array and shard your tensors along the length dimension, and flash_mha will automatically use the ring attention algorithm (forward and backward).

```py
os.environ["XLA_FLAGS"] = '--xla_gpu_enable_latency_hiding_scheduler=true --xla_gpu_enable_async_collectives=true'
#...
with Mesh(devices, axis_names=('len',)) as mesh:
        sharding = NamedSharding(mesh, P(None,'len')) # n l
        tokens = jax.device_put(tokens, sharding)
        # invoke your jax.jit'd transformer.forward
```

It's not entirely reliable at hiding the communication latency though, depending on the whims of the xla optimizer. I'm waiting https://github.com/google/jax/issues/20864 to be fixed, then I can make it better.

### GPU support

FlashAttention-2 currently supports:
1. Ampere, Ada, or Hopper GPUs (e.g., A100, RTX 3090, RTX 4090, H100). Support for Turing
   GPUs (T4, RTX 2080) is coming soon, please use FlashAttention 1.x for Turing
   GPUs for now.
2. Datatype fp16 and bf16 (bf16 requires Ampere, Ada, or Hopper GPUs).
3. All head dimensions up to 256. ~~Head dim > 192 backward requires A100/A800 or H100/H800~~. Head dim 256 backward now works on consumer GPUs (if there's no dropout) as of flash-attn 2.5.5.

## Citation
If you use this codebase, or otherwise found our work valuable, please cite:
```
@inproceedings{dao2022flashattention,
  title={Flash{A}ttention: Fast and Memory-Efficient Exact Attention with {IO}-Awareness},
  author={Dao, Tri and Fu, Daniel Y. and Ermon, Stefano and Rudra, Atri and R{\'e}, Christopher},
  booktitle={Advances in Neural Information Processing Systems},
  year={2022}
}
@article{dao2023flashattention2,
  title={Flash{A}ttention-2: Faster Attention with Better Parallelism and Work Partitioning},
  author={Dao, Tri},
  year={2023}
}
```
