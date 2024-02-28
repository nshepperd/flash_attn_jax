# FlashAttention JAX
This repository provides a jax binding to <https://github.com/Dao-AILab/flash-attention>. To avoid depending on pytorch, since torch and jax installations often conflict, this is a fork of the official repo.

Please see [Tri Dao's repo](https://github.com/Dao-AILab/flash-attention) for more information about flash attention.

## Usage

FlashAttention and FlashAttention-2 are free to use and modify (see LICENSE).
Please cite (see below) and credit FlashAttention if you use it.

## Installation and features

Requirements:
- CUDA 11.8 and above.
- Linux. Same story as with the pytorch repo. I haven't tested compilation of the jax bindings on windows.
- JAX >=`0.4.24`. The custom sharding used for ring attention requires some somewhat advanced features.

To install: For now, download the appropriate release from the releases page and install it with pip.

Interface: `src/flash_attn_jax/flash.py`

```py
from flash_attn_jax import flash_mha

flash_mha(q,k,v,softmax_scale=None, is_causal=False, window_size=(-1,-1))
```

Accepts q,k,v with shape `[n, l, h, d]`, and returns `[n, l, h, d]`. `softmax_scale` is the
multiplier for the softmax, defaulting to `1/sqrt(d)`. Set window_size
to positive values for sliding window attention.

### Now Supports Ring Attention

Use jax.Array and shard your tensors along the length dimension, and flash_mha will automatically use the ring attention algorithm:

```py
with Mesh(devices, axis_names=('len',)) as mesh:
        sharding = NamedSharding(mesh, P(None,'len',None)) # n l d
        tokens = jax.device_put(tokens, sharding)
        # invoke your jax.jit'd transformer.forward
```

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
