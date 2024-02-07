# FlashAttention JAX
This repository provides a jax binding to <https://github.com/Dao-AILab/flash-attention>. To avoid depending on pytorch, since torch and jax installations often conflict, this is a fork of the official repo.

Please see [Tri Dao's repo](https://github.com/Dao-AILab/flash-attention) for more information about flash attention.

## Usage

FlashAttention and FlashAttention-2 are free to use and modify (see LICENSE).
Please cite (see below) and credit FlashAttention if you use it.

## Installation and features

Requirements:
- CUDA 11.6 and above.
- Linux. Same story as with the pytorch repo. I haven't tested compilation of the jax bindings on windows.

To install: TODO

Interface: `src/flash_attn_jax/flash.py`

FlashAttention-2 currently supports:
1. Ampere, Ada, or Hopper GPUs (e.g., A100, RTX 3090, RTX 4090, H100). Support for Turing
   GPUs (T4, RTX 2080) is coming soon, please use FlashAttention 1.x for Turing
   GPUs for now.
2. Datatype fp16 and bf16 (bf16 requires Ampere, Ada, or Hopper GPUs).
3. All head dimensions up to 256. Head dim > 192 backward requires A100/A800 or H100/H800.

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
