from functools import partial, wraps
from typing import Tuple

import numpy as np
import jax
import jax.numpy as jnp
from jax import core, dtypes
from jax.core import ShapedArray
from jax.interpreters import mlir
from jax.interpreters import xla
from jax.interpreters.mlir import ir

from jax.extend.core import Primitive

from einops import rearrange
import einops
import math

import flash_attn_jax_lib.flash_api as flash_api

def register_custom_calls():
    # Register functions defined in gpu_ops as custom call target for GPUs
    for _name, _value in flash_api.get_ffi_registrations().items():
        jax.ffi.register_ffi_target(_name, _value, platform="CUDA")
