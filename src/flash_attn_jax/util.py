import math
from jax.experimental.custom_partitioning import (
    ArrayMapping,
    CompoundFactor,
)
import re

def ceildiv(a: int, b: int) -> int:
    return (a + b - 1) // b

def num_splits_heuristic(batch_nheads_mblocks: int, num_SMs: int, num_n_blocks: int, max_splits: int) -> int:
    if batch_nheads_mblocks >= 0.8 * num_SMs:
        return 1

    max_splits = min(max_splits, num_SMs, num_n_blocks)

    def eligible(s):
        return s == 1 or ceildiv(num_n_blocks, s) != ceildiv(num_n_blocks, s - 1)

    effs = [
        (batch_nheads_mblocks * s / num_SMs) / math.ceil(batch_nheads_mblocks * s / num_SMs)
        if eligible(s) else 0.0
        for s in range(1, max_splits + 1)
    ]

    max_eff = max(effs, default=0.0)

    return next((s for s, eff in enumerate(effs, 1) if eligible(s) and eff >= 0.85 * max_eff), 1)

# auto round_multiple = [](int x, int m) { return (x + m - 1) / m * m; };
def round_multiple(x: int, m: int) -> int:
    return (x + m - 1) // m * m

def array_mapping(spec: str) -> ArrayMapping:
    """Parse a spec string like 'n l (h g) d' into an ArrayMapping.

    - Simple dimensions: single letters/words become string factors
    - Compound factors: parenthesized groups like (h g) become CompoundFactor
    """
    factors = []
    for token in re.findall(r'\([^)]+\)|\S+', spec):
        if token.startswith('('):
            factors.append(CompoundFactor(*token[1:-1].split()))
        else:
            factors.append(token)
    return ArrayMapping(*factors)