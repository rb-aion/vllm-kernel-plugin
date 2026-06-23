"""TileLang RMSNorm kernel + a thin torch wrapper.

Kept free of any vLLM import so it can be tested in isolation
(see tests/test_kernel.py). The wrapper's contract matches vLLM's reference
rms_norm (vllm/ir/ops/layernorm.py):

    out = (x / sqrt(mean(x^2, dim=-1) + eps)) * weight

with the normalization accumulated in float32 and weight applied after the
cast back to the input dtype.
"""

import functools

import torch

import tilelang
import tilelang.language as T

# torch dtype -> TileLang dtype string
_TORCH_TO_TL = {
    torch.float16: "float16",
    torch.bfloat16: "bfloat16",
    torch.float32: "float32",
}


@functools.lru_cache(maxsize=None)
def _compile(m: int, n: int, dtype_str: str, eps: float, blk_m: int):
    """Compile (and cache) a TileLang RMSNorm kernel for a concrete shape.

    TileLang compiles for fixed (M, N). `m` here is already padded to a
    multiple of `blk_m`, so the cache is keyed on the padded row count. Note:
    a new `m` triggers a recompile — acceptable for a first version. To remove
    recompiles, switch to TileLang dynamic shapes (see their dynamic_shape
    example) or bucket the token dimension.
    """
    accum = "float32"

    @T.prim_func
    def kernel(
        X: T.Tensor((m, n), dtype_str),
        Out: T.Tensor((m, n), dtype_str),
    ):
        # One block handles `blk_m` rows; rows are independent in RMSNorm.
        with T.Kernel(T.ceildiv(m, blk_m), threads=128) as bx:
            x_shared = T.alloc_shared((blk_m, n), dtype_str)
            x_f = T.alloc_fragment((blk_m, n), accum)
            x_sq = T.alloc_fragment((blk_m, n), accum)
            inv = T.alloc_fragment((blk_m,), accum)

            T.copy(X[bx * blk_m, 0], x_shared)
            T.copy(x_shared, x_f)  # cast to fp32 for the reduction

            for i, j in T.Parallel(blk_m, n):
                x_sq[i, j] = x_f[i, j] * x_f[i, j]
            T.reduce_sum(x_sq, inv, dim=1)  # sum of squares per row

            for i in T.Parallel(blk_m):
                # NOTE: if your tilelang lacks T.rsqrt, use 1.0 / T.sqrt(...).
                inv[i] = T.rsqrt(inv[i] / n + eps)

            for i, j in T.Parallel(blk_m, n):
                x_f[i, j] = x_f[i, j] * inv[i]

            T.copy(x_f, Out[bx * blk_m, 0])  # cast fp32 -> input dtype

    # out_idx=[1]: TileLang allocates+returns the 2nd arg (Out).
    return tilelang.compile(kernel, out_idx=[1])


def tilelang_rms_norm(
    x: torch.Tensor,
    weight: torch.Tensor,
    eps: float,
    blk_m: int = 1,
) -> torch.Tensor:
    """RMSNorm via TileLang.

    Args:
        x: (num_tokens, hidden), fp16/bf16/fp32, CUDA.
        weight: (hidden,), broadcast over rows.
        eps: variance epsilon.
        blk_m: rows processed per CUDA block.

    Returns:
        (num_tokens, hidden) tensor, same dtype/device as `x`.
    """
    assert x.is_cuda, "TileLang kernel requires a CUDA tensor"
    assert x.dim() == 2, "pass a 2-D (num_tokens, hidden) tensor"
    m, n = x.shape
    dtype_str = _TORCH_TO_TL[x.dtype]

    # Pad rows up to a multiple of blk_m so the last block is fully in-bounds.
    # Padded rows have variance 0 -> rsqrt(eps); they are sliced off below.
    pad = (-m) % blk_m
    x_in = x.contiguous()
    if pad:
        x_in = torch.nn.functional.pad(x_in, (0, 0, 0, pad))

    kernel = _compile(x_in.shape[0], n, dtype_str, float(eps), blk_m)
    out = kernel(x_in)
    if pad:
        out = out[:m]

    # Weight applied after the cast-back, matching vLLM's reference order.
    return out * weight.to(out.dtype)
