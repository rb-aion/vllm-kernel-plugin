"""Correctness test for the TileLang RMSNorm kernel (no vLLM needed).

Run on a CUDA machine:
    uv pip install -e ".[test]"
    .venv/bin/python -m pytest tests/test_kernel.py -v
"""

import pytest
import torch

cuda_only = pytest.mark.skipif(
    not torch.cuda.is_available(), reason="TileLang kernel needs CUDA"
)


def _reference(x: torch.Tensor, w: torch.Tensor, eps: float) -> torch.Tensor:
    """Matches vllm/ir/ops/layernorm.py:rms_norm."""
    o = x.to(torch.float32)
    var = o.pow(2).mean(dim=-1, keepdim=True)
    o = o * torch.rsqrt(var + eps)
    return (o.to(w.dtype) * w).to(x.dtype)


@cuda_only
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
@pytest.mark.parametrize("shape", [(8, 128), (128, 4096), (1, 8192), (333, 1024)])
def test_matches_reference(dtype, shape):
    from vllm_tilelang_rmsnorm.kernel import tilelang_rms_norm

    torch.manual_seed(0)
    m, n = shape
    x = torch.randn(m, n, dtype=dtype, device="cuda")
    w = torch.randn(n, dtype=dtype, device="cuda")
    eps = 1e-6

    out = tilelang_rms_norm(x, w, eps)
    exp = _reference(x, w, eps)

    # Same tolerance vLLM uses for rms_norm in float16 (override_tolerance).
    torch.testing.assert_close(out, exp, atol=1e-2, rtol=2e-3)
