"""The vLLM glue: a RMSNorm subclass whose CUDA path runs the TileLang kernel.

@RMSNorm.register_oot registers this class under the name "RMSNorm" in vLLM's
op_registry_oot. CustomOp.__new__ then substitutes it whenever vLLM builds an
RMSNorm, on any platform. On CUDA the dispatcher routes forward() to
forward_cuda (see vllm/model_executor/custom_op.py:dispatch_forward), so our
kernel runs there.
"""

import torch

from vllm.model_executor.layers.layernorm import RMSNorm

from .kernel import tilelang_rms_norm


@RMSNorm.register_oot
class TileLangRMSNorm(RMSNorm):
    def forward_cuda(
        self,
        x: torch.Tensor,
        residual: torch.Tensor | None = None,
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        # The kernel normalizes over the full last dim only. vLLM's
        # variance_size_override (partial-width variance) is rare and only the
        # native impl supports it — fall back for correctness.
        if self.variance_size_override is not None:
            return self.forward_native(x, residual)

        residual_out = None
        if residual is not None:
            # Fused-add contract: x + residual becomes the new residual, then
            # the sum is normalized. Mirrors RMSNorm.forward_native.
            x = x + residual
            residual_out = x

        orig_shape = x.shape
        x_2d = x.reshape(-1, x.shape[-1])
        out = tilelang_rms_norm(x_2d, self.weight.data, self.variance_epsilon)
        out = out.reshape(orig_shape)

        if residual_out is not None:
            return out, residual_out
        return out
