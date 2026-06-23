"""Verify the plugin registers and swaps the RMSNorm class (needs vLLM, not CUDA)."""

import pytest

vllm = pytest.importorskip("vllm")


def test_register_oot_swaps_class():
    import vllm_tilelang_rmsnorm
    from vllm.model_executor.custom_op import op_registry_oot

    vllm_tilelang_rmsnorm.register()

    assert "RMSNorm" in op_registry_oot
    assert op_registry_oot["RMSNorm"].__name__ == "TileLangRMSNorm"
