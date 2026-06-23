"""vLLM plugin: replace RMSNorm with a TileLang kernel.

The entry point declared in pyproject.toml ("vllm.general_plugins") makes vLLM
import and call register() once per process. register() imports the `op` module,
whose @RMSNorm.register_oot decorator does the actual swap.
"""


def register() -> None:
    # Importing `op` runs the @RMSNorm.register_oot decorator as a side effect,
    # which inserts TileLangRMSNorm into vLLM's op_registry_oot. After that,
    # every RMSNorm(...) construction is transparently swapped (via
    # CustomOp.__new__) for our subclass.
    from . import op  # noqa: F401

    try:
        from vllm.logger import init_logger

        init_logger(__name__).info("TileLang RMSNorm plugin registered.")
    except Exception:
        # Logging is best-effort; never let it break plugin loading.
        pass
