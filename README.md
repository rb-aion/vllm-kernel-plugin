# vllm-tilelang-rmsnorm

A vLLM plugin that replaces `RMSNorm` with a [TileLang](https://github.com/tile-ai/tilelang) kernel.

It uses two vLLM mechanisms:

1. **Plugin system** (`vllm.general_plugins` entry point) — gets our code loaded
   into every vLLM process. vLLM calls `register()` at startup.
2. **CustomOp `register_oot`** — `@RMSNorm.register_oot` swaps our subclass in
   for vLLM's built-in `RMSNorm`. On CUDA the dispatcher calls our
   `forward_cuda`, which runs the TileLang kernel.

```
pyproject entry point ──► vllm/plugins loader ──► register() ──► import op
   ──► @RMSNorm.register_oot ──► TileLangRMSNorm replaces RMSNorm
```

## Layout

| File | Role |
|------|------|
| `vllm_tilelang_rmsnorm/kernel.py` | TileLang kernel + torch wrapper (no vLLM import) |
| `vllm_tilelang_rmsnorm/op.py` | `TileLangRMSNorm(RMSNorm)` — the vLLM glue |
| `vllm_tilelang_rmsnorm/__init__.py` | `register()` (the plugin entry point) |
| `tests/test_kernel.py` | kernel correctness vs reference (CUDA) |
| `tests/test_registration.py` | confirms the class swap (vLLM) |

## Install

In the **same environment as vLLM** (per vLLM's AGENTS.md, use `uv`):

```bash
uv pip install -e ".[test]"
```

## Verify in stages

1. **Kernel math** (no vLLM):
   ```bash
   .venv/bin/python -m pytest tests/test_kernel.py -v
   ```
2. **Registration** (no CUDA):
   ```bash
   .venv/bin/python -m pytest tests/test_registration.py -v
   ```
3. **End-to-end in vLLM** — run with `--enforce-eager` (see gotcha below):
   ```bash
   vllm serve <model> --enforce-eager
   ```
   You should see "TileLang RMSNorm plugin registered." in the logs.

## Gotcha: the op must be *enabled*

`CustomOp` dispatch only routes to `forward_cuda` when the op is enabled. Under
the default `torch.compile`/Inductor path, custom ops default **off** and vLLM
uses `forward_native` — silently ignoring our kernel. Force it on with either:

- `--enforce-eager`, or
- `--compilation-config '{"custom_ops": ["+rms_norm"]}'`

To prove the kernel runs, temporarily `raise` inside `forward_cuda`: if
generation still works, the op wasn't enabled.

## Known limitations (v1)

- The kernel recompiles per unique `(num_tokens, hidden, dtype)`. Fine to start;
  use TileLang dynamic shapes or token bucketing to remove recompiles.
- `variance_size_override` falls back to the native implementation.
- Tuned for correctness, not speed (`blk_m=1`, weight applied in torch).

## Status

The vLLM integration is complete. The TileLang DSL in `kernel.py` targets a
recent `tilelang`; a couple of constructs (`T.rsqrt`, `T.reduce_sum` signature)
are version-sensitive and noted in comments. `tests/test_kernel.py` is the gate —
run it on your GPU and reconcile any DSL differences with your installed version.
