# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""AMD gfx95 (MI355X) SGMV dispatch for multi-LoRA serving.

The Punica-style multi-LoRA path in vLLM originally gated its fast
``segment_gemm_v`` (SGMV) kernel to NVIDIA CUDA.  On AMD ROCm the path fell
back to the Triton ``lora_shrink`` / ``lora_expand`` kernels.  This module
wires an AMD-native SGMV path — backed by AITER ``batched_gemm_bf16`` on
gfx950 (MI355X) — so a single engine can serve multiple LoRA adapters with
weight memory ≈ base_mem + N × adapter_mem.

Dispatch policy
---------------
* **gfx95 (MI355X / MI350)** → ``vllm.lora.kernels._amd_sgmv`` (AITER backend)
* **other ROCm / non-gfx95** → Triton ``lora_shrink`` / ``lora_expand`` fallback
* **NVIDIA CUDA** → unchanged (CUDA ``segment_gemm_v``)

``PunicaWrapperGPU`` calls :func:`get_sgmv_ops` to obtain the ``(shrink,
expand)`` callables appropriate for the current platform, so the gfx95 AMD
SGMV path activates automatically on MI355X hardware.
"""

from __future__ import annotations

from typing import Callable

import torch

from vllm.lora.kernels._amd_sgmv import (
    is_gfx95,
    is_rocm,
    sgmv_expand as _amd_sgmv_expand,
    sgmv_shrink as _amd_sgmv_shrink,
)

# Re-export for callers that want the AMD kernel directly.
amd_sgmv_shrink = _amd_sgmv_shrink
amd_sgmv_expand = _amd_sgmv_expand


def _triton_fallback_ops():
    """Lazy import of the Triton lora_shrink/lora_expand fallback ops."""
    from vllm.lora.ops.triton_ops import lora_expand, lora_shrink
    return lora_shrink, lora_expand


def get_sgmv_ops() -> tuple[Callable, Callable]:
    """Return ``(shrink, expand)`` callables for the active platform.

    On gfx95 (MI355X) the AMD SGMV kernel path from
    ``vllm.lora.kernels._amd_sgmv`` is used.  Otherwise the Triton fallback
    is returned so behaviour is unchanged on non-gfx95 hardware.
    """
    if is_gfx95():
        return _amd_sgmv_shrink, _amd_sgmv_expand
    return _triton_fallback_ops()


def sgmv_shrink_dispatch(*args, **kwargs):
    """Platform-dispatched SGMV shrink (hidden → rank)."""
    shrink, _ = get_sgmv_ops()
    return shrink(*args, **kwargs)


def sgmv_expand_dispatch(*args, **kwargs):
    """Platform-dispatched SGMV expand (rank → hidden)."""
    _, expand = get_sgmv_ops()
    return expand(*args, **kwargs)


__all__ = [
    "is_gfx95",
    "is_rocm",
    "amd_sgmv_shrink",
    "amd_sgmv_expand",
    "get_sgmv_ops",
    "sgmv_shrink_dispatch",
    "sgmv_expand_dispatch",
]
