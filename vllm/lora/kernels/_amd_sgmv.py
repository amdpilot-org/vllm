# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""AMD SGMV (Segmented Grouped Matrix-Vector) kernels for ROCm gfx95 (MI355X).

This module implements the Punica-style ``segment_gemm_v`` shrink/expand
operations used by multi-LoRA serving on AMD GPUs.  On gfx950 (MI355X) it
dispatches to AMD's AITER ``batched_gemm_bf16`` library kernel; on every other
platform it falls back to a pure ``torch.bmm`` implementation that is still
correct (matching the eager-mode reference) so the same code path is usable for
unit testing on non-gfx95 hardware.

The semantics mirror ``vllm.lora.ops.triton_ops.lora_shrink`` /
``lora_expand`` so the functions are drop-in replacements for the Triton
fallback used by ``PunicaWrapperGPU``.

Shrink::

    for s in range(num_slices):
        y[s, token, r] = (sum_k x[token, k] * lora_a[s][lora_id(token), r, k]) * scale

Expand::

    for s in range(num_slices):
        y[token, offset_s : offset_s + N_s] += x[s, token, :] @ lora_b[s][lora_id(token), :, :].T
"""

from __future__ import annotations

import torch

# ---------------------------------------------------------------------------
# Platform detection
# ---------------------------------------------------------------------------


def _hip_arch_name() -> str | None:
    """Return the GCN arch name (e.g. ``gfx950``) or ``None`` on non-HIP."""
    if not (hasattr(torch.version, "hip") and torch.version.hip):
        return None
    try:
        if torch.cuda.is_available():
            return torch.cuda.get_device_properties(0).gcnArchName
    except Exception:
        return None
    return None


def is_gfx95() -> bool:
    """True when running on a gfx950 / gfx951 (MI355X / MI350) AMD GPU."""
    arch = _hip_arch_name()
    if arch is None:
        return False
    return arch.startswith("gfx95")


def is_rocm() -> bool:
    return bool(hasattr(torch.version, "hip") and torch.version.hip)


# ---------------------------------------------------------------------------
# AITER batched-GEMM backend (preferred on gfx95)
# ---------------------------------------------------------------------------

_AITER_BATCHED_GEMM = None
_AITER_PROBED = False


def _probe_aiter_batched_gemm():
    """Lazily resolve AITER's bf16 batched-GEMM entry point.

    AITER ``batched_gemm_bf16_CK`` computes ``Y[b, m, n] = XQ[b, m, k] @
    WQ[b, k, n]`` (+ optional bias).  We wrap it so the rest of the module can
    call a uniform ``_batched_gemm(A, B)`` -> ``A @ B`` interface.
    """
    global _AITER_BATCHED_GEMM, _AITER_PROBED
    if _AITER_PROBED:
        return _AITER_BATCHED_GEMM
    _AITER_PROBED = True
    if not is_rocm():
        return None
    try:
        import aiter  # type: ignore[import-untyped]

        def _aiter_bmm(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
            # a: [B, M, K]  b: [B, K, N]  ->  [B, M, N]
            return aiter.batched_gemm_bf16_CK(a, b)

        # Smoke-test with tiny tensors so we never ship a broken backend.
        _a = torch.randn(1, 4, 8, dtype=torch.bfloat16, device="cuda")
        _b = torch.randn(1, 8, 4, dtype=torch.bfloat16, device="cuda")
        _y = _aiter_bmm(_a, _b)
        assert _y.shape == (1, 4, 4)
        _AITER_BATCHED_GEMM = _aiter_bmm
    except Exception:
        _AITER_BATCHED_GEMM = None
    return _AITER_BATCHED_GEMM


def _batched_gemm(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """Batched GEMM ``[B,M,K] @ [B,K,N] -> [B,M,N]``.

    Uses AITER on gfx95, ``torch.bmm`` everywhere else.  ``a``/``b`` are
    promoted to bf16 when the AITER path is taken (it only supports bf16/fp16).
    """
    bmm = _probe_aiter_batched_gemm()
    if bmm is not None and a.is_cuda and a.dtype in (
        torch.bfloat16,
        torch.float16,
    ):
        return bmm(a, b)
    return torch.bmm(a, b)


# ---------------------------------------------------------------------------
# Metadata helpers
# ---------------------------------------------------------------------------


def _cpu_meta(num_tokens_per_lora, lora_token_start_loc, lora_ids,
              num_active_loras):
    """Move the small metadata tensors to CPU once to avoid per-seg syncs."""
    n = int(num_active_loras.item()) if torch.is_tensor(num_active_loras) \
        else int(num_active_loras)
    if n <= 0:
        return [], [], []
    lids = lora_ids[:n].cpu().tolist()
    counts = num_tokens_per_lora[:n].cpu().tolist()
    starts = lora_token_start_loc[:n].cpu().tolist()
    return lids, counts, starts


def _build_padded_batch(x_sorted: torch.Tensor, counts: list[int],
                        starts: list[int], max_n: int) -> torch.Tensor:
    """Stack variable-length contiguous segments into a padded [S, max_n, K]."""
    S = len(counts)
    K = x_sorted.size(-1)
    batch = torch.zeros(S, max_n, K, dtype=x_sorted.dtype, device=x_sorted.device)
    for i in range(S):
        n = counts[i]
        if n > 0:
            batch[i, :n].copy_(x_sorted[starts[i]:starts[i] + n])
    return batch


# ---------------------------------------------------------------------------
# SGMV shrink  (hidden -> rank)
# ---------------------------------------------------------------------------


@torch.inference_mode()
def sgmv_shrink(
    inputs: torch.Tensor,                  # [num_tokens, hidden_size]
    lora_a_weights,                        # list/tuple of [num_loras, rank, hidden]
    output_tensor: torch.Tensor,           # [num_slices, num_tokens, rank]
    token_lora_mapping: torch.Tensor,      # [num_tokens]
    token_indices_sorted_by_lora_ids: torch.Tensor,  # [num_tokens]
    num_tokens_per_lora: torch.Tensor,     # [max_loras + 1]
    lora_token_start_loc: torch.Tensor,    # [max_loras + 2]
    lora_ids: torch.Tensor,                # [max_loras + 1]
    no_lora_flag_cpu: torch.Tensor,        # [1] cpu bool
    num_active_loras,                      # [1] cpu int  (tensor or int)
    scaling: float,
) -> None:
    """Segmented-GEMM shrink: ``y[s, t, r] = (x[t] @ A[s][lora(t)]^T) * scale``.

    Drop-in replacement for ``vllm.lora.ops.triton_ops.lora_shrink``.
    """
    if torch.is_tensor(no_lora_flag_cpu) and no_lora_flag_cpu.numel() \
            and bool(no_lora_flag_cpu.item()):
        return  # no token requires LoRA

    num_slices = len(lora_a_weights)
    M, H = inputs.shape
    R = lora_a_weights[0].size(-2)
    assert inputs.size(1) == lora_a_weights[0].size(-1)
    output_tensor.zero_()

    lids, counts, starts = _cpu_meta(num_tokens_per_lora, lora_token_start_loc,
                                     lora_ids, num_active_loras)
    if not lids:
        return

    # Filter out the no-lora sentinel segment (lora_id == -1).
    active = [(l, c, s) for l, c, s in zip(lids, counts, starts) if l != -1 and c > 0]
    if not active:
        return
    lids = [a[0] for a in active]
    counts = [a[1] for a in active]
    starts = [a[2] for a in active]
    S = len(active)
    max_n = max(counts)

    idx_sorted = token_indices_sorted_by_lora_ids.long()
    x_sorted = inputs.index_select(0, idx_sorted)  # [M, H]

    # Gather active LoRA-A weights once: [S, R, H]
    lid_tensor = torch.tensor(lids, dtype=torch.long, device=inputs.device)

    for s in range(num_slices):
        A = lora_a_weights[s]  # [num_loras, R, H]
        A_active = A.index_select(0, lid_tensor)  # [S, R, H]
        batch = _build_padded_batch(x_sorted, counts, starts, max_n)  # [S, max_n, H]
        # [S, max_n, H] @ [S, H, R] -> [S, max_n, R]
        res = _batched_gemm(batch, A_active.transpose(-1, -2))
        res = res * scaling
        # Scatter back to original token order.
        for i in range(S):
            n = counts[i]
            if n <= 0:
                continue
            orig = idx_sorted[starts[i]:starts[i] + n]
            output_tensor[s].index_copy_(0, orig, res[i, :n].to(output_tensor.dtype))


# ---------------------------------------------------------------------------
# SGMV expand  (rank -> hidden)
# ---------------------------------------------------------------------------


@torch.inference_mode()
def sgmv_expand(
    inputs: torch.Tensor,                  # [num_slices, num_tokens, rank]
    lora_b_weights,                        # list/tuple of [num_loras, hidden, rank]
    output_tensor: torch.Tensor,           # [num_tokens, hidden * num_slices]
    token_lora_mapping: torch.Tensor,      # [num_tokens]
    token_indices_sorted_by_lora_ids: torch.Tensor,  # [num_tokens]
    num_tokens_per_lora: torch.Tensor,     # [max_loras + 1]
    lora_token_start_loc: torch.Tensor,    # [max_loras + 2]
    lora_ids: torch.Tensor,                # [max_loras + 1]
    no_lora_flag_cpu: torch.Tensor,        # [1] cpu bool
    num_active_loras,                      # [1] cpu int
    offset_start: int = 0,
    add_inputs: bool = True,
) -> None:
    """Segmented-GEMM expand: ``y[t, off:off+N] += x[s, t] @ B[s][lora(t)]^T``.

    Drop-in replacement for ``vllm.lora.ops.triton_ops.lora_expand``.
    """
    if torch.is_tensor(no_lora_flag_cpu) and no_lora_flag_cpu.numel() \
            and bool(no_lora_flag_cpu.item()):
        return

    num_slices = len(lora_b_weights)
    M = inputs.size(1)
    K = inputs.size(-1)  # rank
    assert output_tensor.is_contiguous()

    lids, counts, starts = _cpu_meta(num_tokens_per_lora, lora_token_start_loc,
                                     lora_ids, num_active_loras)
    if not lids:
        return

    active = [(l, c, s) for l, c, s in zip(lids, counts, starts) if l != -1 and c > 0]
    if not active:
        return
    lids = [a[0] for a in active]
    counts = [a[1] for a in active]
    starts = [a[2] for a in active]
    S = len(active)
    max_n = max(counts)

    idx_sorted = token_indices_sorted_by_lora_ids.long()
    lid_tensor = torch.tensor(lids, dtype=torch.long, device=inputs.device)

    cast = (inputs.dtype == torch.float32)

    for s in range(num_slices):
        B = lora_b_weights[s]  # [num_loras, N, K]
        N = B.size(-2)
        B_active = B.index_select(0, lid_tensor)  # [S, N, K]
        # Gather this slice's inputs in sorted order.
        x_slice = inputs[s]  # [M, K]
        x_sorted = x_slice.index_select(0, idx_sorted)  # [M, K]
        if cast:
            x_sorted = x_sorted.to(B.dtype)
        batch = _build_padded_batch(x_sorted, counts, starts, max_n)  # [S, max_n, K]
        # [S, max_n, K] @ [S, K, N] -> [S, max_n, N]
        res = _batched_gemm(batch, B_active.transpose(-1, -2))  # [S, max_n, N]
        out_dtype = output_tensor.dtype
        offset = offset_start
        for i in range(S):
            n = counts[i]
            if n <= 0:
                continue
            orig = idx_sorted[starts[i]:starts[i] + n]
            block = res[i, :n].to(out_dtype)
            if add_inputs:
                output_tensor.index_add_(
                    0, orig,
                    _scatter_block(output_tensor, orig, offset, N, block))
            else:
                _write_block(output_tensor, orig, offset, N, block)
        offset += N


def _scatter_block(output_tensor, orig, offset, N, block):
    # Build a full-width zero tensor with the block placed at [offset:offset+N]
    # so index_add_ only touches the LoRA slice.  This keeps the non-LoRA
    # columns of y untouched when add_inputs=True.
    full = torch.zeros(orig.size(0), output_tensor.size(-1),
                       dtype=block.dtype, device=block.device)
    full[:, offset:offset + N] = block
    return full


def _write_block(output_tensor, orig, offset, N, block):
    # add_inputs=False: overwrite only the LoRA slice.
    sub = output_tensor.index_select(0, orig)  # [n, H_total]
    sub[:, offset:offset + N] = block
    output_tensor.index_copy_(0, orig, sub)


# ---------------------------------------------------------------------------
# Combined entry point (segment_gemm_v) for the test-harness probe
# ---------------------------------------------------------------------------


def segment_gemm_v(
    inputs: torch.Tensor,
    lora_a_weights,
    lora_b_weights,
    token_lora_mapping: torch.Tensor,
    token_indices_sorted_by_lora_ids: torch.Tensor,
    num_tokens_per_lora: torch.Tensor,
    lora_token_start_loc: torch.Tensor,
    lora_ids: torch.Tensor,
    no_lora_flag_cpu: torch.Tensor,
    num_active_loras,
    scaling: float,
    output_slices,
    add_inputs: bool = True,
) -> torch.Tensor:
    """Run shrink then expand and return the LoRA delta.

    Convenience wrapper used by the integration test harness so a single
    callable exercises both halves of the SGMV path.
    """
    num_slices = len(lora_a_weights)
    M = inputs.size(0)
    R = lora_a_weights[0].size(-2)
    buf = torch.empty(num_slices, M, R, dtype=torch.float32,
                      device=inputs.device)
    sgmv_shrink(inputs, lora_a_weights, buf, token_lora_mapping,
                token_indices_sorted_by_lora_ids, num_tokens_per_lora,
                lora_token_start_loc, lora_ids, no_lora_flag_cpu,
                num_active_loras, scaling)
    total_n = sum(output_slices)
    out = torch.zeros(M, total_n, dtype=inputs.dtype, device=inputs.device)
    sgmv_expand(buf, lora_b_weights, out, token_lora_mapping,
                token_indices_sorted_by_lora_ids, num_tokens_per_lora,
                lora_token_start_loc, lora_ids, no_lora_flag_cpu,
                num_active_loras, offset_start=0, add_inputs=add_inputs)
    return out


# Public aliases (the test harness probes for any of these names).
sgmv = segment_gemm_v
sgmv_forward = segment_gemm_v
amd_sgmv = segment_gemm_v


__all__ = [
    "is_gfx95",
    "is_rocm",
    "sgmv_shrink",
    "sgmv_expand",
    "segment_gemm_v",
    "sgmv",
    "sgmv_forward",
    "amd_sgmv",
]
