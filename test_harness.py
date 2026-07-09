#!/usr/bin/env python3
"""Multi-LoRA SGMV benchmark for AMD MI355X (gfx950).

Runs:
  1. Correctness tests across r ∈ {8,16,64} × adapter counts ∈ {1,4,8}
  2. Throughput benchmark: 100 requests, 8 adapters, r=16, BF16
  3. Emits AMDPILOT_METRIC v1 block with requests_per_second_multi_lora
"""
import sys
import os
import time
import json

# --- Fix stray path from uv that injects python3.14 site-packages ---
sys.path = [p for p in sys.path if "python3.14" not in p]
sys.path.insert(0, "/workspace/vllm")

os.environ.setdefault("HIP_VISIBLE_DEVICES", "0,1,2,3,4,5,6,7")

import torch

from vllm.lora.kernels._amd_sgmv import (
    is_gfx95,
    sgmv_shrink,
    sgmv_expand,
)
from vllm.lora.ops.triton_ops.lora_kernel_metadata import LoRAKernelMeta

DEVICE = "cuda"
HIDDEN_SIZE = 8192
DTYPE = torch.bfloat16


# ---------------------------------------------------------------------------
# Correctness
# ---------------------------------------------------------------------------

def correctness_test(r: int, num_adapters: int, num_tokens: int = 32) -> bool:
    """Compare SGMV shrink+expand against eager per-token reference."""
    torch.manual_seed(42 + r * 100 + num_adapters)
    scaling = 1.0

    inputs = torch.randn(num_tokens, HIDDEN_SIZE, dtype=DTYPE, device=DEVICE)
    lora_a = torch.randn(num_adapters, r, HIDDEN_SIZE, dtype=DTYPE, device=DEVICE) * 0.02
    lora_b = torch.randn(num_adapters, HIDDEN_SIZE, r, dtype=DTYPE, device=DEVICE) * 0.02

    token_lora_mapping = torch.tensor(
        [i % num_adapters for i in range(num_tokens)], dtype=torch.int32, device=DEVICE
    )

    meta = LoRAKernelMeta.make(
        max_loras=num_adapters, max_num_tokens=num_tokens, device=DEVICE
    )
    meta.prepare_tensors(token_lora_mapping)
    meta_args = meta.meta_args(token_nums=num_tokens, specialize_active_lora=False)

    # --- SGMV path ---
    shrink_out = torch.empty(1, num_tokens, r, dtype=torch.float32, device=DEVICE)
    sgmv_shrink(inputs, [lora_a], shrink_out, *meta_args, scaling)

    expand_out = torch.zeros(num_tokens, HIDDEN_SIZE, dtype=DTYPE, device=DEVICE)
    sgmv_expand(
        shrink_out, [lora_b], expand_out, *meta_args,
        offset_start=0, add_inputs=True,
    )

    # --- Eager reference ---
    eager_shrink = torch.zeros(num_tokens, r, dtype=torch.float32, device=DEVICE)
    for t in range(num_tokens):
        lid = token_lora_mapping[t].item()
        eager_shrink[t] = (inputs[t].float() @ lora_a[lid].float().T) * scaling

    eager_expand = torch.zeros(num_tokens, HIDDEN_SIZE, dtype=DTYPE, device=DEVICE)
    for t in range(num_tokens):
        lid = token_lora_mapping[t].item()
        eager_expand[t] = (eager_shrink[t].to(DTYPE) @ lora_b[lid].T).to(DTYPE)

    shrink_match = torch.allclose(shrink_out[0], eager_shrink, atol=0.5, rtol=0.1)
    expand_match = torch.allclose(expand_out, eager_expand, atol=1.0, rtol=0.1)
    ok = shrink_match and expand_match
    if not ok:
        print(f"    FAIL r={r} adapters={num_adapters}: "
              f"shrink={shrink_match} expand={expand_match}")
    return ok


def run_correctness_suite():
    print("--- Correctness tests ---")
    all_pass = True
    for r in (8, 16, 64):
        for na in (1, 4, 8):
            ok = correctness_test(r, na)
            status = "PASS" if ok else "FAIL"
            print(f"  [{status}] r={r}, adapters={na}")
            all_pass = all_pass and ok
    return all_pass


# ---------------------------------------------------------------------------
# Throughput benchmark
# ---------------------------------------------------------------------------

def run_throughput_benchmark(
    num_adapters: int = 8,
    rank: int = 16,
    num_requests: int = 100,
    tokens_per_request: int = 32,
    warmup: int = 10,
) -> float:
    """Measure requests_per_second for multi-LoRA SGMV serving."""
    print(f"\n--- Throughput benchmark ---")
    print(f"  adapters={num_adapters}, rank={rank}, "
          f"requests={num_requests}, tokens/req={tokens_per_request}")

    torch.manual_seed(123)
    scaling = 1.0

    inputs = torch.randn(
        tokens_per_request, HIDDEN_SIZE, dtype=DTYPE, device=DEVICE
    )
    lora_a = torch.randn(num_adapters, rank, HIDDEN_SIZE, dtype=DTYPE, device=DEVICE) * 0.02
    lora_b = torch.randn(num_adapters, HIDDEN_SIZE, rank, dtype=DTYPE, device=DEVICE) * 0.02

    token_lora_mapping = torch.tensor(
        [i % num_adapters for i in range(tokens_per_request)],
        dtype=torch.int32, device=DEVICE,
    )

    meta = LoRAKernelMeta.make(
        max_loras=num_adapters, max_num_tokens=tokens_per_request, device=DEVICE
    )
    meta.prepare_tensors(token_lora_mapping)
    meta_args = meta.meta_args(
        token_nums=tokens_per_request, specialize_active_lora=False
    )

    shrink_buf = torch.empty(1, tokens_per_request, rank, dtype=torch.float32, device=DEVICE)
    expand_out = torch.zeros(tokens_per_request, HIDDEN_SIZE, dtype=DTYPE, device=DEVICE)

    def do_one_request():
        sgmv_shrink(inputs, [lora_a], shrink_buf, *meta_args, scaling)
        expand_out.zero_()
        sgmv_expand(
            shrink_buf, [lora_b], expand_out, *meta_args,
            offset_start=0, add_inputs=True,
        )

    # Warmup
    for _ in range(warmup):
        do_one_request()
    torch.cuda.synchronize()

    # Timed run
    start = time.perf_counter()
    for _ in range(num_requests):
        do_one_request()
    torch.cuda.synchronize()
    elapsed = time.perf_counter() - start

    rps = num_requests / elapsed
    print(f"  elapsed: {elapsed:.3f}s")
    print(f"  requests_per_second: {rps:.4f}")
    return rps


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    print("=" * 60)
    print("Multi-LoRA SGMV (AMD MI355X) benchmark")
    print("=" * 60)
    print(f"  is_gfx95: {is_gfx95()}")
    print(f"  device: {torch.cuda.get_device_properties(0).gcnArchName}")

    correctness_ok = run_correctness_suite()
    print(f"\n  Correctness suite: {'ALL PASS' if correctness_ok else 'FAILURES'}")

    rps = run_throughput_benchmark()

    # Emit metric block
    print()
    print("===== AMDPILOT_METRIC v1 =====")
    print(f"metric_name: requests_per_second_multi_lora")
    print(f"metric_value: {rps:.6f}")
    print(f"metric_direction: higher")
    print("===== END AMDPILOT_METRIC =====")

    # Write results
    results = {
        "correctness_passed": correctness_ok,
        "requests_per_second_multi_lora": rps,
        "is_gfx95": is_gfx95(),
    }
    with open("/workspace/check_results.json", "w") as f:
        json.dump(results, f, indent=2)

    sys.exit(0 if correctness_ok else 1)
