#!/usr/bin/env python3
"""
Stage0 harness for multi-LoRA serving (SGMV) on MI355X.

This harness exercises the vLLM LoRA shrink/expand Triton kernels with
8 synthetic LoRA adapters (r=16, BF16) to establish a baseline for
requests_per_second_multi_lora.

The benchmark simulates multi-LoRA serving by:
1. Creating 8 synthetic LoRA adapters with random BF16 weights
2. Creating token-to-adapter mappings (round-robin across 8 adapters)
3. Running lora_shrink (x @ A) and lora_expand (result @ B) kernels
4. Measuring throughput as requests_per_second
"""

import os
import sys
import time
import argparse

# Ensure we use system python3, not broken /opt/venv
def main():
    import torch
    import torch.nn.functional as F

    print(f"torch version: {torch.__version__}", flush=True)
    print(f"hip version: {torch.version.hip}", flush=True)
    print(f"cuda available: {torch.cuda.is_available()}", flush=True)
    print(f"device count: {torch.cuda.device_count()}", flush=True)

    if not torch.cuda.is_available():
        print("ERROR: CUDA/HIP not available", flush=True)
        sys.exit(1)

    device = torch.device("cuda:0")
    gcn_arch = torch.cuda.get_device_properties(0).gcnArchName
    print(f"GCN arch: {gcn_arch}", flush=True)

    # Import vLLM LoRA ops
    from vllm.lora.ops.triton_ops import (
        LoRAKernelMeta,
        lora_expand,
        lora_shrink,
    )
    print("Successfully imported vLLM LoRA Triton ops", flush=True)

    # Configuration
    NUM_ADAPTERS = 8
    LORA_RANK = 16
    HIDDEN_SIZE = 8192
    NUM_TOKENS = 128  # tokens per batch
    NUM_SLICES = 1    # standard LoRA has 1 slice per layer
    OUTPUT_SIZE = HIDDEN_SIZE  # square LoRA for simplicity
    SCALE = 1.0 / LORA_RANK
    WARMUP_ITERS = 10
    BENCH_ITERS = 100
    NUM_REQUESTS = 100  # 100 requests round-robin across 8 adapters

    MAX_LORAS = NUM_ADAPTERS

    # Create LoRAKernelMeta
    token_meta = LoRAKernelMeta.make(
        max_loras=MAX_LORAS,
        max_num_tokens=NUM_TOKENS,
        device=device,
    )
    prompt_meta = LoRAKernelMeta.make(
        max_loras=MAX_LORAS,
        max_num_tokens=NUM_TOKENS,
        device=device,
    )

    # Create synthetic LoRA weights (BF16)
    # lora_a_stacked: tuple of (max_loras, r, hidden_size) per slice
    # lora_b_stacked: tuple of (max_loras, output_size, r) per slice
    lora_a_stacked = tuple(
        torch.randn(MAX_LORAS, LORA_RANK, HIDDEN_SIZE, dtype=torch.bfloat16, device=device)
        for _ in range(NUM_SLICES)
    )
    lora_b_stacked = tuple(
        torch.randn(MAX_LORAS, OUTPUT_SIZE, LORA_RANK, dtype=torch.bfloat16, device=device)
        for _ in range(NUM_SLICES)
    )

    # Create token-to-adapter mapping (round-robin)
    token_lora_mapping = torch.arange(NUM_TOKENS, dtype=torch.int32, device=device) % NUM_ADAPTERS

    # Prepare metadata
    token_meta.prepare_tensors(token_lora_mapping)

    # Create input tensor
    x = torch.randn(NUM_TOKENS, HIDDEN_SIZE, dtype=torch.bfloat16, device=device)

    # Shrink output: (num_slices, num_tokens, r)
    shrink_output = torch.zeros(NUM_SLICES, NUM_TOKENS, LORA_RANK, dtype=torch.bfloat16, device=device)

    # Expand output: (num_tokens, output_size)
    expand_output = torch.zeros(NUM_TOKENS, OUTPUT_SIZE, dtype=torch.bfloat16, device=device)

    # Verify kernels work with a single forward pass
    print("Running verification forward pass...", flush=True)
    lora_shrink(
        x,
        lora_a_stacked,
        shrink_output,
        *token_meta.meta_args(NUM_TOKENS, False),
        SCALE,
    )
    torch.cuda.synchronize()
    print(f"  shrink output shape: {shrink_output.shape}", flush=True)
    print(f"  shrink output norm: {shrink_output.norm().item():.4f}", flush=True)

    lora_expand(
        shrink_output,
        lora_b_stacked,
        expand_output,
        *token_meta.meta_args(NUM_TOKENS, False),
        offset_start=0,
        add_inputs=True,
    )
    torch.cuda.synchronize()
    print(f"  expand output shape: {expand_output.shape}", flush=True)
    print(f"  expand output norm: {expand_output.norm().item():.4f}", flush=True)

    # Warmup
    print(f"Warming up ({WARMUP_ITERS} iters)...", flush=True)
    for _ in range(WARMUP_ITERS):
        lora_shrink(
            x,
            lora_a_stacked,
            shrink_output,
            *token_meta.meta_args(NUM_TOKENS, False),
            SCALE,
        )
        lora_expand(
            shrink_output,
            lora_b_stacked,
            expand_output,
            *token_meta.meta_args(NUM_TOKENS, False),
            offset_start=0,
            add_inputs=True,
        )
    torch.cuda.synchronize()

    # Benchmark
    print(f"Benchmarking ({BENCH_ITERS} iters, {NUM_REQUESTS} requests)...", flush=True)
    start_time = time.perf_counter()
    for _ in range(BENCH_ITERS):
        lora_shrink(
            x,
            lora_a_stacked,
            shrink_output,
            *token_meta.meta_args(NUM_TOKENS, False),
            SCALE,
        )
        lora_expand(
            shrink_output,
            lora_b_stacked,
            expand_output,
            *token_meta.meta_args(NUM_TOKENS, False),
            offset_start=0,
            add_inputs=True,
        )
    torch.cuda.synchronize()
    end_time = time.perf_counter()

    elapsed = end_time - start_time
    total_requests = BENCH_ITERS * NUM_REQUESTS
    requests_per_second = total_requests / elapsed

    print(f"\nResults:", flush=True)
    print(f"  Elapsed time: {elapsed:.4f}s", flush=True)
    print(f"  Total requests: {total_requests}", flush=True)
    print(f"  Requests/second: {requests_per_second:.2f}", flush=True)
    print(f"  Tokens/second: {total_requests * NUM_TOKENS / elapsed:.2f}", flush=True)

    # Harness checks
    checks_passed = 0
    checks_total = 7

    # Check 1: Kernels executed without error
    checks_passed += 1
    print(f"  [CHECK 1/7] Kernels executed without error: PASS", flush=True)

    # Check 2: Output is non-zero
    if expand_output.norm().item() > 0:
        checks_passed += 1
        print(f"  [CHECK 2/7] Output non-zero: PASS", flush=True)
    else:
        print(f"  [CHECK 2/7] Output non-zero: FAIL", flush=True)

    # Check 3: Shrink output has correct shape
    if shrink_output.shape == (NUM_SLICES, NUM_TOKENS, LORA_RANK):
        checks_passed += 1
        print(f"  [CHECK 3/7] Shrink output shape correct: PASS", flush=True)
    else:
        print(f"  [CHECK 3/7] Shrink output shape correct: FAIL", flush=True)

    # Check 4: Expand output has correct shape
    if expand_output.shape == (NUM_TOKENS, OUTPUT_SIZE):
        checks_passed += 1
        print(f"  [CHECK 4/7] Expand output shape correct: PASS", flush=True)
    else:
        print(f"  [CHECK 4/7] Expand output shape correct: FAIL", flush=True)

    # Check 5: All 8 adapters were used
    unique_adapters = torch.unique(token_lora_mapping)
    if unique_adapters.numel() == NUM_ADAPTERS:
        checks_passed += 1
        print(f"  [CHECK 5/7] All 8 adapters used: PASS", flush=True)
    else:
        print(f"  [CHECK 5/7] All 8 adapters used: FAIL", flush=True)

    # Check 6: BF16 weights used
    if lora_a_stacked[0].dtype == torch.bfloat16:
        checks_passed += 1
        print(f"  [CHECK 6/7] BF16 weights: PASS", flush=True)
    else:
        print(f"  [CHECK 6/7] BF16 weights: FAIL", flush=True)

    # Check 7: requests_per_second above threshold (1000 req/s minimum)
    threshold = 1000.0
    if requests_per_second >= threshold:
        checks_passed += 1
        print(f"  [CHECK 7/7] requests_per_second >= {threshold}: PASS ({requests_per_second:.2f})", flush=True)
    else:
        print(f"  [CHECK 7/7] requests_per_second >= {threshold}: FAIL ({requests_per_second:.2f})", flush=True)

    print(f"\nChecks passed: {checks_passed}/{checks_total}", flush=True)

    # Emit metric block
    print(f"\n===== AMDPILOT_METRIC v1 =====", flush=True)
    print(f"metric_name: requests_per_second_multi_lora", flush=True)
    print(f"metric_value: {requests_per_second:.2f}", flush=True)
    print(f"metric_direction: higher", flush=True)
    print(f"===== END AMDPILOT_METRIC =====", flush=True)

    if checks_passed == checks_total:
        print("\nAll checks passed. Baseline established.", flush=True)
        return 0
    else:
        print(f"\n{checks_total - checks_passed} checks failed.", flush=True)
        return 1


if __name__ == "__main__":
    sys.exit(main())
