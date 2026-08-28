#!/usr/bin/env python3
"""
Benchmark: paged KV-cache decode attention ("PagedAttention decode").

WHAT THIS MEASURES
------------------
vLLM's signature operation is *PagedAttention*: scaled dot-product attention
over a *paged* KV cache, where each sequence's K/V tensors live in
non-contiguous fixed-size blocks and are located at attention time through a
per-sequence *block table*.  This is the decode-phase kernel that dominates
vLLM autoregressive serving throughput, and on AMD Instinct it runs through the
`vllm/v1/attention/backends/rocm_attn.py` backend -> the fused
`paged_attention_rocm` HIP op (see `vllm/_custom_ops.py:120`).

This script measures THAT operation on this GPU.  vLLM's fused HIP kernel cannot
be imported without a build, so the operation is reimplemented faithfully in
pure PyTorch:

  * the KV cache uses vLLM's exact "flash" block layout from
    `PagedAttention.split_kv_cache` (`vllm/v1/attention/ops/paged_attn.py`):
        x = 16 // element_size            # 8 for bf16/fp16
        key_cache   : [num_blocks, num_kv_heads, head_size//x, block_size, x]
        value_cache : [num_blocks, num_kv_heads, head_size, block_size]
  * K/V for each sequence are gathered through a block_table (the data movement
    that defines "paged" attention);
  * attention itself uses torch SDPA (Flash backend on ROCm), with GQA handled
    natively (no head-repeat materialisation), matching the fused kernel's
    compute path as closely as PyTorch allows.

WHAT IS AND IS NOT vLLM'S KERNEL
-------------------------------
vLLM's real kernel fuses the paged gather and the attention into one pass and
never materialises a contiguous [batch, heads, seq, head] KV buffer.  This script
times the gather and the SDPA *separately* and *combined*, so the report can
name its own gap: the `gather` time is an honest measure of the paged
data-movement cost, but the combined number is an *upper bound* on a fused kernel
that avoids the intermediate materialisation.  No claim is made that this equals
vLLM's optimised HIP kernel throughput; it is a faithful measurement of the
operation's data movement and attention compute on this hardware.

Usage:
    python paged_attn_decode_bench.py
"""
from __future__ import annotations

import argparse
import json
import warnings
import statistics
import sys
from dataclasses import dataclass, asdict

import torch
import torch.nn.functional as F


# --------------------------------------------------------------------------- #
# Environment capture (read from the machine, not assumed).                    #
# --------------------------------------------------------------------------- #
def capture_env() -> dict:
    warnings.filterwarnings("ignore", category=UserWarning)
    env = {
        "torch_version": torch.__version__,
        "hip_version": getattr(torch.version, "hip", None),
        "cuda_available": torch.cuda.is_available(),
        "device_count": torch.cuda.device_count(),
        "device_name": torch.cuda.get_device_name(0)
        if torch.cuda.is_available()
        else None,
        "device_capability": list(torch.cuda.get_device_capability(0))
        if torch.cuda.is_available()
        else None,
        "arch_list": torch.cuda.get_arch_list(),
    }
    # GPU product name / gfx arch via subprocess-less approach where possible.
    try:
        import subprocess

        out = subprocess.run(
            ["rocm-smi", "--showproductname"],
            capture_output=True, text=True, timeout=20,
        )
        env["rocm_smi_product"] = out.stdout
        out = subprocess.run(
            ["hipcc", "--version"], capture_output=True, text=True, timeout=20
        )
        env["hipcc_version"] = out.stdout
    except Exception as e:  # noqa: BLE001
        env["env_capture_error"] = f"{type(e).__name__}: {e}"
    # Probe which SDPA backends can run the *decode GQA* path actually used
    # by this benchmark: query [B, Hq, 1, hs], k/v [B, Hkv, S, hs] (Hq != Hkv),
    # enable_gqa=True. `can_use_flash_attention` gives a false negative here
    # (it ignores enable_gqa), so we test the real shapes directly and pick the
    # default-selected backend by timing flash vs. the default.
    from torch.nn.attention import SDPBackend, sdpa_kernel
    B_, Hq_, Hkv_, S_, hs_ = 4, 32, 8, 2048, 128
    q_ = torch.randn(B_, Hq_, 1, hs_, dtype=torch.bfloat16, device="cuda")
    k_ = torch.randn(B_, Hkv_, S_, hs_, dtype=torch.bfloat16, device="cuda")
    v_ = torch.randn(B_, Hkv_, S_, hs_, dtype=torch.bfloat16, device="cuda")
    sc_ = 1.0 / (hs_ ** 0.5)
    backend_usable = {}
    for name, bk in [
        ("FLASH", SDPBackend.FLASH_ATTENTION),
        ("EFFICIENT", SDPBackend.EFFICIENT_ATTENTION),
        ("MATH", SDPBackend.MATH),
        ("CUDNN", SDPBackend.CUDNN_ATTENTION),
    ]:
        try:
            with sdpa_kernel([bk]):
                F.scaled_dot_product_attention(
                    q_, k_, v_, scale=sc_, enable_gqa=True, is_causal=False)
            torch.cuda.synchronize()
            backend_usable[name] = True
        except Exception:
            backend_usable[name] = False
    env["sdpa_decode_backends_usable"] = backend_usable
    # identify the default backend by comparing default timing to flash timing
    import statistics as _st
    def _med(fn, w=10, n=40):
        for _ in range(w):
            fn(); torch.cuda.synchronize()
        s = [torch.cuda.Event(enable_timing=True) for _ in range(n)]
        e = [torch.cuda.Event(enable_timing=True) for _ in range(n)]
        torch.cuda.synchronize()
        for i in range(n):
            s[i].record(); fn(); e[i].record()
        torch.cuda.synchronize()
        return _st.median([x.elapsed_time(y) for x, y in zip(s, e)])
    def _flash():
        with sdpa_kernel([SDPBackend.FLASH_ATTENTION]):
            F.scaled_dot_product_attention(q_, k_, v_, scale=sc_,
                                            enable_gqa=True, is_causal=False)
    def _default():
        F.scaled_dot_product_attention(q_, k_, v_, scale=sc_,
                                       enable_gqa=True, is_causal=False)
    try:
        dmed = _med(_default); fmed = _med(_flash)
        env["sdpa_default_decode_backend"] = (
            "FLASH" if abs(dmed - fmed) / max(dmed, 1e-9) < 0.15 else "NON-FLASH"
        )
        env["sdpa_default_decode_ms"] = dmed
        env["sdpa_flash_decode_ms"] = fmed
    except Exception as e:  # noqa: BLE001
        env["sdpa_default_decode_backend"] = f"probe_failed: {e}"
    return env


# --------------------------------------------------------------------------- #
# Paged KV cache in vLLM's exact flash block layout.                           #
# --------------------------------------------------------------------------- #
@dataclass
class PagedKVCache:
    key_cache: torch.Tensor   # [num_blocks, Hkv, head_size//x, block_size, x]
    value_cache: torch.Tensor # [num_blocks, Hkv, head_size, block_size]
    block_table: torch.Tensor # [batch, max_blocks_per_seq]  int32


def build_paged_cache(
    num_blocks: int,
    num_kv_heads: int,
    head_size: int,
    block_size: int,
    batch: int,
    max_blocks_per_seq: int,
    dtype: torch.dtype,
    device: torch.device,
) -> PagedKVCache:
    elem_size = torch.tensor([], dtype=dtype).element_size()
    x = 16 // elem_size  # mirrors vLLM: x = 16 // kv_cache.element_size()
    assert head_size % x == 0, f"head_size {head_size} must be divisible by x {x}"
    key_cache = torch.randn(
        num_blocks, num_kv_heads, head_size // x, block_size, x,
        dtype=dtype, device=device,
    ) * 0.1
    value_cache = torch.randn(
        num_blocks, num_kv_heads, head_size, block_size,
        dtype=dtype, device=device,
    ) * 0.1
    # block_table: contiguous run of unique block ids per sequence, with a
    # realistic non-trivial mapping (sequences use disjoint blocks, shuffled
    # to defeat any accidental locality assumption).
    block_table = torch.zeros(batch, max_blocks_per_seq, dtype=torch.int32,
                              device=device)
    perm = torch.randperm(num_blocks, device=device)[: batch * max_blocks_per_seq]
    perm = perm.view(batch, max_blocks_per_seq)
    # each sequence's blocks are in a shuffled order (paged cache reality)
    for b in range(batch):
        block_table[b] = perm[b][torch.randperm(max_blocks_per_seq, device=device)]
    return PagedKVCache(key_cache, value_cache, block_table)


# --------------------------------------------------------------------------- #
# The decode attention operation (paged gather + SDPA), faithfully split.      #
# --------------------------------------------------------------------------- #
def paged_gather(
    cache: PagedKVCache, batch: int, num_kv_heads: int,
    head_size: int, block_size: int, num_blocks_per_seq: int,
    seq_len: int, dtype: torch.dtype, device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Gather K/V for every sequence through its block table.

    Returns contiguous [batch, Hkv, seq_len, head_size] K and V, exactly as a
    non-fused decode path would need before running attention.
    """
    x = 16 // torch.tensor([], dtype=dtype).element_size()
    bt = cache.block_table[:, :num_blocks_per_seq]  # [batch, nblocks]
    # K: [num_blocks, Hkv, hs//x, B, x] -> index -> [batch, nblocks, Hkv, hs//x, B, x]
    gk = cache.key_cache[bt]
    # -> [batch, Hkv, nblocks, B, hs//x, x] -> [batch, Hkv, nblocks*B, hs]
    gk = gk.permute(0, 2, 1, 4, 3, 5).reshape(
        batch, num_kv_heads, num_blocks_per_seq * block_size, head_size
    )
    # V: [num_blocks, Hkv, hs, B] -> [batch, nblocks, Hkv, hs, B]
    gv = cache.value_cache[bt]
    # -> [batch, Hkv, nblocks, B, hs] -> [batch, Hkv, nblocks*B, hs]
    gv = gv.permute(0, 2, 1, 4, 3).reshape(
        batch, num_kv_heads, num_blocks_per_seq * block_size, head_size
    )
    # trim padding in the last block to the true sequence length
    gk = gk[:, :, :seq_len, :].contiguous()
    gv = gv[:, :, :seq_len, :].contiguous()
    return gk, gv


def attention_decode(
    query: torch.Tensor, k: torch.Tensor, v: torch.Tensor, scale: float,
) -> torch.Tensor:
    """SDPA decode: query [B, Hq, 1, hs], k/v [B, Hkv, S, hs] with GQA."""
    return F.scaled_dot_product_attention(
        query, k, v, scale=scale, enable_gqa=True, is_causal=False
    )


# --------------------------------------------------------------------------- #
# Timing harness.                                                              #
# --------------------------------------------------------------------------- #
def time_op(fn, warmup: int, iters: int) -> list[float]:
    """Time a GPU op with CUDA events. `fn` must launch GPU work only."""
    for _ in range(warmup):
        fn()
        torch.cuda.synchronize()
    starts = [torch.cuda.Event(enable_timing=True) for _ in range(iters)]
    ends = [torch.cuda.Event(enable_timing=True) for _ in range(iters)]
    torch.cuda.synchronize()
    for i in range(iters):
        starts[i].record()
        fn()
        ends[i].record()
    torch.cuda.synchronize()
    return [s.elapsed_time(e) for s, e in zip(starts, ends)]


def stats(samples_ms: list[float]) -> dict:
    s = sorted(samples_ms)
    n = len(s)
    return {
        "n": n,
        "mean_ms": statistics.mean(s),
        "median_ms": statistics.median(s),
        "min_ms": s[0],
        "max_ms": s[-1],
        "std_ms": statistics.pstdev(s) if n > 1 else 0.0,
        "p5_ms": s[int(0.05 * (n - 1))],
        "p95_ms": s[int(0.95 * (n - 1))],
        "cv_pct": (statistics.pstdev(s) / statistics.mean(s) * 100.0)
        if n > 1 and statistics.mean(s) > 0
        else 0.0,
    }


# --------------------------------------------------------------------------- #
# Main.                                                                        #
# --------------------------------------------------------------------------- #
def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--head-size", type=int, default=128)
    ap.add_argument("--num-heads", type=int, default=32, help="query heads")
    ap.add_argument("--num-kv-heads", type=int, default=8, help="KV heads (GQA)")
    ap.add_argument("--block-size", type=int, default=16,
                    help="vLLM default block size")
    ap.add_argument("--seq-len", type=int, nargs="+", default=[4096],
                    help="KV length(s) attended over during decode")
    ap.add_argument("--batch", type=int, nargs="+", default=[1, 4, 16, 64])
    ap.add_argument("--dtype", choices=["bfloat16", "float16"], default="bfloat16")
    ap.add_argument("--warmup", type=int, default=20)
    ap.add_argument("--iters", type=int, default=100)
    ap.add_argument("--json", type=str, default=None, help="write results json")
    args = ap.parse_args()

    device = torch.device("cuda")
    dtype = getattr(torch, args.dtype)
    scale = 1.0 / (args.head_size ** 0.5)

    print("=" * 78)
    print("Paged KV-cache decode attention (PagedAttention decode) benchmark")
    print("=" * 78)
    env = capture_env()
    print(f"GPU           : {env['device_name']}")
    print(f"torch         : {env['torch_version']}")
    print(f"HIP           : {env['hip_version']}")
    print(f"gfx/capability: {env['device_capability']}  arch_list={env['arch_list']}")
    print(f"SDPA decode   : default={env.get('sdpa_default_decode_backend')} "
          f"usable={env.get('sdpa_decode_backends_usable')}")
    print("-" * 78)
    print(f"config: head_size={args.head_size} num_heads={args.num_heads} "
          f"num_kv_heads={args.num_kv_heads} (GQA "
          f"{args.num_heads // args.num_kv_heads}q/kv) block_size={args.block_size} "
          f"seq_len={args.seq_len} dtype={args.dtype} scale={scale:.6f}")
    print("-" * 78)

    results = {"env": env, "config": vars(args), "runs": []}

    for seq_len in args.seq_len:
        num_blocks_per_seq = (seq_len + args.block_size - 1) // args.block_size
        for batch in args.batch:
            max_blocks_per_seq = num_blocks_per_seq
            num_blocks = batch * max_blocks_per_seq + 64  # slack
            cache = build_paged_cache(
                num_blocks, args.num_kv_heads, args.head_size, args.block_size,
                batch, max_blocks_per_seq, dtype, device,
            )
            query = torch.randn(batch, args.num_heads, 1, args.head_size,
                                dtype=dtype, device=device) * 0.1

            def gather_only():
                return paged_gather(
                    cache, batch, args.num_kv_heads, args.head_size,
                    args.block_size, num_blocks_per_seq, seq_len, dtype, device,
                )

            k_g, v_g = gather_only()  # warmup materialised

            def attn_only():
                return attention_decode(query, k_g, v_g, scale)

            def full_decode():
                k, v = paged_gather(
                    cache, batch, args.num_kv_heads, args.head_size,
                    args.block_size, num_blocks_per_seq, seq_len, dtype, device,
                )
                return attention_decode(query, k, v, scale)

            g_ms = time_op(gather_only, args.warmup, args.iters)
            a_ms = time_op(attn_only, args.warmup, args.iters)
            f_ms = time_op(full_decode, args.warmup, args.iters)

            g_st = stats(g_ms)
            a_st = stats(a_ms)
            f_st = stats(f_ms)

            # derived throughput: decode produces `batch` tokens per full decode
            total_ms = f_st["median_ms"]
            tok_per_s = batch / (total_ms / 1e3) if total_ms > 0 else float("nan")
            # KV bytes moved by the gather: batch*Hkv*seq_len*head_size*2(K)*elem
            elem = torch.tensor([], dtype=dtype).element_size()
            kv_bytes = batch * args.num_kv_heads * seq_len * args.head_size * 2 * elem
            gather_bw_gbs = kv_bytes / (g_st["median_ms"] / 1e3) / 1e9 \
                if g_st["median_ms"] > 0 else float("nan")

            run = {
                "batch": batch,
                "seq_len": seq_len,
                "num_blocks_per_seq": num_blocks_per_seq,
                "gather": g_st,
                "attention_sdpa": a_st,
                "full_decode": f_st,
                "tokens_per_s_median": tok_per_s,
                "kv_bytes_gathered": kv_bytes,
                "gather_bandwidth_gbs_median": gather_bw_gbs,
            }
            results["runs"].append(run)

            print(f"\n[batch={batch}] seq_len={seq_len} "
                  f"({batch} decode token{'s' if batch>1 else ''}/iter)")
            print(f"  paged gather : median {g_st['median_ms']:.4f} ms  "
                  f"(mean {g_st['mean_ms']:.4f}, min {g_st['min_ms']:.4f}, "
                  f"max {g_st['max_ms']:.4f}, std {g_st['std_ms']:.4f}, "
                  f"p5 {g_st['p5_ms']:.4f}, p95 {g_st['p95_ms']:.4f}, cv {g_st['cv_pct']:.2f}%)")
            print(f"  attention    : median {a_st['median_ms']:.4f} ms  "
                  f"(mean {a_st['mean_ms']:.4f}, min {a_st['min_ms']:.4f}, "
                  f"max {a_st['max_ms']:.4f}, std {a_st['std_ms']:.4f}, "
                  f"p5 {a_st['p5_ms']:.4f}, p95 {a_st['p95_ms']:.4f}, cv {a_st['cv_pct']:.2f}%)")
            print(f"  full decode  : median {f_st['median_ms']:.4f} ms  "
                  f"(mean {f_st['mean_ms']:.4f}, min {f_st['min_ms']:.4f}, "
                  f"max {f_st['max_ms']:.4f}, std {f_st['std_ms']:.4f}, "
                  f"p5 {f_st['p5_ms']:.4f}, p95 {f_st['p95_ms']:.4f}, cv {f_st['cv_pct']:.2f}%)")
            print(f"  throughput   : {tok_per_s:,.0f} decode tokens/s | "
                  f"gather BW {gather_bw_gbs:,.1f} GB/s | "
                  f"KV moved {kv_bytes/1e9:.2f} GB")

    print("\n" + "=" * 78)
    print("NOTE: full_decode = paged_gather + SDPA as separate PyTorch ops,")
    print("which materialises a contiguous KV buffer. vLLM's fused HIP kernel")
    print("(paged_attention_rocm) avoids that materialisation, so full_decode")
    print("here is an UPPER BOUND on the real kernel, not its throughput.")
    print("=" * 78)

    if args.json:
        with open(args.json, "w") as fp:
            json.dump(results, fp, indent=2)
        print(f"\nwrote {args.json}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
