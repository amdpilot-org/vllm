# Report: Paged KV-cache decode attention (PagedAttention decode) on MI355X

## TL;DR

Measured **vLLM's signature operation — paged KV-cache decode attention
("PagedAttention decode")** — on an AMD Instinct MI355X (gfx950) with
PyTorch 2.9.1+rocm7.2.0 / ROCm 7.2.0. vLLM's fused HIP kernel
(`paged_attention_rocm`) cannot be imported without a build, so the operation
is reimplemented **faithfully in pure PyTorch** using vLLM's exact paged KV-cache
"flash" block layout, and timed with 200 CUDA-event samples after a 30-iter
warmup. All results are stable (coefficient of variation **≤ 2.0%**, mostly
**< 1%**). At `seq_len=4096`, decode throughput saturates around
**~24.5k decode tokens/s** (batch ≥ 16); the paged-gather data movement
saturates around **~830 GB/s**. The reported `full_decode` time is an **upper
bound** on vLLM's fused kernel (see Gaps).

## The operation, and why it represents this project

**PagedAttention decode** is the kernel vLLM is named for. During autoregressive
decode each request emits one query token that attends to its full key/value
history, and that KV history lives in **non-contiguous fixed-size blocks**
located at attention time through a per-sequence **block table**. This fused
paged-gather-plus-attention op is the single largest determinant of vLLM's
decode throughput, and on AMD Instinct it is the hot path of the
`vllm/v1/attention/backends/rocm_attn.py` backend, which dispatches to the
fused HIP op `paged_attention_rocm`
([`vllm/_custom_ops.py`](../../vllm/_custom_ops.py), `paged_attention_rocm`)
and the paged KV-cache layout defined by
`PagedAttention.split_kv_cache`
([`vllm/v1/attention/ops/paged_attn.py`](../../vllm/v1/attention/ops/paged_attn.py)).

It is representative because:

- it is the namesake data-movement pattern (block-table indirection) that
  distinguishes vLLM from a naive contiguous attention;
- it is the decode-phase critical path that dominates serving latency/throughput;
- its config space (GQA heads, head_size, block_size, KV length, batch) is the
  exact space vLLM tunes its attention backends over.

## Method — faithful PyTorch reimplementation (no build)

vLLM's `_custom_ops` require a compiled `torch.ops._rocm_C` extension
(build-from-source), which this task explicitly forbids. So the operation is
reimplemented faithfully in `paged_attn_decode_bench.py`:

- **Paged KV cache in vLLM's exact "flash" block layout**, copied verbatim from
  the semantics of `PagedAttention.split_kv_cache`:
  `x = 16 // element_size` (→ 8 for bf16),
  `key_cache   : [num_blocks, num_kv_heads, head_size//x, block_size, x]`,
  `value_cache : [num_blocks, num_kv_heads, head_size, block_size]`.
- **Block-table indirection**: each sequence's K/V are gathered through a
  shuffled per-sequence `block_table` (sequences use disjoint, out-of-order
  blocks, defeating accidental locality assumptions) — this is the data
  movement that defines "paged" attention.
- **Grouped-query attention (GQA)**: `num_heads=32` query heads,
  `num_kv_heads=8` (4 q-per-kv), with GQA handled natively via SDPA
  `enable_gqa=True` (no head-repeat materialisation), matching the fused
  kernel's compute path as closely as PyTorch allows.
- **Attention math** uses `torch.nn.functional.scaled_dot_product_attention`.
  On this build the **Flash backend is selected by default** for the decode GQA
  path (confirmed by timing: default ≈ forced-FLASH ≈ 0.081 ms vs forced-MATH
  ≈ 1.69 ms on a 4×32×8×2048×128 probe); EFFICIENT and CUDNN backends are
  unavailable for this shape.

The script times three things per config, each with 30 warmup + 200 timed
CUDA-event iterations:

| component   | what it measures                                              |
|-------------|---------------------------------------------------------------|
| `gather`    | paged block-table gather → contiguous `[B, Hkv, S, hs]` K/V   |
| `attention` | SDPA decode (Flash) over pre-gathered contiguous K/V          |
| `full_decode` | `gather` + `attention` back-to-back (one decode step)       |

Config (Llama-3-class decode, TP=1): `head_size=128`, `num_heads=32`,
`num_kv_heads=8`, `block_size=16` (vLLM `DEFAULT_BLOCK_SIZE`), `dtype=bfloat16`,
`scale = 1/√128`.

## Environment (read from the machine)

| item | value |
|------|-------|
| GPU | AMD Instinct MI355X (Card Model 0x75a3, **gfx950**, cap `(9,5)`) |
| VRAM | 288 GB (per `amd-smi`) |
| PyTorch | `2.9.1+rocm7.2.0.git7e1940d4` |
| HIP | `7.2.26015-fc0010cf6a` |
| hipcc | AMD clang 22.0.0git (roc-7.2.0) |
| arch_list | `gfx908,gfx90a,gfx942,gfx1030,gfx1100,gfx1101,gfx1200,gfx1201,gfx950,gfx1151,gfx1150` |
| SDPA decode path | default backend = **FLASH** (EFFICIENT/CUDNN unusable for this GQA shape; MATH usable but ~20× slower) |
| Python | CPython 3.12 |

## Results

All times in **ms**. Spread reported as `median | mean | min | max | std | p5 | p95 | cv%`.
200 timed iterations after 30 warmup. `tok/s` = decode tokens / `full_decode` median.
`BW` = KV bytes gathered / `gather` median (GB/s).

### Batch sweep — `seq_len = 4096`

| batch | gather (med) | attention (med) | full_decode (med) | full cv% | tok/s | gather BW |
|------:|-------------:|----------------:|------------------:|---------:|------:|----------:|
| 1  | 0.0427 | 0.1518 | 0.1909 | 0.66 | 5,239 | 393 GB/s |
| 4  | 0.0979 | 0.1521 | 0.2488 | 0.55 | 16,077 | 685 GB/s |
| 16 | 0.3231 | 0.3053 | 0.6587 | 0.42 | 24,291 | 831 GB/s |
| 64 | 1.2920 | 1.3242 | 2.6135 | 0.31 | 24,488 | 831 GB/s |

Full spread (`median|mean|min|max|std|p5|p95|cv%`) for `full_decode`:

- batch 1 : `0.1909 | 0.1911 | 0.1894 | 0.1976 | 0.0013 | 0.1898 | 0.1938 | 0.66`
- batch 4 : `0.2488 | 0.2489 | 0.2463 | 0.2546 | 0.0014 | 0.2470 | 0.2518 | 0.55`
- batch 16: `0.6587 | 0.6588 | 0.6523 | 0.6668 | 0.0028 | 0.6548 | 0.6632 | 0.42`
- batch 64: `2.6135 | 2.6140 | 2.5928 | 2.6328 | 0.0080 | 2.6006 | 2.6273 | 0.31`

### Sequence-length sweep — `batch = 16`

| seq_len | gather (med) | attention (med) | full_decode (med) | full cv% | tok/s | gather BW |
|--------:|-------------:|----------------:|------------------:|---------:|------:|----------:|
| 1024 | 0.0978 | 0.0491 | 0.1439 | 0.72 | 111,187 | 686 GB/s |
| 2048 | 0.1766 | 0.1595 | 0.3356 | 0.65 | 47,675 | 760 GB/s |
| 4096 | 0.3212 | 0.3056 | 0.6549 | 0.55 | 24,432 | 836 GB/s |
| 8192 | 0.6446 | 0.6595 | 1.3035 | 0.41 | 12,274 | 833 GB/s |

`full_decode` time scales near-linearly with KV length (1024→8192 ≈ 9× the KV,
9.1× the time), confirming the measurement is in a bandwidth/compute-bound
regime rather than launch-overhead-dominated (at batch 16).

### Interpretation

- **Throughput saturates at ~24.5k decode tokens/s** (batch ≥ 16, seq 4096): at
  small batch the op is launch/latency bound (5.2k tok/s at batch 1); the GPU
  only fills up around batch 16.
- **The paged gather is ~half the decode cost** at batch 16+ (gather ≈ attention),
  and its bandwidth plateaus at **~830 GB/s** — the signature cost of the
  block-table data movement that defines PagedAttention.
- **Attention (SDPA-Flash) and gather track each other** as batch/KV grows, so
  neither is negligible; a fused kernel that removes the gather's intermediate
  materialisation is where real vLLM gains over this baseline would come from.

## Gaps and limitations (be explicit)

- **This is NOT vLLM's fused `paged_attention_rocm` kernel.** vLLM fuses the
  paged gather and the attention into one pass and never materialises a
  contiguous `[B, H, S, hs]` KV buffer. This script materialises that buffer,
  so `full_decode` here is an **upper bound** on the real kernel, not its
  throughput. The `gather` number is, however, an honest measure of the paged
  data-movement cost on this hardware.
- **No build, so no apples-to-apples comparison** to `_rocm_C.paged_attention`.
  The numbers characterise the *operation* (its data movement + attention
  compute) on MI355X, not vLLM's specific fused-kernel performance.
- **SDPA backend is PyTorch's Flash path**, not vLLM's tuned FlashAttention/CK
  MLA variants; decode-time attention here is therefore a proxy for the compute
  portion, optimised but not vLLM-specific.
- **Single GPU, TP=1, bf16 only.** No FP8/MXFP8 KV cache, no tensor parallelism,
  no cascade/prefix sharing, no sliding window — all are real vLLM paths not
  exercised here.
- **Block table is shuffled but synthetic.** Real serving has shared-prefix /
  cache-reuse locality that this synthetic workload does not model; the gather
  BW is thus a *lower* bound on achievable locality benefit, not an upper.
- **head_size=128 only.** Some models use head_size=64/96/256; not swept.

If the chosen operation had not been measurable in budget, the simpler
substitution would have been `reshape_and_cache` (the KV-cache write, a pure
data-movement op trivially faithful to reimplement). It was not needed: the
decode attention measured cleanly and stably within budget.

## Reproduce

Exact commands run for the numbers above (from the repo root, on the MI355X
box with the environment above):

```sh
# Batch sweep (seq_len=4096) -> results_batch_sweep.json + batch_sweep.txt
python reports/j-496dd0e3a756/paged_attn_decode_bench.py \
    --batch 1 4 16 64 --seq-len 4096 --warmup 30 --iters 200 \
    --json reports/j-496dd0e3a756/results_batch_sweep.json

# Sequence-length sweep (batch=16) -> results_seqlen_sweep.json + seqlen_sweep.txt
python reports/j-496dd0e3a756/paged_attn_decode_bench.py \
    --batch 16 --seq-len 1024 2048 4096 8192 --warmup 30 --iters 200 \
    --json reports/j-496dd0e3a756/results_seqlen_sweep.json
```

The script is self-contained (only `torch` + stdlib). Raw machine-readable
results are in `results_batch_sweep.json` / `results_seqlen_sweep.json`; the
captured console output is in `batch_sweep.txt` / `seqlen_sweep.txt`.

Options of note: `--dtype float16`, `--num-heads/--num-kv-heads/--head-size`
(model geometry), `--block-size` (vLLM default 16), `--warmup/--iters` (timing).
