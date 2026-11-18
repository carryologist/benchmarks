# RDMA Fabric Recovery on a 2x DGX Spark Cluster

**Date:** 2026-11-18
**Run:** [2026-11-18--inference--local-dgx-spark--v1](../runs/2026-11-18--inference--local-dgx-spark--v1/)
**Harness:** [harnesses/local-inference](../harnesses/local-inference/)

## What changed

Three changes were applied to the DeepSeek-V4-Flash Vision-Exp DSpark recipe
(MiaAI 2x DGX Spark):

1. **Context window 1M → 500K** (`MAX_MODEL_LEN=500000`) — unlocks the
   UE8M0 DeepGEMM path and fixes prefix-cache residency.
2. **RDMA fabric enabled** — `NCCL_NET=IB`, `NCCL_IB_DISABLE=0`, HCA
   `rocep1s0f1`, on kernel `6.17.0-1032-nvidia` (the `7.0.0-1019-nvidia`
   NCCL/RoCE regression causes `ibv_reg_mr_iova2 ... Cannot allocate memory`).
3. **Tool-loop fix** — backend encoder hotfix
   (`DSPARK_ENABLE_ASSISTANT_FINAL_HOTFIX=1`) plus router-side loop guard.

## Results

| Case | Socket (500K) | RDMA (500K) | Delta |
|---|---|---|---|
| c=1, ~256-token prompt, decode | 34.3 tok/s | **61.5 tok/s** | +79% |
| c=1, ~256-token prompt, TTFT | ~480–565 ms | **~300 ms** | −40% |
| c=1, ~32K-token prompt, decode | 38.4 tok/s | **44.7 tok/s** | +16% |
| c=1, ~32K-token prompt, TTFT | ~32.4 s | **~18.3 s** | −44% |
| c=6, ~4K-token prompt, per-stream | 9.7 tok/s | **17.0 tok/s** | +75% |
| c=6, ~4K-token prompt, aggregate | 27.0 tok/s | **45.6 tok/s** | +69% |
| c=6, ~4K-token prompt, TTFT | ~14.5 s | **~8.7 s** | −40% |

Against the 1M-context socket baseline (c=1/256 = 32.8 tok/s, c=6/4K
aggregate = 25.5 tok/s), RDMA + 500K context gives **+88%** single-stream
decode and **+79%** multi-stream aggregate.

The 40–50 tok/s target the user set is exceeded on the single-stream lane.

## Fabric check

- `ib_write_bw` over `rocep1s0f1` (192.168.100.10 ↔ .11): **12995 MB/s (~104 Gbps)**
  vs TCP `enp1s0f1np1` ~16–19 Gbps (~5–6x).
- Boot log confirms `NCCL INFO Using network IB` over `[0]rocep1s0f1:1/RoCE`.
- GPUDirect RDMA is **not** supported on GB10 (`NCCL_DMABUF_ENABLE=0`,
  `DMABUF_SUPPORTED=0`); NCCL falls back to host-buffer registration, which
  still delivers the bandwidth win.

## Tool-loop verification on this boot

- `tool-loop-probe.py`: 15 cases, **0 empty turns**, no DSML leakage,
  assistant-final hotfix patched and verified on both ranks.
- Router-side guard (LAR): 5 identical calls → HTTP 409 without a backend call;
  3 identical calls → 200 with a user-turn nudge and a clean answer.

## Notes

- Kernel rollback required the GRUB submenu form
  `gnulinux-advanced-<uuid> >gnulinux-6.17.0-1032-nvidia-advanced-<uuid>`;
  the plain menuentry ID did not take effect.
- `6.17.0-1014-nvidia` has no matching NVIDIA driver modules in this image
  (`linux-modules-nvidia-580-open-6.17.0-1014-nvidia` is in `rc` state and its
  dependency conflicts with installed `nvidia-kernel-common-580 580.173.02`),
  so `6.17.0-1032-nvidia` was used instead.
- Benchmarks were run serially; overlapping runs inflate variance.
