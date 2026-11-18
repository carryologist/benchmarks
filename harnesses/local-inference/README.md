# Local Inference Harness

Runner scripts for benchmarking local inference endpoints (DGX Spark cluster,
RTX 5090 workstation) that expose an OpenAI-compatible `/v1` API.

## `bench-miaai.py`

Faithful replica of MiaAI's 2026-08-14 matrix methodology for the
DeepSeek-V4-Flash Vision-Exp DSpark recipe:

- unique cold prefix per request (no prefix-cache reuse)
- `thinking=false`
- `min_tokens = max_tokens = 128`, `ignore_eos`
- numbered-word instruction so output length is fixed
- reports median per-stream decode tok/s after the first token, plus TTFT

```bash
python3 bench-miaai.py \
  --base-url http://192.168.0.4:8888/v1 \
  --model deepseek-v4-flash-vision-exp \
  --prompt 256 --concurrency 1 --repeat 8
```

Run cases **serially**, not in parallel: concurrent matrix points contaminate
each other's decode rates (observed 6.9–52.1 tok/s spread when a c=1 run
overlapped a 32K-prefill run).

## `tool-loop-probe.py`

Measures DeepSeek-V4 tool-calling failure modes against a live endpoint. It
replays a repeated-transcript agent loop and classifies each reply as:

- `tool_call` — structured call, correct
- `final_answer` — loop broken, model stopped
- DSML markup leaked into content/reasoning
- empty / length-truncated no-op turn

```bash
python3 tool-loop-probe.py \
  --base-url http://192.168.0.4:8888/v1 \
  --model deepseek-v4-flash-vision-exp
```

## No auth

Both scripts target an unauthenticated local endpoint. No credentials are used.
