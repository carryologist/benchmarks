#!/usr/bin/env python3
"""tool-loop-probe.py — measure the DeepSeek V4 tool-calling failure modes live.

Sends a repeated-transcript agent loop (same tool result replayed) and classifies
each reply:
  - tool_calls present  (structured call, correct)
  - bare DSML markup leaked into content/reasoning (parser recovery case)
  - final answer with no tool call (loop broken)
  - empty / length-truncated (no-op loop)

Usage:
  python3 tool-loop-probe.py --base-url http://127.0.0.1:8888/v1 --model deepseek-v4-flash-vision-exp
  python3 tool-loop-probe.py --base-url http://192.168.0.32:8088/v1 --model deepseek-vision
"""
import argparse
import json
import re
import sys
import time
import urllib.request

TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "Read",
            "description": "Read a file from the filesystem.",
            "parameters": {
                "type": "object",
                "properties": {"path": {"type": "string"}},
                "required": ["path"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "Glob",
            "description": "List files matching a glob pattern.",
            "parameters": {
                "type": "object",
                "properties": {"pattern": {"type": "string"}},
                "required": ["pattern"],
            },
        },
    },
]

DSML_RE = re.compile(r"DSML|</invoke>|</parameter>|tool_calls>|<｜DSML｜tool_calls>|toolcalls")


def post(base_url, model, body, timeout=180):
    req = urllib.request.Request(
        base_url.rstrip("/") + "/chat/completions",
        data=json.dumps(body).encode(),
        headers={"Content-Type": "application/json"},
    )
    t0 = time.perf_counter()
    try:
        with urllib.request.urlopen(req, timeout=timeout) as r:
            return json.load(r), time.perf_counter() - t0, r.status
    except urllib.error.HTTPError as e:
        return {"error": e.read().decode()[:400]}, time.perf_counter() - t0, e.code


def build_loop(tool, n_calls, arg):
    msgs = [
        {"role": "system", "content": "You are a coding agent. Use tools only when needed. If the answer is already in the tool results, reply immediately without calling more tools."},
        {"role": "user", "content": "List the files in /tmp and report them."},
    ]
    for i in range(n_calls):
        msgs.append({"role": "assistant", "content": "", "tool_calls": [
            {"id": f"call_{i}", "type": "function", "function": {"name": tool, "arguments": json.dumps(arg)}},
        ]})
        msgs.append({"role": "tool", "tool_call_id": f"call_{i}", "content": "a.txt\nb.txt\nc.txt"})
    return msgs


def classify(d):
    if "error" in d:
        return "http_error", d["error"]
    ch = d["choices"][0]
    m = ch["message"]
    fr = ch.get("finish_reason")
    tcs = m.get("tool_calls") or []
    content = m.get("content") or ""
    reasoning = m.get("reasoning_content") or ""
    leaked = bool(DSML_RE.search(content) or DSML_RE.search(reasoning))
    if tcs:
        return "tool_call", f"finish={fr} n={len(tcs)} names={[t['function']['name'] for t in tcs]} leaked={leaked}"
    if leaked:
        return "dsml_leak", f"finish={fr} content={content[:200]!r}"
    if not content.strip():
        return "empty", f"finish={fr} reasoning_len={len(reasoning)}"
    return "final_answer", f"finish={fr} content={content[:120]!r}"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base-url", default="http://127.0.0.1:8888/v1")
    ap.add_argument("--model", default="deepseek-v4-flash-vision-exp")
    ap.add_argument("--max-tokens", type=int, default=1024)
    ap.add_argument("--repeat", type=int, default=3)
    args = ap.parse_args()

    cases = [
        ("Glob", 2, {"pattern": "/tmp/*"}),
        ("Glob", 3, {"pattern": "/tmp/*"}),
        ("Glob", 4, {"pattern": "/tmp/*"}),
        ("Read", 2, {"path": "/tmp/a.txt"}),
        ("Read", 3, {"path": "/tmp/a.txt"}),
    ]
    tally = {}
    for tool, n, arg in cases:
        for r in range(args.repeat):
            body = {
                "model": args.model,
                "messages": build_loop(tool, n, arg),
                "tools": TOOLS,
                "tool_choice": "auto",
                "max_tokens": args.max_tokens,
                "temperature": 0,
            }
            d, elapsed, status = post(args.base_url, args.model, body)
            kind, detail = classify(d)
            tally[kind] = tally.get(kind, 0) + 1
            print(f"{tool:5s} n={n} rep={r} status={status} {elapsed:6.1f}s -> {kind:13s} {detail}")
    print("\nTALLY:", json.dumps(tally))


if __name__ == "__main__":
    sys.exit(main())
