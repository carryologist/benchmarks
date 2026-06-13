#!/usr/bin/env python3
"""
Frontier Bakeoff — Fable 5 vs Opus 4.8 vs Sonnet 4.6 vs GPT-5.5
Adapted from Round 5 homelab benchmark for cloud models.
See CHANGES.md for all differences from Round 5.
"""

import argparse
import json
import os
import re
import subprocess
import sys
import tempfile
import time
import statistics
import textwrap
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

try:
    import anthropic
except ImportError:
    print("Installing anthropic SDK...")
    subprocess.check_call([sys.executable, "-m", "pip", "install", "anthropic", "-q"])
    import anthropic

try:
    import openai
except ImportError:
    print("Installing openai SDK...")
    subprocess.check_call([sys.executable, "-m", "pip", "install", "openai", "-q"])
    import openai

# ─── Constants ───────────────────────────────────────────────────────────────

SCRIPT_DIR = Path(__file__).parent.resolve()
RESULTS_DIR = SCRIPT_DIR / "results"
FIXTURES_DIR = SCRIPT_DIR / "fixtures"

MODELS = {
    "fable5": {
        "name": "Claude Fable 5",
        "provider": "anthropic",
        "model_id": "claude-fable-5",
        "temperature": 1.0,
        "thinking_type": "adaptive",
        "thinking_budget": 10000,
    },
    "opus48": {
        "name": "Claude Opus 4.8",
        "provider": "anthropic",
        "model_id": "claude-opus-4-8",
        "temperature": 1.0,
        "thinking_type": "adaptive",
        "thinking_budget": 10000,
    },
    "sonnet46": {
        "name": "Claude Sonnet 4.6",
        "provider": "anthropic",
        "model_id": "claude-sonnet-4-6",
        "temperature": 1.0,
        "thinking_type": "adaptive",
        "thinking_budget": 10000,
    },
    "gpt55": {
        "name": "GPT-5.5",
        "provider": "openai",
        "model_id": "gpt-5.5",
        "temperature": 1.0,  # GPT-5.5 only supports temperature=1
        "reasoning_effort": "medium",
    },
}

# ─── Clients ─────────────────────────────────────────────────────────────────

_ANTHROPIC_CLIENT = None
_OPENAI_CLIENT = None


def get_anthropic_client() -> anthropic.Anthropic:
    global _ANTHROPIC_CLIENT
    if _ANTHROPIC_CLIENT is None:
        api_key = os.environ.get("ANTHROPIC_API_KEY") or os.environ.get("ANTHROPIC_AUTH_TOKEN")
        base_url = os.environ.get("ANTHROPIC_BASE_URL")
        if not api_key:
            print("ERROR: Set ANTHROPIC_API_KEY or ANTHROPIC_AUTH_TOKEN.")
            sys.exit(1)
        kwargs = {"api_key": api_key}
        if base_url:
            kwargs["base_url"] = base_url
        _ANTHROPIC_CLIENT = anthropic.Anthropic(**kwargs)
    return _ANTHROPIC_CLIENT


def get_openai_client() -> openai.OpenAI:
    global _OPENAI_CLIENT
    if _OPENAI_CLIENT is None:
        api_key = os.environ.get("OPENAI_API_KEY")
        base_url = os.environ.get("OPENAI_BASE_URL")
        if not api_key:
            print("ERROR: Set OPENAI_API_KEY.")
            sys.exit(1)
        kwargs = {"api_key": api_key}
        if base_url:
            kwargs["base_url"] = base_url
        _OPENAI_CLIENT = openai.OpenAI(**kwargs)
    return _OPENAI_CLIENT


def log(msg: str):
    ts = datetime.now().strftime("%H:%M:%S")
    print(f"[{ts}] {msg}")


def extract_code(text: str, lang: str = "") -> str:
    """Extract best code block from markdown-fenced response, or return raw text.
    Uses longest match to skip planning snippets."""
    pattern = rf"```(?:{lang})?\s*\n(.*?)```"
    matches = list(re.finditer(pattern, text, re.DOTALL))
    if matches:
        best = max(matches, key=lambda m: len(m.group(1)))
        code = best.group(1).strip()
        code = textwrap.dedent(code)
        return code.strip()
    return text.strip()


def chat(messages: list[dict], model_key: str, speed_mode: bool = False,
         max_tokens: int = 4096) -> dict:
    """Route to the correct provider's chat function."""
    cfg = MODELS[model_key]
    if cfg["provider"] == "openai":
        return _chat_openai(messages, model_key, speed_mode, max_tokens)
    return _chat_anthropic(messages, model_key, speed_mode, max_tokens)


def _chat_anthropic(messages: list[dict], model_key: str, speed_mode: bool = False,
                    max_tokens: int = 4096) -> dict:
    """Send a message via the Anthropic Messages API."""
    cfg = MODELS[model_key]

    # Build thinking config based on model type
    thinking_budget = 1024 if speed_mode else cfg["thinking_budget"]
    thinking_type = cfg["thinking_type"]

    # Anthropic API requires max_tokens to be > thinking budget
    effective_max = max_tokens + thinking_budget

    thinking_config = {"type": thinking_type}
    if thinking_type == "enabled":
        thinking_config["budget_tokens"] = thinking_budget

    t0 = time.perf_counter()

    try:
        response = get_anthropic_client().messages.create(
            model=cfg["model_id"],
            max_tokens=effective_max,
            temperature=cfg["temperature"],
            thinking=thinking_config,
            messages=messages,
        )
    except anthropic.APIError as e:
        return {
            "content": "", "reasoning": "", "error": str(e),
            "prompt_tokens": 0, "completion_tokens": 0,
            "tokens_per_sec": 0, "ttft_ms": 0,
            "wall_time_s": round(time.perf_counter() - t0, 2),
            "finish_reason": "error",
        }

    wall_time = time.perf_counter() - t0

    content_text = ""
    thinking_text = ""
    for block in response.content:
        if block.type == "text":
            content_text += block.text
        elif block.type == "thinking":
            thinking_text += block.thinking

    finish_reason = response.stop_reason
    if finish_reason == "refusal":
        log("  WARNING: Request was refused by safety classifiers.")
        content_text = "[REFUSED BY SAFETY CLASSIFIER]"

    usage = response.usage
    output_tokens = usage.output_tokens
    tokens_per_sec = round(output_tokens / wall_time, 1) if wall_time > 0 else 0

    return {
        "content": content_text, "reasoning": thinking_text,
        "prompt_tokens": usage.input_tokens, "completion_tokens": output_tokens,
        "tokens_per_sec": tokens_per_sec, "ttft_ms": 0,
        "wall_time_s": round(wall_time, 2), "finish_reason": finish_reason,
    }


def _chat_openai(messages: list[dict], model_key: str, speed_mode: bool = False,
                 max_tokens: int = 4096) -> dict:
    """Send a message via the OpenAI Chat Completions API."""
    cfg = MODELS[model_key]

    t0 = time.perf_counter()

    try:
        response = get_openai_client().chat.completions.create(
            model=cfg["model_id"],
            messages=messages,
            max_completion_tokens=max_tokens,
            temperature=cfg["temperature"],
        )
    except openai.APIError as e:
        return {
            "content": "", "reasoning": "", "error": str(e),
            "prompt_tokens": 0, "completion_tokens": 0,
            "tokens_per_sec": 0, "ttft_ms": 0,
            "wall_time_s": round(time.perf_counter() - t0, 2),
            "finish_reason": "error",
        }

    wall_time = time.perf_counter() - t0

    choice = response.choices[0]
    content_text = choice.message.content or ""
    reasoning_text = ""
    # OpenAI may include reasoning in a separate field
    if hasattr(choice.message, "reasoning_content") and choice.message.reasoning_content:
        reasoning_text = choice.message.reasoning_content
    # Strip any leaked <think> blocks (some OpenAI models)
    content_text = re.sub(r"<think>.*?</think>\s*", "", content_text, flags=re.DOTALL).strip()

    usage = response.usage
    prompt_tokens = usage.prompt_tokens if usage else 0
    completion_tokens = usage.completion_tokens if usage else 0
    tokens_per_sec = round(completion_tokens / wall_time, 1) if wall_time > 0 else 0

    return {
        "content": content_text, "reasoning": reasoning_text,
        "prompt_tokens": prompt_tokens, "completion_tokens": completion_tokens,
        "tokens_per_sec": tokens_per_sec, "ttft_ms": 0,
        "wall_time_s": round(wall_time, 2),
        "finish_reason": choice.finish_reason or "unknown",
    }


def chat_streaming_ttft(messages: list[dict], model_key: str,
                        max_tokens: int = 4096) -> dict:
    """Streaming variant for speed tests. Measures true TTFT."""
    cfg = MODELS[model_key]
    if cfg["provider"] == "openai":
        return _stream_openai(messages, model_key, max_tokens)
    return _stream_anthropic(messages, model_key, max_tokens)


def _stream_anthropic(messages: list[dict], model_key: str,
                      max_tokens: int = 4096) -> dict:
    cfg = MODELS[model_key]
    thinking_budget = 1024
    thinking_type = cfg["thinking_type"]
    effective_max = max_tokens + thinking_budget

    thinking_config = {"type": thinking_type}
    if thinking_type == "enabled":
        thinking_config["budget_tokens"] = thinking_budget

    t0 = time.perf_counter()
    ttft = None
    content_text = ""

    try:
        with get_anthropic_client().messages.stream(
            model=cfg["model_id"],
            max_tokens=effective_max,
            temperature=cfg["temperature"],
            thinking=thinking_config,
            messages=messages,
        ) as stream:
            for event in stream:
                if ttft is None and hasattr(event, 'type'):
                    if event.type == 'content_block_start':
                        ttft = time.perf_counter() - t0
            response = stream.get_final_message()
    except anthropic.APIError as e:
        return {"content": "", "error": str(e), "completion_tokens": 0,
                "tokens_per_sec": 0, "ttft_ms": 0,
                "wall_time_s": round(time.perf_counter() - t0, 2)}

    wall_time = time.perf_counter() - t0
    for block in response.content:
        if block.type == "text":
            content_text += block.text

    output_tokens = response.usage.output_tokens
    tokens_per_sec = round(output_tokens / wall_time, 1) if wall_time > 0 else 0
    ttft_ms = round(ttft * 1000, 1) if ttft is not None else 0

    return {"content": content_text, "completion_tokens": output_tokens,
            "tokens_per_sec": tokens_per_sec, "ttft_ms": ttft_ms,
            "wall_time_s": round(wall_time, 2)}


def _stream_openai(messages: list[dict], model_key: str,
                   max_tokens: int = 4096) -> dict:
    cfg = MODELS[model_key]

    t0 = time.perf_counter()
    ttft = None
    chunks = []

    try:
        stream = get_openai_client().chat.completions.create(
            model=cfg["model_id"],
            messages=messages,
            max_completion_tokens=max_tokens,
            temperature=cfg["temperature"],
            stream=True,
            stream_options={"include_usage": True},
        )
        for chunk in stream:
            if ttft is None and chunk.choices and chunk.choices[0].delta.content:
                ttft = time.perf_counter() - t0
            chunks.append(chunk)
    except openai.APIError as e:
        return {"content": "", "error": str(e), "completion_tokens": 0,
                "tokens_per_sec": 0, "ttft_ms": 0,
                "wall_time_s": round(time.perf_counter() - t0, 2)}

    wall_time = time.perf_counter() - t0
    content_text = "".join(
        c.choices[0].delta.content for c in chunks
        if c.choices and c.choices[0].delta and c.choices[0].delta.content
    )

    # Get usage from the final chunk
    output_tokens = 0
    for c in reversed(chunks):
        if c.usage:
            output_tokens = c.usage.completion_tokens or 0
            break
    if output_tokens == 0:
        # Estimate from content length
        output_tokens = len(content_text) // 4

    tokens_per_sec = round(output_tokens / wall_time, 1) if wall_time > 0 else 0
    ttft_ms = round(ttft * 1000, 1) if ttft is not None else 0

    return {"content": content_text, "completion_tokens": output_tokens,
            "tokens_per_sec": tokens_per_sec, "ttft_ms": ttft_ms,
            "wall_time_s": round(wall_time, 2)}


# ─── Task Definitions ───────────────────────────────────────────────────────

def task_1_1_todo(model_key: str) -> dict:
    """CLI Todo App (Python)."""
    prompt = (
        "Build a Python CLI todo app with SQLite persistence. "
        "CRUD commands: add, list, complete, delete. Timestamps. Pretty output. "
        "Error handling. A __main__ block. Using argparse or sys.argv, NOT interactive input. "
        "Respond with ONLY the Python code, no explanation."
    )
    result = chat([{"role": "user", "content": prompt}], model_key, max_tokens=4096)
    code = extract_code(result["content"], "python")
    result["code"] = code
    result["lines"] = len(code.splitlines())

    try:
        compile(code, "<todo>", "exec")
        result["syntax_valid"] = True
    except SyntaxError as e:
        result["syntax_valid"] = False
        result["syntax_error"] = str(e)

    features = {
        "sqlite": bool(re.search(r"sqlite3|import sqlite", code, re.I)),
        "add": bool(re.search(r"def.*add|['\"]add['\"]|add.*todo", code, re.I)),
        "list": bool(re.search(r"def.*list|['\"]list['\"]|list.*todo", code, re.I)),
        "complete": bool(re.search(r"def.*complete|['\"]complete['\"]|mark.*done|complete.*todo", code, re.I)),
        "delete": bool(re.search(r"def.*delete|['\"]delete['\"]|remove.*todo|delete.*todo", code, re.I)),
        "timestamps": bool(re.search(r"datetime|timestamp|created|strftime", code, re.I)),
        "completion_tracking": bool(re.search(r"complete|done|status|finished", code, re.I)),
        "error_handling": bool(re.search(r"try:|except |raise ", code)),
        "main_block": "__main__" in code,
        "pretty_output": bool(re.search(r"tabulate|format|ljust|rjust|f['\"]|print.*\|", code, re.I)),
    }
    result["features"] = features
    result["feature_score"] = sum(features.values())
    result["feature_max"] = len(features)

    if result["syntax_valid"]:
        result["functional"] = run_todo_functional_test(code)
    else:
        result["functional"] = {"score": 0, "max": 7, "details": "Skipped: syntax invalid"}

    feat_pct = result["feature_score"] / result["feature_max"]
    func_pct = result["functional"]["score"] / result["functional"]["max"] if result["functional"]["max"] > 0 else 0
    composite = (feat_pct * 60) + (func_pct * 40)
    if not result["syntax_valid"]:
        composite *= (2 / 3)
    result["composite_score"] = round(composite, 1)

    return result


def run_todo_functional_test(code: str) -> dict:
    """Run the todo app through 7 functional operations."""
    details = []
    score = 0
    max_score = 7

    with tempfile.TemporaryDirectory() as tmpdir:
        app_path = os.path.join(tmpdir, "todo.py")
        with open(app_path, "w") as f:
            f.write(code)

        def run_cmd(args: list[str]) -> tuple[int, str, str]:
            try:
                r = subprocess.run(
                    [sys.executable, app_path] + args,
                    capture_output=True, text=True, timeout=10,
                    cwd=tmpdir
                )
                return r.returncode, r.stdout, r.stderr
            except subprocess.TimeoutExpired:
                return -1, "", "timeout"
            except Exception as e:
                return -1, "", str(e)

        # 1. Add item 1
        rc, out, err = run_cmd(["add", "Buy milk"])
        if rc == 0:
            score += 1
            details.append("PASS: add item 1")
        else:
            details.append(f"FAIL: add item 1 (rc={rc}, err={err[:100]})")

        # 2. Add item 2
        rc, out, err = run_cmd(["add", "Walk dog"])
        if rc == 0:
            score += 1
            details.append("PASS: add item 2")
        else:
            details.append(f"FAIL: add item 2 (rc={rc}, err={err[:100]})")

        # 3. List: verify both items present
        rc, out, err = run_cmd(["list"])
        if rc == 0 and "milk" in out.lower() and "dog" in out.lower():
            score += 1
            details.append("PASS: list shows both items")
        else:
            details.append(f"FAIL: list ({out[:100]})")

        # 4. Complete item 1
        rc, out, err = run_cmd(["complete", "1"])
        if rc == 0:
            score += 1
            details.append("PASS: complete item 1")
        else:
            details.append(f"FAIL: complete item 1 (rc={rc}, err={err[:100]})")

        # 5. List: verify completion status
        rc, out, err = run_cmd(["list"])
        has_status = any(w in out.lower() for w in ["\u2713", "done", "complete", "\u2714", "[x]", "yes"])
        if rc == 0 and has_status:
            score += 1
            details.append("PASS: list shows completion status")
        else:
            details.append(f"FAIL: list completion status ({out[:100]})")

        # 6. Delete item 2
        rc, out, err = run_cmd(["delete", "2"])
        if rc == 0:
            score += 1
            details.append("PASS: delete item 2")
        else:
            details.append(f"FAIL: delete item 2 (rc={rc}, err={err[:100]})")

        # 7. List: verify deletion
        rc, out, err = run_cmd(["list"])
        if rc == 0 and "dog" not in out.lower():
            score += 1
            details.append("PASS: list confirms deletion")
        else:
            details.append(f"FAIL: list after delete ({out[:100]})")

    return {"score": score, "max": max_score, "details": details}


def task_1_2_url_shortener(model_key: str) -> dict:
    """URL Shortener (FastAPI)."""
    prompt = (
        "Write a complete Python FastAPI application that implements a URL shortener. "
        "Requirements: POST /shorten accepts a URL and returns a short code. "
        "GET /{code} redirects to the original URL. "
        "GET /stats/{code} returns click count and creation date. "
        "SQLite persistence. Input validation (reject non-URLs). "
        "Rate limiting (10 requests/minute per IP). "
        "Respond with ONLY the Python code, no explanation."
    )
    result = chat([{"role": "user", "content": prompt}], model_key, max_tokens=4096)
    code = extract_code(result["content"], "python")
    result["code"] = code
    result["lines"] = len(code.splitlines())

    try:
        compile(code, "<shortener>", "exec")
        result["syntax_valid"] = True
    except SyntaxError as e:
        result["syntax_valid"] = False
        result["syntax_error"] = str(e)

    features = {
        "post_endpoint": bool(re.search(r"@app\.(post|route).*shorten|post.*shorten", code, re.I)),
        "get_redirect": bool(re.search(r"RedirectResponse|redirect|@app\.get.*\{.*code", code, re.I)),
        "stats_endpoint": bool(re.search(r"stats|@app\.get.*stats", code, re.I)),
        "sqlite": bool(re.search(r"sqlite3|import sqlite|aiosqlite|sqlalchemy.*sqlite|sqlite://", code, re.I)),
        "validation": bool(re.search(r"urlparse|validators|HttpUrl|http[s]?://|https\?://|re\.match.*http|url.*valid", code, re.I)),
        "rate_limit": bool(re.search(r"rate.?limit|slowapi|limiter|requests?.?per", code, re.I)),
        "error_handling": bool(re.search(r"HTTPException|raise |except |status_code", code, re.I)),
        "http_status": bool(re.search(r"status_code.*[234]\d\d|30[127]|422|429", code)),
    }
    result["features"] = features
    result["feature_score"] = sum(features.values())
    result["feature_max"] = len(features)

    # No functional test: would require running a FastAPI server
    result["functional"] = {"score": 0, "max": 6, "details": "Manual or server-based testing required"}

    feat_pct = result["feature_score"] / result["feature_max"]
    composite = feat_pct * 60
    if not result["syntax_valid"]:
        composite *= (2 / 3)
    result["composite_score"] = round(composite, 1)

    return result


def task_1_3_lru_cache(model_key: str) -> dict:
    """LRU Cache (TypeScript)."""
    prompt = (
        "Write a TypeScript module that implements a Least Recently Used (LRU) cache with the following: "
        "constructor takes capacity, get(key) returns value or -1, put(key, value) inserts or updates, "
        "evicts least recently used when at capacity. O(1) time for both get and put. "
        "Include a test suite using console.assert that verifies: basic get/put, capacity eviction, "
        "access-order update, overwrite existing key. Respond with ONLY the TypeScript code, no explanation."
    )
    result = chat([{"role": "user", "content": prompt}], model_key, max_tokens=4096)
    code = extract_code(result["content"], "typescript")
    result["code"] = code
    result["lines"] = len(code.splitlines())

    result["syntax_valid"] = check_typescript_syntax(code)

    features = {
        "constructor_capacity": bool(re.search(r"constructor.*capacity|new.*\(\d", code, re.I)),
        "get_method": bool(re.search(r"get\s*\(", code)),
        "put_method": bool(re.search(r"put\s*\(|set\s*\(", code)),
        "o1_impl": bool(re.search(r"Map|HashMap|map.*\.get|map.*\.set", code, re.I)),
        "eviction": bool(re.search(r"evict|delete|remove.*least|size.*>.*capacity|capacity", code, re.I)),
        "test_suite": bool(re.search(r"console\.assert|assert\(|test\(", code)),
    }
    result["features"] = features
    result["feature_score"] = sum(features.values())
    result["feature_max"] = len(features)

    result["functional"] = run_typescript_test(code)

    feat_pct = result["feature_score"] / result["feature_max"]
    func_pct = result["functional"]["score"] / result["functional"]["max"] if result["functional"]["max"] > 0 else 0
    composite = (feat_pct * 60) + (func_pct * 40)
    if not result["syntax_valid"]:
        composite *= (2 / 3)
    result["composite_score"] = round(composite, 1)

    return result


def check_typescript_syntax(code: str) -> bool:
    """Check TypeScript syntax using npx tsc if available, else heuristic."""
    with tempfile.TemporaryDirectory() as tmpdir:
        ts_path = os.path.join(tmpdir, "test.ts")
        with open(ts_path, "w") as f:
            f.write(code)
        try:
            r = subprocess.run(
                ["npx", "--yes", "typescript", "tsc", "--noEmit", "--strict", ts_path],
                capture_output=True, text=True, timeout=30, cwd=tmpdir
            )
            if r.returncode == 0:
                return True
        except (FileNotFoundError, subprocess.TimeoutExpired):
            pass

    # Fallback: strip type annotations and check as JS
    js_code = re.sub(r":\s*\w+(\[\])?(\s*\|\s*\w+)*", "", code)
    js_code = re.sub(r"<\w+(\s*,\s*\w+)*>", "", js_code)
    js_code = re.sub(r"(interface|type)\s+\w+\s*\{[^}]*\}", "", js_code)
    opens = sum(1 for c in js_code if c in "({[")
    closes = sum(1 for c in js_code if c in ")}]")
    return abs(opens - closes) <= 2


def run_typescript_test(code: str) -> dict:
    """Try to run the TypeScript code and check assert results."""
    with tempfile.TemporaryDirectory() as tmpdir:
        ts_path = os.path.join(tmpdir, "test.ts")
        with open(ts_path, "w") as f:
            f.write(code)

        for runner in [["npx", "--yes", "tsx", ts_path], ["npx", "--yes", "ts-node", ts_path]]:
            try:
                r = subprocess.run(
                    runner, capture_output=True, text=True, timeout=30, cwd=tmpdir
                )
                if r.returncode == 0:
                    asserts_in_code = len(re.findall(r"console\.assert", code))
                    return {"score": asserts_in_code, "max": max(asserts_in_code, 4),
                            "details": f"All {asserts_in_code} assertions passed"}
                else:
                    return {"score": 0, "max": 4, "details": f"Runtime error: {r.stderr[:200]}"}
            except (FileNotFoundError, subprocess.TimeoutExpired):
                continue

    return {"score": 0, "max": 4, "details": "No TypeScript runner available"}


def task_2_1_bugfix(model_key: str) -> dict:
    """Bug fix across files: Express.js auth bug."""
    fixture_dir = FIXTURES_DIR / "express-bug-app"
    server_js = (fixture_dir / "server.js").read_text()
    auth_js = (fixture_dir / "middleware" / "auth.js").read_text()
    items_js = (fixture_dir / "routes" / "items.js").read_text()

    prompt = (
        "This Express.js app has a bug: the authentication middleware checks "
        "`req.headers.authorization` but the login endpoint sets the token in "
        "`req.headers['x-auth-token']`. Find and fix the inconsistency. "
        "The fix should be minimal: change the fewest lines possible. "
        "Explain what you changed and why.\n\n"
        f"**server.js:**\n```javascript\n{server_js}\n```\n\n"
        f"**middleware/auth.js:**\n```javascript\n{auth_js}\n```\n\n"
        f"**routes/items.js:**\n```javascript\n{items_js}\n```"
    )
    result = chat([{"role": "user", "content": prompt}], model_key, max_tokens=4096)

    content = result["content"]
    scoring = {
        "found_bug": bool(re.search(r"x-auth-token|authorization.*mismatch|header.*inconsisten", content, re.I)),
        "minimal_fix": not bool(re.search(r"rewrite|refactor|restructur", content, re.I)),
        "no_breakage": True,
        "explanation_quality": 0,
    }
    scoring["auto_score"] = sum([
        scoring["found_bug"] * 3,
        scoring["minimal_fix"] * 2,
    ])
    result["scoring"] = scoring
    return result


def task_2_2_pagination(model_key: str) -> dict:
    """Add pagination to Flask app."""
    fixture_dir = FIXTURES_DIR / "flask-pagination-app"
    app_py = (fixture_dir / "app.py").read_text()
    test_py = (fixture_dir / "test_app.py").read_text()

    prompt = (
        "Add pagination to this REST API. The /items endpoint currently returns all items. "
        "Add: query params `page` (default 1) and `per_page` (default 20), "
        "response includes `total`, `page`, `per_page`, `total_pages` metadata, "
        "proper handling of out-of-range pages. "
        "Update the existing tests to cover pagination. Don't break existing functionality.\n\n"
        f"**app.py:**\n```python\n{app_py}\n```\n\n"
        f"**test_app.py:**\n```python\n{test_py}\n```"
    )
    result = chat([{"role": "user", "content": prompt}], model_key, max_tokens=8192)

    content = result["content"]
    features = {
        "page_param": bool(re.search(r"page.*request\.args|args.*get.*page|page.*param", content, re.I)),
        "per_page_param": bool(re.search(r"per_page|page_size|limit", content, re.I)),
        "metadata": bool(re.search(r"total.*page|total_pages|pagination.*meta", content, re.I)),
        "out_of_range": bool(re.search(r"out.?of.?range|page.*>.*total|404|empty|invalid.*page", content, re.I)),
        "tests_updated": bool(re.search(r"def test.*pag|test.*page|assert.*page", content, re.I)),
    }
    result["features"] = features
    result["feature_score"] = sum(features.values())
    result["feature_max"] = len(features)
    return result


def task_3_1_debug(model_key: str) -> dict:
    """Debug from error log: reasoning task."""
    prompt = textwrap.dedent("""\
        Here's an error log from a production Python application. Diagnose the root cause and propose a fix.

        ```
        [2026-05-04 03:14:22] ERROR: ConnectionPool.get_connection() timeout after 30s
        [2026-05-04 03:14:22] ERROR: Active connections: 20/20, Idle: 0, Waiting: 47
        [2026-05-04 03:14:22] WARNING: Transaction started at 03:12:01 still open (connection #7)
        [2026-05-04 03:14:22] DEBUG: Last query on #7: SELECT * FROM orders JOIN products ON orders.product_id = products.id WHERE orders.status = 'pending'
        [2026-05-04 03:14:22] DEBUG: Query returned 2,847,293 rows at 03:12:03
        [2026-05-04 03:14:22] DEBUG: Application processing rows in loop (no batch/cursor)
        ```
    """)
    result = chat([{"role": "user", "content": prompt}], model_key, max_tokens=4096)

    content = result["content"]
    rubric = {
        "pool_exhaustion": bool(re.search(r"pool.*exhaust|connection.*pool.*full|no.*idle|all.*connections.*use", content, re.I)),
        "long_transaction": bool(re.search(r"long.*running.*transaction|transaction.*open.*too|held.*connection", content, re.I)),
        "large_query": bool(re.search(r"2.?8.*million|unbounded|SELECT \*|no.*LIMIT|too many rows|large.*result", content, re.I)),
        "row_by_row": bool(re.search(r"row.?by.?row|no.*cursor|no.*batch|processing.*loop|server.?side.*cursor", content, re.I)),
        "pool_config_fix": bool(re.search(r"timeout|pool.*size|max.*connection|connection.*limit", content, re.I)),
        "query_fix": bool(re.search(r"LIMIT|cursor|pagination|batch.*process|WHERE.*clause", content, re.I)),
        "transaction_fix": bool(re.search(r"transaction.*scope|commit|close.*transaction|context.*manager", content, re.I)),
    }
    auto_score = sum([
        rubric["pool_exhaustion"] * 2,
        rubric["long_transaction"] * 2,
        rubric["large_query"] * 2,
        rubric["row_by_row"] * 1,
        rubric["pool_config_fix"] * 1,
        rubric["query_fix"] * 1,
        rubric["transaction_fix"] * 1,
    ])
    result["rubric"] = rubric
    result["auto_score"] = auto_score
    result["max_score"] = 10
    result["manual_score"] = None
    return result


def task_3_2_architecture(model_key: str) -> dict:
    """Architecture decision: CRDT vs OT."""
    prompt = (
        "You're building a real-time collaborative document editor (like Google Docs). "
        "Users need to see each other's cursors and edits within 100ms. "
        "Expected scale: 10,000 concurrent documents, 50 users per document max. "
        "Compare CRDT vs OT (Operational Transform) approaches. Recommend one with justification. "
        "Outline the system architecture including: conflict resolution, data persistence, "
        "WebSocket management, and offline support."
    )
    result = chat([{"role": "user", "content": prompt}], model_key, max_tokens=4096)

    content = result["content"]
    rubric = {
        "crdt_accurate": bool(re.search(r"conflict.?free|commutative|convergent|merge.*automatic|no.*central", content, re.I)),
        "ot_accurate": bool(re.search(r"operational.?transform|transform.*function|server.*canonical|central.*server", content, re.I)),
        "recommendation": bool(re.search(r"recommend|suggest|choose|prefer|go with", content, re.I)),
        "architecture": bool(re.search(r"websocket|persistence|conflict.*resolut", content, re.I)),
        "edge_cases": bool(re.search(r"offline|partition|network|large.*document|latency", content, re.I)),
    }
    auto_score = sum([
        rubric["crdt_accurate"] * 2,
        rubric["ot_accurate"] * 2,
        rubric["recommendation"] * 2,
        rubric["architecture"] * 2,
        rubric["edge_cases"] * 2,
    ])
    result["rubric"] = rubric
    result["auto_score"] = auto_score
    result["max_score"] = 10
    result["manual_score"] = None
    return result


def task_3_3_bayes(model_key: str) -> dict:
    """Bayes theorem problem.
    FIXED from Round 5: correct answer is 40.54%, not 41.67%.
    See CHANGES.md for the math."""
    prompt = (
        "A company has 3 servers. Server A handles 40% of traffic and has a 1% error rate. "
        "Server B handles 35% of traffic and has a 2% error rate. "
        "Server C handles 25% of traffic and has a 3% error rate. "
        "A user reports an error. What is the probability the error came from Server C? "
        "Show your work using Bayes' theorem."
    )
    result = chat([{"role": "user", "content": prompt}], model_key, max_tokens=4096)

    content = result["content"]
    # FIXED: P(C|E) = 0.0075/0.0185 = 0.40541... = 40.54%
    # Accept 40.54%, 40.5%, 15/37, 0.4054, and the old wrong answer 41.67% (will note separately)
    correct_answer = bool(re.search(r"40\.5[0-9]|0\.405[0-9]|15\s*/\s*37|75\s*/\s*185", content))
    old_wrong_answer = bool(re.search(r"41\.6[0-9]|0\.416[0-9]|5/12|0\.4167", content))
    uses_bayes = bool(re.search(r"[Bb]ayes|P\(.*\|.*\)|posterior|prior.*likelihood", content))
    shows_work = bool(re.search(r"0\.004|0\.007|0\.0075|0\.0185|P\(E\)", content))

    rubric = {
        "correct_answer": correct_answer,
        "old_wrong_answer": old_wrong_answer,  # Tracked for transparency
        "uses_bayes": uses_bayes,
        "shows_work": shows_work,
    }
    auto_score = sum([
        correct_answer * 2,
        uses_bayes * 2,
        shows_work * 1,
    ])
    result["rubric"] = rubric
    result["auto_score"] = auto_score
    result["max_score"] = 5
    result["manual_score"] = None
    return result


def task_4_1_structured_output(model_key: str) -> dict:
    """Structured JSON output test."""
    prompt = (
        "Given this JSON schema, generate 5 realistic test records for a user management system. "
        'Schema: {id: uuid, name: string, email: string, role: "admin"|"user"|"moderator", '
        'created_at: ISO8601, preferences: {theme: "light"|"dark", notifications: boolean, language: string}}. '
        "Return ONLY valid JSON array, no markdown fences, no explanation."
    )
    result = chat([{"role": "user", "content": prompt}], model_key, max_tokens=4096)

    content = result["content"].strip()
    json_match = re.search(r"\[.*\]", content, re.DOTALL)
    json_str = json_match.group(0) if json_match else content

    scoring = {
        "valid_json": False,
        "five_records": False,
        "fields_match": False,
        "enums_valid": False,
        "no_extra_text": content.startswith("[") and content.endswith("]"),
    }

    try:
        data = json.loads(json_str)
        scoring["valid_json"] = True
        scoring["five_records"] = isinstance(data, list) and len(data) == 5

        if scoring["five_records"]:
            required_fields = {"id", "name", "email", "role", "created_at", "preferences"}
            all_have_fields = all(required_fields.issubset(set(r.keys())) for r in data)
            scoring["fields_match"] = all_have_fields

            valid_roles = {"admin", "user", "moderator"}
            valid_themes = {"light", "dark"}
            enums_ok = all(
                r.get("role") in valid_roles and
                r.get("preferences", {}).get("theme") in valid_themes and
                isinstance(r.get("preferences", {}).get("notifications"), bool)
                for r in data
            )
            scoring["enums_valid"] = enums_ok
    except (json.JSONDecodeError, TypeError):
        pass

    result["scoring"] = scoring
    result["auto_score"] = sum(scoring.values())
    result["max_score"] = 5
    return result


def task_4_2_tool_use(model_key: str) -> dict:
    """Multi-step tool use simulation."""
    prompt = (
        "You have access to these tools: read_file(path), write_file(path, content), execute(command). "
        "Task: Read the file config.yaml, find the database.host value, ping that host to check connectivity, "
        "and write the results to health-check.txt with timestamp. "
        "Respond with the exact sequence of tool calls you would make, in order, as JSON."
    )
    result = chat([{"role": "user", "content": prompt}], model_key, max_tokens=4096)

    content = result["content"]
    scoring = {
        "read_file_call": bool(re.search(r"read_file.*config\.yaml|read.*config", content, re.I | re.DOTALL)),
        "parses_yaml": bool(re.search(r"database.*host|parse.*yaml|find.*host", content, re.I | re.DOTALL)),
        "ping_call": bool(re.search(r"execute.*ping|ping.*host|execute.*\bping\b", content, re.I | re.DOTALL)),
        "write_file_call": bool(re.search(r"write_file.*health|write.*health.?check", content, re.I | re.DOTALL)),
        "logical_order": True,
    }

    read_pos = re.search(r"read_file", content, re.I)
    exec_pos = re.search(r"execute.*ping|ping", content, re.I)
    write_pos = re.search(r"write_file", content, re.I)
    if read_pos and exec_pos and write_pos:
        scoring["logical_order"] = read_pos.start() < exec_pos.start() < write_pos.start()

    result["scoring"] = scoring
    result["auto_score"] = sum(scoring.values())
    result["max_score"] = 5
    return result


def task_speed(model_key: str, task_id: str, prompt: str, runs: int = 3) -> dict:
    """Speed microbenchmark using streaming for TTFT."""
    results = []
    for i in range(runs):
        log(f"  Speed {task_id} run {i+1}/{runs}...")
        r = chat_streaming_ttft(
            [{"role": "user", "content": prompt}],
            model_key, max_tokens=4096
        )
        results.append({
            "tokens_per_sec": r["tokens_per_sec"],
            "ttft_ms": r["ttft_ms"],
            "completion_tokens": r["completion_tokens"],
            "wall_time_s": r["wall_time_s"],
        })
        time.sleep(1)  # Brief cooldown

    tps_values = [r["tokens_per_sec"] for r in results]
    ttft_values = [r["ttft_ms"] for r in results]

    return {
        "task_id": task_id,
        "runs": results,
        "median_tokens_per_sec": round(statistics.median(tps_values), 1),
        "median_ttft_ms": round(statistics.median(ttft_values), 1),
        "median_completion_tokens": int(statistics.median([r["completion_tokens"] for r in results])),
    }


# ─── Speed Prompts ───────────────────────────────────────────────────────────

SPEED_PROMPTS = {
    "5.1": "Write a Python function that checks if a string is a valid IPv4 address. No imports. Respond with ONLY the code.",
    "5.2": "Write a Python class that implements a basic binary search tree with insert, search, delete, and in-order traversal. Include docstrings. Respond with ONLY the code.",
    "5.3": (
        "Write a complete Python script that implements a command-line Markdown to HTML converter. "
        "Support: headings (h1-h6), bold, italic, code blocks, inline code, unordered lists, ordered lists, "
        "links, horizontal rules. Use only the standard library. Include a __main__ block that reads from stdin. "
        "Respond with ONLY the code."
    ),
}

# ─── Orchestrator ────────────────────────────────────────────────────────────

TASK_MAP = {
    "1.1": ("CLI Todo App", task_1_1_todo),
    "1.2": ("URL Shortener", task_1_2_url_shortener),
    "1.3": ("LRU Cache (TS)", task_1_3_lru_cache),
    "2.1": ("Bug Fix", task_2_1_bugfix),
    "2.2": ("Pagination", task_2_2_pagination),
    "3.1": ("Debug Log", task_3_1_debug),
    "3.2": ("Architecture", task_3_2_architecture),
    "3.3": ("Bayes Theorem", task_3_3_bayes),
    "4.1": ("Structured Output", task_4_1_structured_output),
    "4.2": ("Tool Use", task_4_2_tool_use),
}


def run_benchmark(model_key: str):
    """Run all tasks for a single model."""
    model_info = MODELS[model_key]
    model_dir = RESULTS_DIR / model_key
    model_dir.mkdir(parents=True, exist_ok=True)

    log(f"=== Benchmarking {model_info['name']} ({model_info['model_id']}) ===")

    all_results = {
        "model": model_key,
        "name": model_info["name"],
        "model_id": model_info["model_id"],
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "harness": "frontier-bakeoff-v1",
    }

    # Run quality tasks (1.1 - 4.2)
    for task_id, (task_name, task_fn) in TASK_MAP.items():
        log(f"Running task {task_id}: {task_name}...")
        try:
            result = task_fn(model_key)
            all_results[f"task_{task_id}"] = result
            log(f"  Done. Wall time: {result.get('wall_time_s', '?')}s")

            with open(model_dir / f"task_{task_id.replace('.', '_')}.json", "w") as f:
                json.dump(result, f, indent=2, default=str)

        except Exception as e:
            log(f"  ERROR on task {task_id}: {e}")
            all_results[f"task_{task_id}"] = {"error": str(e)}

    # Run speed benchmarks (5.1 - 5.3)
    for task_id, prompt in SPEED_PROMPTS.items():
        log(f"Running speed task {task_id}...")
        try:
            result = task_speed(model_key, task_id, prompt)
            all_results[f"task_{task_id}"] = result
            log(f"  Median: {result['median_tokens_per_sec']} tok/s, "
                f"TTFT: {result['median_ttft_ms']}ms")

            with open(model_dir / f"task_{task_id.replace('.', '_')}.json", "w") as f:
                json.dump(result, f, indent=2, default=str)

        except Exception as e:
            log(f"  ERROR on speed task {task_id}: {e}")
            all_results[f"task_{task_id}"] = {"error": str(e)}

    with open(model_dir / "all_results.json", "w") as f:
        json.dump(all_results, f, indent=2, default=str)

    log(f"=== {model_info['name']} complete. Results in {model_dir} ===")
    print_model_summary(all_results)
    return all_results


def print_model_summary(results: dict):
    """Print a quick summary table for one model."""
    name = results.get("name", "Unknown")
    print(f"\n{'='*60}")
    print(f"  SUMMARY: {name}")
    print(f"{'='*60}")

    for tid in ["1.1", "1.2", "1.3"]:
        key = f"task_{tid}"
        if key in results and "composite_score" in results[key]:
            r = results[key]
            func = r.get("functional", {})
            print(f"  Task {tid}: Composite={r['composite_score']}, "
                  f"Features={r['feature_score']}/{r['feature_max']}, "
                  f"Functional={func.get('score','?')}/{func.get('max','?')}, "
                  f"Syntax={'Y' if r.get('syntax_valid') else 'N'}")

    for tid in ["3.1", "3.2", "3.3"]:
        key = f"task_{tid}"
        if key in results and "auto_score" in results[key]:
            r = results[key]
            print(f"  Task {tid}: Auto={r['auto_score']}/{r['max_score']}")

    for tid in ["4.1", "4.2"]:
        key = f"task_{tid}"
        if key in results and "auto_score" in results[key]:
            r = results[key]
            print(f"  Task {tid}: Score={r['auto_score']}/{r['max_score']}")

    for tid in ["5.1", "5.2", "5.3"]:
        key = f"task_{tid}"
        if key in results and "median_tokens_per_sec" in results[key]:
            r = results[key]
            print(f"  Task {tid}: {r['median_tokens_per_sec']} tok/s, TTFT={r['median_ttft_ms']}ms")

    print(f"{'='*60}\n")


# ─── Cross-Model Comparison ─────────────────────────────────────────────────

def compare_models():
    """Generate comparison tables from saved results."""
    all_data = {}
    for model_key in MODELS:
        results_file = RESULTS_DIR / model_key / "all_results.json"
        if results_file.exists():
            with open(results_file) as f:
                all_data[model_key] = json.load(f)

    if not all_data:
        log("No results found. Run benchmarks first.")
        return

    log(f"Comparing {len(all_data)} models: {', '.join(all_data.keys())}")

    # Table 1: Speed
    print("\n## Table 1: Speed Comparison (wall-clock)")
    print(f"| {'Model':<20} | {'Short t/s':>9} | {'Short TTFT':>10} | {'Med t/s':>7} | {'Med TTFT':>8} | {'Long t/s':>8} | {'Long TTFT':>9} |")
    print(f"|{'-'*22}|{'-'*11}|{'-'*12}|{'-'*9}|{'-'*10}|{'-'*10}|{'-'*11}|")
    for mk, data in all_data.items():
        row = [MODELS[mk]["name"][:20]]
        for tid in ["5.1", "5.2", "5.3"]:
            t = data.get(f"task_{tid}", {})
            tps = t.get("median_tokens_per_sec", "-")
            ttft = t.get("median_ttft_ms", "-")
            row.extend([tps, ttft])
        print(f"| {row[0]:<20} | {row[1]:>9} | {row[2]:>10} | {row[3]:>7} | {row[4]:>8} | {row[5]:>8} | {row[6]:>9} |")

    # Table 2: Coding Quality
    print("\n## Table 2: Coding Quality")
    print(f"| {'Model':<20} | {'Todo':>5} | {'API':>5} | {'LRU':>5} | {'BugFix':>6} | {'Paginate':>8} | {'Avg':>5} |")
    print(f"|{'-'*22}|{'-'*7}|{'-'*7}|{'-'*7}|{'-'*8}|{'-'*10}|{'-'*7}|")
    for mk, data in all_data.items():
        scores = []
        t11 = data.get("task_1.1", {}).get("composite_score", "-")
        t12 = data.get("task_1.2", {}).get("composite_score", "-")
        t13 = data.get("task_1.3", {}).get("composite_score", "-")
        t21 = data.get("task_2.1", {}).get("scoring", {}).get("auto_score", "-")
        t22 = data.get("task_2.2", {}).get("feature_score", "-")
        for v in [t11, t12, t13]:
            if isinstance(v, (int, float)):
                scores.append(v)
        avg = round(sum(scores) / len(scores), 1) if scores else "-"
        print(f"| {MODELS[mk]['name'][:20]:<20} | {t11:>5} | {t12:>5} | {t13:>5} | {t21:>6} | {t22:>8} | {avg:>5} |")

    # Table 3: Reasoning
    print("\n## Table 3: Reasoning Quality")
    print(f"| {'Model':<20} | {'Debug(10)':>9} | {'Arch(10)':>8} | {'Bayes(5)':>8} | {'Avg':>5} |")
    print(f"|{'-'*22}|{'-'*11}|{'-'*10}|{'-'*10}|{'-'*7}|")
    for mk, data in all_data.items():
        scores = []
        for tid, mx in [("3.1", 10), ("3.2", 10), ("3.3", 5)]:
            t = data.get(f"task_{tid}", {})
            ms = t.get("manual_score")
            auto = t.get("auto_score", "-")
            scores.append(ms if ms is not None else auto)
        norm = []
        for s, mx in zip(scores, [10, 10, 5]):
            if isinstance(s, (int, float)):
                norm.append(s / mx * 10)
        avg = round(sum(norm) / len(norm), 1) if norm else "-"
        print(f"| {MODELS[mk]['name'][:20]:<20} | {scores[0]:>9} | {scores[1]:>8} | {scores[2]:>8} | {avg:>5} |")

    # Table 4: Tool Use
    print("\n## Table 4: Tool Use & Instruction Following")
    print(f"| {'Model':<20} | {'Struct(5)':>9} | {'Tools(5)':>8} | {'Avg':>5} |")
    print(f"|{'-'*22}|{'-'*11}|{'-'*10}|{'-'*7}|")
    for mk, data in all_data.items():
        t41 = data.get("task_4.1", {}).get("auto_score", "-")
        t42 = data.get("task_4.2", {}).get("auto_score", "-")
        vals = [v for v in [t41, t42] if isinstance(v, (int, float))]
        avg = round(sum(vals) / len(vals), 1) if vals else "-"
        print(f"| {MODELS[mk]['name'][:20]:<20} | {t41:>9} | {t42:>8} | {avg:>5} |")

    # Table 5: Overall Rankings
    print("\n## Table 5: Overall Rankings (Coding 40%, Reasoning 20%, Tool Use 20%, Speed 20%)")
    rankings = []
    for mk, data in all_data.items():
        coding_scores = []
        for tid in ["1.1", "1.2", "1.3"]:
            v = data.get(f"task_{tid}", {}).get("composite_score")
            if isinstance(v, (int, float)):
                coding_scores.append(v)
        coding_avg = sum(coding_scores) / len(coding_scores) if coding_scores else 0

        reasoning_scores = []
        for tid, mx in [("3.1", 10), ("3.2", 10), ("3.3", 5)]:
            t = data.get(f"task_{tid}", {})
            s = t.get("manual_score") if t.get("manual_score") is not None else t.get("auto_score", 0)
            if isinstance(s, (int, float)):
                reasoning_scores.append(s / mx * 100)
        reasoning_avg = sum(reasoning_scores) / len(reasoning_scores) if reasoning_scores else 0

        tool_scores = []
        for tid in ["4.1", "4.2"]:
            s = data.get(f"task_{tid}", {}).get("auto_score", 0)
            if isinstance(s, (int, float)):
                tool_scores.append(s / 5 * 100)
        tool_avg = sum(tool_scores) / len(tool_scores) if tool_scores else 0

        speed_vals = []
        for tid in ["5.1", "5.2", "5.3"]:
            v = data.get(f"task_{tid}", {}).get("median_tokens_per_sec", 0)
            if isinstance(v, (int, float)):
                speed_vals.append(v)
        speed_raw = sum(speed_vals) / len(speed_vals) if speed_vals else 0

        rankings.append({
            "model": mk,
            "name": MODELS[mk]["name"],
            "coding": round(coding_avg, 1),
            "reasoning": round(reasoning_avg, 1),
            "tool_use": round(tool_avg, 1),
            "speed_raw": round(speed_raw, 1),
        })

    max_speed = max(r["speed_raw"] for r in rankings) if rankings else 1
    for r in rankings:
        r["speed"] = round(r["speed_raw"] / max_speed * 100, 1) if max_speed > 0 else 0
        r["weighted"] = round(
            r["coding"] * 0.4 +
            r["reasoning"] * 0.2 +
            r["tool_use"] * 0.2 +
            r["speed"] * 0.2,
            1
        )

    rankings.sort(key=lambda r: r["weighted"], reverse=True)

    print(f"| {'Rank':>4} | {'Model':<20} | {'Coding':>6} | {'Reason':>6} | {'Tools':>5} | {'Speed':>5} | {'Total':>5} |")
    print(f"|{'-'*6}|{'-'*22}|{'-'*8}|{'-'*8}|{'-'*7}|{'-'*7}|{'-'*7}|")
    for i, r in enumerate(rankings, 1):
        print(f"| {i:>4} | {r['name'][:20]:<20} | {r['coding']:>6} | {r['reasoning']:>6} | {r['tool_use']:>5} | {r['speed']:>5} | {r['weighted']:>5} |")

    if rankings:
        print(f"\n### Winners")
        best_coding = max(rankings, key=lambda r: r["coding"])
        best_reasoning = max(rankings, key=lambda r: r["reasoning"])
        best_overall = rankings[0]
        print(f"  Best for Coding:    {best_coding['name']} ({best_coding['coding']})")
        print(f"  Best for Reasoning: {best_reasoning['name']} ({best_reasoning['reasoning']})")
        print(f"  Best Overall:       {best_overall['name']} ({best_overall['weighted']})")

    comparison_file = RESULTS_DIR / "comparison.json"
    with open(comparison_file, "w") as f:
        json.dump(rankings, f, indent=2)
    log(f"Comparison saved to {comparison_file}")


# ─── Main ────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Frontier Bakeoff: Fable 5 vs Opus 4.8 vs Sonnet 4.6 vs GPT-5.5")
    parser.add_argument("--model", choices=list(MODELS.keys()) + ["all"],
                        help="Model to benchmark (or 'all')")
    parser.add_argument("--compare", action="store_true",
                        help="Generate comparison tables from existing results")
    parser.add_argument("--task", help="Run a single task (e.g., '1.1', '5.2')")
    args = parser.parse_args()

    if args.compare:
        compare_models()
        return

    if not args.model:
        parser.print_help()
        return

    models_to_run = list(MODELS.keys()) if args.model == "all" else [args.model]

    for model_key in models_to_run:
        if args.task:
            if args.task.startswith("5."):
                result = task_speed(model_key, args.task, SPEED_PROMPTS[args.task])
            elif args.task in TASK_MAP:
                _, fn = TASK_MAP[args.task]
                result = fn(model_key)
            else:
                log(f"Unknown task: {args.task}")
                return
            print(json.dumps(result, indent=2, default=str))
        else:
            run_benchmark(model_key)

    if len(models_to_run) > 1:
        compare_models()


if __name__ == "__main__":
    main()
