# Frontier Bakeoff: Changes from Round 5

This document tracks every change between the Round 5 homelab benchmark
(`coder-templates/scripts/benchmarks/round5/benchmark.py`) and this
frontier cloud adaptation. Transparency is the goal: if a score changed,
you should be able to trace it to a specific harness change below.

## Why a new harness

Round 5 tested five local quantized models on a single RTX 5090 via
llama.cpp. This round tests two frontier cloud models (Claude Fable 5 and
Claude Opus 4.8) via the Anthropic Messages API. The local serving
machinery does not apply; the tasks, prompts, fixtures, and scoring logic
are carried forward.

## Changes

### 1. Backend: llama.cpp HTTP to Anthropic SDK

**What:** Replaced raw `requests.post()` to `localhost:8080/v1/chat/completions`
with the `anthropic` Python SDK calling the Messages API.

**Why:** Cloud models use the Anthropic Messages API, not an OpenAI-compatible
local endpoint. Both Fable 5 and Opus 4.8 are Anthropic models, so a single
SDK covers both.

**Impact on scores:** None. The prompts, system messages, and scoring logic
are unchanged. Only the transport layer changed.

### 2. Model switching removed

**What:** Removed `switch_model()`, `wait_for_server()`, and the dependency
on `~/bin/llm-switch.sh`. Models are now selected by passing a model ID
string to the API.

**Why:** Cloud models do not require restarting a local server.

**Impact on scores:** None.

### 3. Bug fix: Bayes theorem answer (Task 3.3)

**What:** The Round 5 regex expected `41.67%` as the correct answer. The
actual correct answer is `40.54%`.

**Math:**
- P(A) = 0.40, P(E|A) = 0.01, so P(E and A) = 0.004
- P(B) = 0.35, P(E|B) = 0.02, so P(E and B) = 0.007
- P(C) = 0.25, P(E|C) = 0.03, so P(E and C) = 0.0075
- P(E) = 0.004 + 0.007 + 0.0075 = **0.0185**
- P(C|E) = 0.0075 / 0.0185 = **0.40541...** = **40.54%**

The Round 5 code comment said `P(C|E) = 0.75/1.80`, implying P(E) = 0.018.
That dropped the 0.0005 contribution from rounding P(E and C) to 0.0075 vs
using 0.25 * 0.03. The denominator should be 0.0185, not 0.018.

**Why:** Every Round 5 model scored `correct_answer: False` on this task.
Frontier models will almost certainly compute the right answer and would be
penalized by the old regex.

**Impact on scores:** Models producing 40.54% (or equivalent fractions like
15/37) now correctly get the 2-point correct_answer credit. Not comparable
to Round 5 scores on this task.

### 4. TypeScript functional test wired up (Task 1.3)

**What:** Round 5 could not run the TypeScript functional test because
`npx tsx` was not available on the homelab. All models scored 0/4 on the
functional portion, capping composite at 60. This environment has `tsx`
installed, so the functional test now runs.

**Why:** Frontier models deserve the full scoring range. A capped score
masks quality differences.

**Impact on scores:** Composite scores for Task 1.3 can now reach 100
instead of being capped at 60. Not comparable to Round 5 scores on this
task.

### 5. Speed measurement methodology

**What:** Round 5 extracted `timings.predicted_per_second` and
`timings.prompt_ms` from llama.cpp's response body. Cloud APIs do not
expose per-token generation timing. This harness uses:
- **Streaming TTFT:** Time from request start to the first content event.
- **Wall-clock tok/s:** `output_tokens / wall_time` as an approximation.

**Why:** Direct tok/s from the inference engine is not available through
cloud APIs. Wall-clock includes network latency, which makes absolute
numbers incomparable to Round 5, but relative comparison between the two
cloud models is still valid.

**Impact on scores:** Speed numbers are not comparable to Round 5.
Relative ranking between Fable 5 and Opus 4.8 is valid.

### 6. Thinking token handling

**What:** Both Fable 5 and Opus 4.8 use extended thinking. The harness
requests thinking with `budget_tokens` and extracts thinking content from
the API's native `thinking` content blocks, rather than regex-stripping
`<think>` tags from plaintext.

**Why:** The Anthropic API returns thinking as structured content blocks,
not inline XML. The old regex approach was a workaround for llama.cpp's
`--reasoning-format deepseek` flag.

**Impact on scores:** Cleaner separation of thinking vs. output content.
No scoring impact, just more reliable extraction.

### 7. Fable 5 refusal handling

**What:** Added handling for `stop_reason: "refusal"`. If Fable 5's safety
classifiers decline a request, the harness logs it and records the refusal
rather than crashing.

**Why:** Fable 5 has safety classifiers that can trigger on ~5% of
sessions. For a coding/reasoning benchmark this is unlikely, but the
harness should handle it gracefully.

**Impact on scores:** A refused task scores 0. Documented in results.

## What did NOT change

- All 10 quality task prompts are identical to Round 5.
- All 3 speed task prompts are identical to Round 5.
- All fixture files are byte-identical copies from Round 5.
- Feature-detection regexes are identical.
- Functional test logic (todo app, TypeScript) is identical.
- Composite scoring formula is identical: `(features * 60) + (functional * 40)`.
- Overall ranking weights are identical: Coding 40%, Reasoning 20%, Tool Use 20%, Speed 20%.
- Tasks 2.1 and 2.2 still run but are not factored into the weighted total.
- Task 1.2 still has no functional test (would require starting a FastAPI server).
