# LLM API Harness

Multi-provider benchmark harness for cloud LLM APIs.

## Supported Providers

- **Anthropic** (Claude models) via `anthropic` Python SDK
- **OpenAI** (GPT models) via `openai` Python SDK

## Auth

Set environment variables — no credentials are hardcoded.

| Provider | Required | Optional |
|----------|----------|----------|
| Anthropic | `ANTHROPIC_API_KEY` or `ANTHROPIC_AUTH_TOKEN` | `ANTHROPIC_BASE_URL` |
| OpenAI | `OPENAI_API_KEY` | `OPENAI_BASE_URL` |

## Usage

```bash
pip install anthropic openai pyyaml
python benchmark.py
```

Configure models in the `MODELS` dict at the top of `benchmark.py`.
