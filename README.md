# Vibe Coder Benchmarks

Benchmark suites, harnesses, and results from [vibescoder.dev](https://vibescoder.dev).

## Structure

```
suites/       — reusable benchmark definitions (tasks, fixtures, scoring)
harnesses/    — reusable runner scripts (cloud API, local inference)
runs/         — dated benchmark executions with results
reports/      — human-readable summaries linked from blog posts
```

## Runs

| Date | Run | Blog Post |
|------|-----|-----------|
| 2026-06-12 | [Frontier Bakeoff](runs/2026-06-12--coding--frontier-api--v1/) | [Frontier Bakeoff: We Benchmarked Fable 5 Hours Before the Shutdown](https://vibescoder.dev/posts/frontier-bakeoff-we-benchmarked-fable-5-hours-before-the-shutdown) |
| 2026-11-18 | [DGX Spark RDMA Recovery](runs/2026-11-18--inference--local-dgx-spark--v1/) | [report](reports/2026-11-18-dgx-spark-rdma.md) (blog pending) |

## Naming Convention

Runs follow the pattern: `YYYY-MM-DD--<domain>--<scope>--<suite-version>`

- **domain**: what's being tested (`coding`, `agentic`, `inference`, `hardware`)
- **scope**: where/how (`frontier-api`, `local-rtx5090`, `coder-cli`)
- **suite-version**: which task suite version (`v1`, `v2`)

## License

Results and analysis are published on [vibescoder.dev](https://vibescoder.dev). Harness code is provided as-is for transparency.
