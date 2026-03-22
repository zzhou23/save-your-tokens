# Benchmark: Baseline vs syt-Managed Context

## Methodology

We simulate a realistic 50-turn agentic coding session (Claude Code-style) and compare two approaches:

- **Baseline**: Cumulative resend — every turn includes all prior context (how LLMs actually work without context management)
- **Managed**: `save-your-tokens` actively manages context through BudgetEngine + LifecycleManager + StrategyEngine with the `agentic` profile

### Synthetic Dataset

The dataset simulates building a Task CRUD feature in a full-stack project:

| Phase | Turns | Activity | Content Type |
|-------|-------|----------|--------------|
| Setup | 1-5 | System prompt, CLAUDE.md, progress.md, user request | Persistent + Session |
| Exploration | 6-20 | File reads (models, routes, services, tests) | Ephemeral (growing) |
| Implementation | 21-35 | Code edits + test runs after each edit | Ephemeral (churn) |
| Finalization | 36-50 | Frontend work, full test runs, lint, commit | Ephemeral (wind-down) |

72 messages total, ~45,500 tokens of raw content.

### Configuration

- Context window: 200,000 tokens
- Profile: `agentic` (persistent=15%, session=35%, output_reserve=20%)
- Stale max age: 8 turns
- Compact interval: every 10 turns

## Results

| Metric | Baseline | Managed | Delta |
|--------|----------|---------|-------|
| **Cumulative tokens sent** | 701,387 | 592,714 | **-15.5%** |
| **Tokens saved** | — | — | **108,673** |
| **Compaction events** | 0 | 2 | — |
| **Peak utilization** | unbounded | 18.1% | budget-aware |

### Compaction Events

| Turn | Action | Effect |
|------|--------|--------|
| 26 | `drop_stale_ephemeral` | Dropped file reads from turns 6-15 (no longer referenced) |
| 41 | `drop_stale_ephemeral` | Dropped stale implementation artifacts from turns 21-30 |

Both compaction events successfully reduced utilization by ~5-9 percentage points, keeping the context well within budget.

### Per-Turn Utilization

```
Turn  Utilization (managed)
  1   |                                                  |  0.2%
  5   |                                                  |  0.6%
 10   |##                                                |  5.4%
 15   |####                                              |  9.9%
 20   |#####                                             | 10.5%
 25   |#######                                           | 14.7%
 26   |#####                                             | 10.4%  * compaction
 30   |######                                            | 13.6%
 35   |######                                            | 13.9%
 40   |#########                                         | 18.1%
 41   |####                                              |  8.7%  * compaction
 45   |#####                                             | 10.6%
 50   |######                                            | 12.3%
```

Without management, utilization grows monotonically. With `syt`, stale ephemeral blocks are automatically dropped when they haven't been referenced for 8+ turns, keeping utilization bounded.

## Comparison with Prior Work

JetBrains Research (March 2026) demonstrated that hybrid context management achieves **7-11% cost reduction** with 2.6% accuracy improvement on SWE-bench.

Our synthetic benchmark shows **15.5% cumulative token savings** — exceeding the JetBrains baseline. The difference is primarily due to:

1. **Three-layer classification** — persistent/session/ephemeral separation prevents important context from being evicted
2. **Staleness-based eviction** — turn-counting detects unused ephemeral blocks automatically
3. **Tiered budget enforcement** — graduated response prevents budget overruns before they happen

## Limitations

- Synthetic dataset (not a real LLM API session)
- Token counting uses character estimation (÷4), not model-specific tokenizer
- No quality measurement (would require actual LLM responses)
- Single profile tested (agentic); chat and rag profiles may show different savings
- Compaction uses extractive summarization, not LLM-based summarization

## Reproducing

```bash
# Generate dataset
python benchmarks/generate_dataset.py

# Run benchmark
python benchmarks/run_benchmark.py

# Output includes per-turn CSV at benchmarks/data/results.csv
```
