## Current Status

- Completed: Project scaffold with all module interfaces and data models
- Completed: Phase 1A test plan — `docs/superpowers/plans/2026-03-22-phase1a-unit-tests.md` (74 test cases, 7 tasks)
- In progress: None (ready to execute Phase 1A test plan)

## Key Decisions

See [CLAUDE.md](../CLAUDE.md) for the full 13-item design decision table from the Q&A session.

Notable:
- Tiered budget overage response (warn / auto-compact / reject)
- Two-layer integration design (inner unified logic + outer framework glue)
- Strategy engine owns compact flow; adapters provide optional `model_compact()`
- Skills are format-agnostic text blocks with metadata, budget-aware load/unload only

## Known Issues

- `TokenCounter` in `reuse/tokenizers.py` uses char-based estimation for Anthropic models (API call needed for accurate count)
- `DefaultCompactor` in `core/strategy.py` uses naive truncation; needs real extractive summarization
- `cli/main.py` uses `len(content) // 4` token estimation throughout; should use adapter-specific counting

## Next Steps

### Phase 1A: Core Tests & Validation
1. Write unit tests for `core/spec.py` — model construction, budget profiles, ephemeral_pct calculation
2. Write unit tests for `core/budget.py` — allocation, tiered overage, stale block detection
3. Write unit tests for `core/strategy.py` — compact action execution, escalation order
4. Write unit tests for `core/lifecycle.py` — session lifecycle, post-turn evaluation
5. Write unit tests for `skills/loader.py` — load/unload, budget rejection on REJECT overage
6. Write unit tests for `skills/registry.py` — directory scanning, frontmatter parsing, JSON skills

### Phase 1B: Integration & CLI
7. Write integration tests for `integrations/claude_code.py` — file scanning, hooks config generation
8. Write CLI end-to-end tests for `syt init`, `syt analyze`, `syt compact`
9. Replace char-based token estimation with adapter-aware counting in CLI commands
10. Implement real extractive summarization in `DefaultCompactor`

### Phase 1C: Benchmark & Release
11. Design benchmark methodology: define baseline (raw conversation) vs managed (syt-enabled)
12. Run benchmark on a real agentic coding session — measure token savings + output quality
13. Write README.md with benchmark results and quickstart guide
14. Publish to PyPI as 0.1.0-alpha
