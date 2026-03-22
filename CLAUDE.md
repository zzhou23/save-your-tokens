# CLAUDE.md — Save Your Tokens

## What Is This

`save-your-tokens` is an open-source, application-layer context budget management framework for LLM applications. Think of it as **virtual memory management for LLM context windows**.

**Package**: `save-your-tokens` on PyPI
**CLI command**: `syt`
**Language**: Python 3.10+

## Design Decisions (from Q&A)

| # | Decision | Detail |
|---|----------|--------|
| Q1 | Three-layer classification | Hybrid: convention-based defaults + user override |
| Q2 | Budget overage response | Tiered: warn (<5%), auto-compact (5-20%), reject (>20%) |
| Q3 | Staleness detection | Phase 1: pure turn counting. Future: + keyword matching |
| Q4 | Compact ownership | Strategy owns flow, adapter provides optional `model_compact()`. Future: independent Compactor for local models |
| Q5 | Integration interface | Two-layer: inner unified context logic + outer framework glue |
| Q6 | Claude Code integration | Hooks + file management combined |
| Q7 | Skills relationship | Complementary: syt manages budget-aware load/unload, not skill content |
| Q8 | Skill granularity | Format-agnostic text block + metadata |
| Q9 | Benchmark metrics | Token savings + output quality (dual metric) |
| Q10 | `syt compact` target | Default: scan directory. Also supports specified file/stdin |
| Q11 | `syt analyze` input | Default: scan directory. Also supports conversation log |
| Q12 | Target users | Phase 1: power users. Phase 2: lower barrier for general devs |
| Q13 | Competitor relationship | Reference but independent. No dependency on context-engineering-toolkit |

## Code Conventions

- Python 3.10+, `uv` preferred / `pip` compatible
- Type hints required on all public APIs
- Google-style docstrings
- Linting: `ruff format . && ruff check .`
- Type checking: `mypy src/`
- Testing: `pytest`, 80%+ coverage on core/
- Core deps: stdlib + pydantic only. Everything else via optional extras
- CLI: click library
- Data models: pydantic v2
- **Immutability preferred**: create new objects, don't mutate existing ones

## Architecture

```
src/save_your_tokens/
├── core/           # Budget engine, lifecycle, strategy (MUST BUILD)
├── skills/         # Budget-aware skill loading (MUST BUILD)
├── adapters/       # ModelAdapter plugins (claude, openai)
├── integrations/   # Framework glue (claude_code, langchain)
├── reuse/          # Wrappers (tokenizers, compression, observability)
└── cli/            # `syt` commands
```

## Key Commands

```bash
syt init                    # Initialize project config
syt analyze                 # Analyze context files in current dir
syt analyze --log file.json # Analyze conversation log
syt compact                 # Compact context files in current dir
syt compact file.md         # Compact a specific file
syt report                  # Generate usage report
```
