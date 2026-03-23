## Current Status

- Completed: Phase 1 — 133 tests, 79% coverage, package built as 0.1.0
- **In Progress: Phase 2 — Ecosystem Expansion (6/8 tasks done)**
  - ✅ Task 1: DeepSeek adapter (cl100k_base tokenizer, OpenAI-compatible API)
  - ✅ Task 2: Gemini adapter (google-genai SDK, contents/parts format) + adapter registry with lazy imports
  - ✅ Task 3: Compactor refactor (extracted to reuse/compactor.py, added LLM/Local/Truncation backends + factory)
  - ✅ Task 4: Observer expansion (track_compaction, track_budget_warning, wired into LifecycleManager + StrategyEngine)
  - ✅ Task 5: Raw SDK integration (RawSDKIntegration + SYTWrapper)
  - ✅ Task 6: LangChain integration (LangChainIntegration + SYTRunnable LCEL adapter)
  - 🔧 Task 7: Recorded session benchmark — PARTIALLY DONE (qa_session fixture created, replay engine + other fixtures + tests still needed)
  - ⬜ Task 8: PyPI docs + version bump to 0.2.0
- Total: ~210 tests passing, ~85% coverage

## Key Decisions

See [CLAUDE.md](../CLAUDE.md) for the full 13-item design decision table from the Q&A session.

Phase 2 additions:
- DeepSeek uses cl100k_base fallback (not in tiktoken model registry)
- Gemini format_context uses [System Instructions] prefix in user role (not "system" role in contents)
- Compactor vs Compressor: Compactor is token-based (higher level), Compressor is ratio-based (low level)
- LLMCompactor requires adapter.supports_native_compact, raises NotImplementedError if not
- LangChain/Raw SDK follow two-layer FrameworkIntegration pattern (inner ABC + outer ergonomic wrapper)
- Observer wired with lazy NoOpObserver default (zero overhead)

## Known Issues

- `TokenCounter` in `reuse/tokenizers.py` uses char-based estimation for Anthropic models
- Benchmark Task 7 timed out mid-execution — needs completion (replay engine, 2 more fixtures, tests)
- `benchmarks/` dir has partial uncommitted files from timed-out subagent

## Next Steps

1. Complete Task 7: recorded session benchmark (replay engine + fixtures + tests)
2. Complete Task 8: PyPI docs + version bump to 0.2.0
3. Final full test suite run + coverage check (target 80%+)
