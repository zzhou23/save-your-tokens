# Phase 2: Ecosystem Expansion — Design Spec

**Date**: 2026-03-22
**Status**: Approved
**Approach**: Bottom-Up (adapters → compactor → observability → integrations → benchmark → publish)

## Overview

Phase 2 expands save-your-tokens from a Claude/OpenAI-focused prototype to a multi-model, multi-framework ecosystem. Six workstreams deliver: new model adapters, an independent compactor interface, production observability, framework integrations, real-session benchmarks, and PyPI publish documentation.

## 1. New Model Adapters (DeepSeek + Gemini)

### 1.1 DeepSeek Adapter

**File**: `src/save_your_tokens/adapters/deepseek.py`

- Token counting via `tiktoken` with `cl100k_base` encoding as fallback (DeepSeek's custom BPE tokenizer is not in tiktoken's public model registry, so `encoding_for_model()` won't work directly)
- Context windows: `deepseek-chat` = 64K, `deepseek-coder` = 128K, `deepseek-reasoner` = 64K
- `format_context()`: OpenAI-compatible message format (system/user/assistant)
- `model_compact()`: via DeepSeek API (OpenAI-compatible endpoint, base_url override)
- Optional dep in pyproject.toml: `deepseek = ["openai>=1.0", "tiktoken>=0.7"]`

### 1.2 Gemini Adapter

**File**: `src/save_your_tokens/adapters/gemini.py`

- Token counting via `google-genai` SDK's native `count_tokens()`
- Context windows: `gemini-2.0-flash` = 1M, `gemini-2.5-pro` = 1M
- `format_context()`: Gemini-native `contents` with `parts` structure (not OpenAI message format)
- `model_compact()`: via `generate_content()` API
- Optional dep: `gemini = ["google-genai>=1.0"]`

### 1.3 pyproject.toml Changes

```toml
[project.optional-dependencies]
deepseek = ["openai>=1.0", "tiktoken>=0.7"]
gemini = ["google-genai>=1.0"]
all = ["save-your-tokens[claude,openai,deepseek,gemini,langfuse,langchain]"]
```

### 1.4 Adapter Registry Update

Update `adapters/__init__.py` to expose all four adapters with lazy imports.

## 2. Independent Compactor Interface

### 2.1 Refactor

Move `Compactor` Protocol and `DefaultCompactor` from `core/strategy.py` to `src/save_your_tokens/reuse/compactor.py`. `strategy.py` imports from the new location (backward compatible).

**Compactor vs Compressor relationship**: `reuse/compression.py` has a ratio-based `Compressor` ABC (`compress(text, target_ratio) -> str`). The token-based `Compactor` Protocol (`compact(content, target_tokens) -> str`) is a higher-level interface. `DefaultCompactor` already wraps `ExtractiveCompressor` with a token-to-ratio conversion (`target_tokens * 4 / len(content)`). The same adapter pattern applies to `TruncationCompressor`. Both `Compressor` classes remain in `compression.py` unchanged — they are low-level utilities. `compactor.py` provides the token-aware wrappers.

### 2.2 New Implementations

**File**: `src/save_your_tokens/reuse/compactor.py`

```python
class Compactor(Protocol):
    """Protocol for content compaction (token-based interface)."""
    def compact(self, content: str, target_tokens: int) -> str: ...

class DefaultCompactor:
    """Wraps ExtractiveCompressor with token-to-ratio conversion (existing, moved here)."""

class TruncationCompactor:
    """Wraps TruncationCompressor with token-to-ratio conversion."""

class LLMCompactor:
    """Uses any ModelAdapter for API-based summarization.
    Requires adapter with model_compact() support. Raises NotImplementedError
    if the adapter doesn't support native compaction."""
    def __init__(self, adapter: ModelAdapter) -> None: ...
    def compact(self, content: str, target_tokens: int) -> str:
        # Calls adapter.model_compact(content, target_tokens)
        # Raises NotImplementedError if adapter.supports_native_compact is False

class LocalModelCompactor:
    """Uses a local model (Ollama/vLLM) for compaction. No cloud API needed."""
    def __init__(
        self,
        endpoint: str = "http://localhost:11434",
        model: str = "llama3",
        token_estimator: Callable[[str], int] | None = None,
    ) -> None: ...
    def compact(self, content: str, target_tokens: int) -> str:
        # HTTP POST to local inference server (OpenAI-compatible /v1/chat/completions)
        # Uses token_estimator if provided, else rough char/4 estimate for max_tokens
```

### 2.3 Compactor Factory

```python
def create_compactor(backend: str = "extractive", **kwargs) -> Compactor:
    """Factory for compactor backends.

    Backends:
      - "extractive": DefaultCompactor (wraps ExtractiveCompressor, no deps)
      - "truncation": TruncationCompactor (wraps TruncationCompressor)
      - "llm": LLMCompactor (requires adapter kwarg with model_compact support)
      - "local": LocalModelCompactor (requires endpoint/model kwargs)
    """
```

## 3. Langfuse Observability

### 3.1 Expanded Observer

**File**: `src/save_your_tokens/reuse/observability.py` (existing, expand)

```python
class LangfuseObserver:
    def track_usage(self, event: dict[str, Any]) -> None:
        # Create structured Langfuse generation/span
        # Event types: "budget_check", "compact_action", "skill_load", "turn_complete"

    def track_compaction(
        self, before_tokens: int, after_tokens: int, method: str
    ) -> None:
        # Dedicated compaction event with savings metrics

    def track_budget_warning(self, usage: ContextUsage, threshold: str) -> None:
        # Budget threshold crossing events
```

### 3.2 Lifecycle Wiring

- `LifecycleManager.__init__()` accepts optional `observer: Observer = None` (defaults to `NoOpObserver`)
- `StrategyEngine.__init__()` accepts optional `observer: Observer = None` (defaults to `NoOpObserver`)
- `post_turn()` emits `turn_complete` event with usage snapshot
- `StrategyEngine.execute_action()` emits `compact_action` events via `self._observer`
- Default: `NoOpObserver` (zero overhead when Langfuse not configured)

## 4. Framework Integrations

### 4.1 LangChain Integration (RunnableMiddleware)

**File**: `src/save_your_tokens/integrations/langchain.py`

**Design note**: `FrameworkIntegration` ABC was designed for hook-based integrations (like Claude Code) with `setup()`/`teardown()` lifecycle. LangChain's LCEL model is fundamentally different — it's a composable chain, not a hook system. Rather than force-fitting LCEL into the hook-based ABC, we use a two-class design:

- `LangChainIntegration(FrameworkIntegration)` — implements the abstract methods for users who want the full lifecycle interface
- `SYTRunnable(RunnableSerializable)` — thin LCEL adapter that delegates to `LangChainIntegration` internally

```python
class LangChainIntegration(FrameworkIntegration):
    """Inner layer: implements FrameworkIntegration for LangChain."""
    def __init__(self, budget_engine, lifecycle, strategy, adapter): ...
    def setup(self, config): ...      # No-op (LCEL doesn't need hooks)
    def teardown(self): ...            # No-op
    def intercept_context(self, messages): ...  # Classify, budget check, compact, format
    def on_response(self, response): ...        # Post-turn lifecycle

class SYTRunnable(RunnableSerializable):
    """Outer layer: LCEL-native Runnable wrapping LangChainIntegration."""

    def invoke(self, input, config=None):
        # Delegates to self._integration.intercept_context() + on_response()

    async def ainvoke(self, input, config=None):
        # Async version of the same flow
```

**Usage**:
```python
from save_your_tokens.integrations.langchain import SYTRunnable

chain = SYTRunnable(budget_engine, adapter) | llm | output_parser
result = chain.invoke({"messages": conversation})
```

**Optional dep**: `langchain = ["langchain-core>=0.3"]`

### 4.2 Raw SDK Integration

**File**: `src/save_your_tokens/integrations/raw_sdk.py`

**Design note**: Same two-layer approach. `RawSDKIntegration` extends `FrameworkIntegration`; `SYTWrapper` provides the ergonomic user-facing API.

```python
class RawSDKIntegration(FrameworkIntegration):
    """Inner layer: implements FrameworkIntegration for raw SDK usage."""
    def setup(self, config): ...      # No-op
    def teardown(self): ...            # No-op
    def intercept_context(self, messages): ...  # Classify, budget check, compact, format
    def on_response(self, response): ...        # Post-turn lifecycle

class SYTWrapper:
    """Outer layer: ergonomic wrapper delegating to RawSDKIntegration."""

    def __init__(
        self,
        adapter: ModelAdapter,
        budget_engine: BudgetEngine,
        lifecycle: LifecycleManager,
        strategy: StrategyEngine,
    ) -> None: ...

    def prepare_context(self, messages: list[dict]) -> list[dict]:
        """Classify, budget check, compact, format — returns ready-to-send messages."""

    def on_response(self, response: dict) -> dict:
        """Post-turn lifecycle. Returns usage summary dict."""
```

**Usage**:
```python
from save_your_tokens.integrations.raw_sdk import SYTWrapper

wrapper = SYTWrapper(adapter=ClaudeAdapter(), ...)
messages = wrapper.prepare_context(my_messages)
response = client.messages.create(messages=messages)
summary = wrapper.on_response(response)
```

## 5. Recorded Session Benchmark

### 5.1 Fixture Format

**Directory**: `benchmarks/fixtures/`

```json
{
  "metadata": {
    "model": "claude-sonnet-4-6",
    "recorded_at": "2026-03-22",
    "description": "50-turn coding session"
  },
  "turns": [
    {"role": "user", "content": "...", "token_count": 1234},
    {"role": "assistant", "content": "...", "token_count": 5678}
  ]
}
```

### 5.2 Replay Engine

**File**: `benchmarks/recorded_replay.py`

- Feeds recorded turns through syt's lifecycle one at a time
- Tracks cumulative tokens sent (with syt) vs. baseline (without syt)
- Reports: savings %, compaction events triggered, budget utilization over time
- No live API calls — uses rough char-based token estimation (`len(text) // 4`) for measurement, since real adapters like `ClaudeAdapter.count_tokens()` require API keys. Fixture `token_count` values serve as ground truth for baseline comparison
- Outputs markdown report (same format as Phase 1 synthetic benchmark)

### 5.3 Sample Fixtures

Include 2-3 fixtures:
- `claude_coding_session.json` — 50-turn coding session (synthetic but realistic)
- `qa_session.json` — 20-turn Q&A session
- `long_context_session.json` — 100+ turn session with large file reads

## 6. PyPI Publishing Documentation

### 6.1 Publishing Guide

**File**: `docs/PUBLISHING.md`

Contents:
1. Prerequisites: PyPI account, API token, `hatch` installed
2. Pre-publish checklist: version bump, tests pass, lint clean, wheel builds
3. Build: `hatch build`
4. Upload: `hatch publish` (or `twine upload dist/*`)
5. Verify: `pip install save-your-tokens` from PyPI
6. Version convention: semver, `0.2.0` for Phase 2

### 6.2 Version Bump

`pyproject.toml`: `version = "0.1.0"` → `version = "0.2.0"`

## Testing Strategy

Each workstream includes its own tests:

| Workstream | Test file | Type |
|---|---|---|
| DeepSeek adapter | `tests/test_adapters/test_deepseek.py` | Unit (mocked SDK) |
| Gemini adapter | `tests/test_adapters/test_gemini.py` | Unit (mocked SDK) |
| Compactor refactor | `tests/test_reuse/test_compactor.py` | Unit |
| LLM/Local compactors | `tests/test_reuse/test_compactor.py` | Unit (mocked) |
| Langfuse observer | `tests/test_reuse/test_observability.py` | Unit (mocked) |
| LangChain integration | `tests/test_integrations/test_langchain.py` | Unit + integration |
| Raw SDK wrapper | `tests/test_integrations/test_raw_sdk.py` | Unit |
| Recorded benchmark | `benchmarks/test_recorded_replay.py` | Integration |

**Coverage target**: 80%+ overall (matching Phase 1 standard)

## Implementation Order

1. DeepSeek + Gemini adapters (pure, no integration deps)
2. Compactor refactor + new backends (pure, tests immediately)
3. Langfuse observability expansion (wire into lifecycle)
4. LangChain + Raw SDK integrations (compose all of the above)
5. Recorded session benchmark (validate end-to-end)
6. PyPI docs + version bump (publish-ready)

## Files Summary

**New files (10)**:
- `src/save_your_tokens/adapters/deepseek.py`
- `src/save_your_tokens/adapters/gemini.py`
- `src/save_your_tokens/reuse/compactor.py`
- `src/save_your_tokens/integrations/langchain.py`
- `src/save_your_tokens/integrations/raw_sdk.py`
- `benchmarks/recorded_replay.py`
- `benchmarks/fixtures/claude_coding_session.json`
- `benchmarks/fixtures/qa_session.json`
- `benchmarks/fixtures/long_context_session.json`
- `docs/PUBLISHING.md`

**Modified files (7)**:
- `pyproject.toml` (deps, version bump)
- `src/save_your_tokens/adapters/__init__.py` (registry)
- `src/save_your_tokens/core/strategy.py` (import compactor from new location, add observer param)
- `src/save_your_tokens/reuse/observability.py` (expand LangfuseObserver)
- `src/save_your_tokens/core/lifecycle.py` (wire observer param)
- `tests/` (6+ new test files)
- `docs/progress.md` (update status)
