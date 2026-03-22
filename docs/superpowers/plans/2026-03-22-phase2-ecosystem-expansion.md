# Phase 2: Ecosystem Expansion Implementation Plan

> **For agentic workers:** REQUIRED: Use superpowers:subagent-driven-development (if subagents available) or superpowers:executing-plans to implement this plan. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Expand save-your-tokens from Claude/OpenAI prototype to multi-model, multi-framework ecosystem with production observability, real-session benchmarks, and PyPI readiness.

**Architecture:** Bottom-up build order — pure adapters first, then compactor refactor, observability wiring, framework integrations, benchmarks, and finally publish docs. Each layer builds on the previous one. All new code follows existing patterns: ABC/Protocol-based interfaces, pydantic models, optional deps via extras.

**Tech Stack:** Python 3.10+, pydantic v2, click, tiktoken, google-genai, openai SDK, langchain-core, langfuse, pytest, ruff, hatch

**Spec:** `docs/superpowers/specs/2026-03-22-phase2-ecosystem-expansion-design.md`

---

## Chunk 1: Model Adapters (DeepSeek + Gemini)

### Task 1: DeepSeek Adapter

**Files:**
- Create: `src/save_your_tokens/adapters/deepseek.py`
- Create: `tests/test_adapters/test_deepseek.py`
- Modify: `src/save_your_tokens/adapters/__init__.py`
- Modify: `pyproject.toml`

**Context:** Look at `src/save_your_tokens/adapters/openai.py` as the reference pattern. DeepSeek is OpenAI-compatible but uses `base_url` override and `cl100k_base` encoding fallback.

- [ ] **Step 1: Add deepseek optional dep to pyproject.toml**

In `pyproject.toml`, add under `[project.optional-dependencies]`:
```toml
deepseek = ["openai>=1.0", "tiktoken>=0.7"]
```
Update `all` to include deepseek:
```toml
all = ["save-your-tokens[claude,openai,deepseek,gemini,langfuse]"]
```

- [ ] **Step 2: Write failing tests for DeepSeek adapter**

Create `tests/test_adapters/__init__.py` (empty) if it doesn't exist.

Create `tests/test_adapters/test_deepseek.py`:

```python
"""Tests for DeepSeek model adapter."""

from unittest.mock import MagicMock, patch

import pytest

from save_your_tokens.core.spec import ContextBlock, ContextLayer


class TestDeepSeekAdapterProperties:
    def test_model_name_default(self):
        from save_your_tokens.adapters.deepseek import DeepSeekAdapter

        adapter = DeepSeekAdapter()
        assert adapter.model_name == "deepseek-chat"

    def test_model_name_custom(self):
        from save_your_tokens.adapters.deepseek import DeepSeekAdapter

        adapter = DeepSeekAdapter(model="deepseek-coder")
        assert adapter.model_name == "deepseek-coder"

    def test_context_window_chat(self):
        from save_your_tokens.adapters.deepseek import DeepSeekAdapter

        adapter = DeepSeekAdapter(model="deepseek-chat")
        assert adapter.context_window == 64_000

    def test_context_window_coder(self):
        from save_your_tokens.adapters.deepseek import DeepSeekAdapter

        adapter = DeepSeekAdapter(model="deepseek-coder")
        assert adapter.context_window == 128_000

    def test_context_window_reasoner(self):
        from save_your_tokens.adapters.deepseek import DeepSeekAdapter

        adapter = DeepSeekAdapter(model="deepseek-reasoner")
        assert adapter.context_window == 64_000

    def test_context_window_unknown_defaults(self):
        from save_your_tokens.adapters.deepseek import DeepSeekAdapter

        adapter = DeepSeekAdapter(model="deepseek-future")
        assert adapter.context_window == 64_000


class TestDeepSeekTokenCounting:
    @patch("save_your_tokens.adapters.deepseek.tiktoken")
    def test_count_tokens_uses_cl100k(self, mock_tiktoken):
        from save_your_tokens.adapters.deepseek import DeepSeekAdapter

        mock_encoding = MagicMock()
        mock_encoding.encode.return_value = [1, 2, 3, 4, 5]
        mock_tiktoken.get_encoding.return_value = mock_encoding

        adapter = DeepSeekAdapter()
        result = adapter.count_tokens("hello world")

        mock_tiktoken.get_encoding.assert_called_once_with("cl100k_base")
        mock_encoding.encode.assert_called_once_with("hello world")
        assert result == 5

    def test_count_tokens_import_error(self):
        from save_your_tokens.adapters.deepseek import DeepSeekAdapter

        adapter = DeepSeekAdapter()
        adapter._encoding = None
        with patch.dict("sys.modules", {"tiktoken": None}):
            with pytest.raises(ImportError, match="tiktoken"):
                adapter.count_tokens("hello")


class TestDeepSeekFormatContext:
    def test_format_persistent_as_system(self):
        from save_your_tokens.adapters.deepseek import DeepSeekAdapter

        adapter = DeepSeekAdapter()
        persistent = [ContextBlock(id="p1", layer=ContextLayer.PERSISTENT, content="rules")]
        result = adapter.format_context(persistent, [], [])
        assert result[0]["role"] == "system"
        assert result[0]["content"] == "rules"

    def test_format_session_as_system(self):
        from save_your_tokens.adapters.deepseek import DeepSeekAdapter

        adapter = DeepSeekAdapter()
        session = [ContextBlock(id="s1", layer=ContextLayer.SESSION, content="session data")]
        result = adapter.format_context([], session, [])
        assert result[0]["role"] == "system"
        assert "[Session Context]" in result[0]["content"]

    def test_format_ephemeral_preserves_role(self):
        from save_your_tokens.adapters.deepseek import DeepSeekAdapter

        adapter = DeepSeekAdapter()
        ephemeral = [
            ContextBlock(
                id="e1", layer=ContextLayer.EPHEMERAL, content="hi", metadata={"role": "user"}
            ),
            ContextBlock(
                id="e2",
                layer=ContextLayer.EPHEMERAL,
                content="hello",
                metadata={"role": "assistant"},
            ),
        ]
        result = adapter.format_context([], [], ephemeral)
        assert result[0]["role"] == "user"
        assert result[1]["role"] == "assistant"

    def test_format_all_layers(self):
        from save_your_tokens.adapters.deepseek import DeepSeekAdapter

        adapter = DeepSeekAdapter()
        p = [ContextBlock(id="p1", layer=ContextLayer.PERSISTENT, content="rules")]
        s = [ContextBlock(id="s1", layer=ContextLayer.SESSION, content="ctx")]
        e = [ContextBlock(id="e1", layer=ContextLayer.EPHEMERAL, content="msg")]
        result = adapter.format_context(p, s, e)
        assert len(result) == 3


class TestDeepSeekModelCompact:
    @patch("save_your_tokens.adapters.deepseek.OpenAI")
    def test_model_compact_calls_api(self, mock_openai_cls):
        from save_your_tokens.adapters.deepseek import DeepSeekAdapter

        mock_client = MagicMock()
        mock_response = MagicMock()
        mock_response.choices = [MagicMock()]
        mock_response.choices[0].message.content = "summarized"
        mock_client.chat.completions.create.return_value = mock_response
        mock_openai_cls.return_value = mock_client

        adapter = DeepSeekAdapter()
        result = adapter.model_compact("long content", 100)
        assert result == "summarized"
        mock_openai_cls.assert_called_once()

    def test_supports_native_compact(self):
        from save_your_tokens.adapters.deepseek import DeepSeekAdapter

        adapter = DeepSeekAdapter()
        assert adapter.supports_native_compact is True
```

- [ ] **Step 3: Run tests to verify they fail**

Run: `pytest tests/test_adapters/test_deepseek.py -v`
Expected: FAIL (ModuleNotFoundError: No module named 'save_your_tokens.adapters.deepseek')

- [ ] **Step 4: Implement DeepSeek adapter**

Create `src/save_your_tokens/adapters/deepseek.py`:

```python
"""DeepSeek model adapter.

Uses tiktoken with cl100k_base encoding (DeepSeek's custom BPE tokenizer is not
in tiktoken's public model registry). DeepSeek API is OpenAI-compatible.
"""

from __future__ import annotations

from typing import Any

from save_your_tokens.adapters.base import ModelAdapter
from save_your_tokens.core.spec import ContextBlock

DEEPSEEK_CONTEXT_WINDOWS: dict[str, int] = {
    "deepseek-chat": 64_000,
    "deepseek-coder": 128_000,
    "deepseek-reasoner": 64_000,
}

DEEPSEEK_BASE_URL = "https://api.deepseek.com"


class DeepSeekAdapter(ModelAdapter):
    """Adapter for DeepSeek models (OpenAI-compatible API)."""

    def __init__(self, model: str = "deepseek-chat", base_url: str = DEEPSEEK_BASE_URL) -> None:
        self._model = model
        self._base_url = base_url
        self._context_window = DEEPSEEK_CONTEXT_WINDOWS.get(model, 64_000)
        self._encoding: Any = None
        self._client: Any = None

    def _get_encoding(self) -> Any:
        if self._encoding is None:
            try:
                import tiktoken
                self._encoding = tiktoken.get_encoding("cl100k_base")
            except ImportError as e:
                raise ImportError(
                    "tiktoken package required. Install with: pip install save-your-tokens[deepseek]"
                ) from e
        return self._encoding

    def _get_client(self) -> Any:
        if self._client is None:
            try:
                from openai import OpenAI
                self._client = OpenAI(base_url=self._base_url)
            except ImportError as e:
                raise ImportError(
                    "openai package required. Install with: pip install save-your-tokens[deepseek]"
                ) from e
        return self._client

    @property
    def model_name(self) -> str:
        return self._model

    @property
    def context_window(self) -> int:
        return self._context_window

    def count_tokens(self, text: str) -> int:
        """Count tokens using tiktoken cl100k_base encoding."""
        encoding = self._get_encoding()
        return len(encoding.encode(text))

    def format_context(
        self,
        persistent: list[ContextBlock],
        session: list[ContextBlock],
        ephemeral: list[ContextBlock],
    ) -> list[dict[str, Any]]:
        """Format context for DeepSeek's OpenAI-compatible API."""
        messages: list[dict[str, Any]] = []

        persistent_text = "\n\n".join(b.content for b in persistent if b.content)
        if persistent_text:
            messages.append({"role": "system", "content": persistent_text})

        session_text = "\n\n".join(b.content for b in session if b.content)
        if session_text:
            messages.append({"role": "system", "content": f"[Session Context]\n{session_text}"})

        for block in ephemeral:
            role = block.metadata.get("role", "user")
            messages.append({"role": role, "content": block.content})

        return messages

    def model_compact(self, content: str, target_tokens: int) -> str | None:
        """Use DeepSeek API for content compaction."""
        client = self._get_client()
        response = client.chat.completions.create(
            model=self._model,
            max_tokens=target_tokens,
            messages=[
                {
                    "role": "user",
                    "content": (
                        f"Summarize the following content concisely, "
                        f"preserving key information, in under {target_tokens} tokens:\n\n{content}"
                    ),
                }
            ],
        )
        return response.choices[0].message.content
```

- [ ] **Step 5: Run tests to verify they pass**

Run: `pytest tests/test_adapters/test_deepseek.py -v`
Expected: All PASS

- [ ] **Step 6: Lint check**

Run: `ruff format src/save_your_tokens/adapters/deepseek.py tests/test_adapters/test_deepseek.py && ruff check src/save_your_tokens/adapters/deepseek.py tests/test_adapters/test_deepseek.py`

- [ ] **Step 7: Commit**

```bash
git add src/save_your_tokens/adapters/deepseek.py tests/test_adapters/ pyproject.toml
git commit -m "feat: add DeepSeek model adapter with cl100k_base tokenizer"
```

---

### Task 2: Gemini Adapter

**Files:**
- Create: `src/save_your_tokens/adapters/gemini.py`
- Create: `tests/test_adapters/test_gemini.py`

**Context:** Different from OpenAI pattern. Gemini uses `contents` with `parts` structure, and `google-genai` SDK for token counting.

- [ ] **Step 1: Add gemini optional dep to pyproject.toml**

In `pyproject.toml`, add under `[project.optional-dependencies]`:
```toml
gemini = ["google-genai>=1.0"]
```
Update `all`:
```toml
all = ["save-your-tokens[claude,openai,deepseek,gemini,langfuse]"]
```

- [ ] **Step 2: Write failing tests for Gemini adapter**

Create `tests/test_adapters/test_gemini.py`:

```python
"""Tests for Gemini model adapter."""

from unittest.mock import MagicMock, patch

import pytest

from save_your_tokens.core.spec import ContextBlock, ContextLayer


class TestGeminiAdapterProperties:
    def test_model_name_default(self):
        from save_your_tokens.adapters.gemini import GeminiAdapter

        adapter = GeminiAdapter()
        assert adapter.model_name == "gemini-2.0-flash"

    def test_model_name_custom(self):
        from save_your_tokens.adapters.gemini import GeminiAdapter

        adapter = GeminiAdapter(model="gemini-2.5-pro")
        assert adapter.model_name == "gemini-2.5-pro"

    def test_context_window_flash(self):
        from save_your_tokens.adapters.gemini import GeminiAdapter

        adapter = GeminiAdapter(model="gemini-2.0-flash")
        assert adapter.context_window == 1_000_000

    def test_context_window_pro(self):
        from save_your_tokens.adapters.gemini import GeminiAdapter

        adapter = GeminiAdapter(model="gemini-2.5-pro")
        assert adapter.context_window == 1_000_000

    def test_context_window_unknown_defaults(self):
        from save_your_tokens.adapters.gemini import GeminiAdapter

        adapter = GeminiAdapter(model="gemini-future")
        assert adapter.context_window == 1_000_000


class TestGeminiTokenCounting:
    @patch("save_your_tokens.adapters.gemini.genai")
    def test_count_tokens_uses_sdk(self, mock_genai):
        from save_your_tokens.adapters.gemini import GeminiAdapter

        mock_client = MagicMock()
        mock_result = MagicMock()
        mock_result.total_tokens = 42
        mock_client.models.count_tokens.return_value = mock_result
        mock_genai.Client.return_value = mock_client

        adapter = GeminiAdapter()
        result = adapter.count_tokens("hello world")

        assert result == 42
        mock_client.models.count_tokens.assert_called_once()

    def test_count_tokens_import_error(self):
        from save_your_tokens.adapters.gemini import GeminiAdapter

        adapter = GeminiAdapter()
        adapter._client = None
        with patch.dict("sys.modules", {"google": None, "google.genai": None}):
            with pytest.raises(ImportError, match="google-genai"):
                adapter.count_tokens("hello")


class TestGeminiFormatContext:
    def test_format_persistent_as_system_instruction(self):
        from save_your_tokens.adapters.gemini import GeminiAdapter

        adapter = GeminiAdapter()
        persistent = [ContextBlock(id="p1", layer=ContextLayer.PERSISTENT, content="rules")]
        result = adapter.format_context(persistent, [], [])
        assert result[0]["role"] == "system"
        assert result[0]["parts"][0]["text"] == "rules"

    def test_format_session_as_user_context(self):
        from save_your_tokens.adapters.gemini import GeminiAdapter

        adapter = GeminiAdapter()
        session = [ContextBlock(id="s1", layer=ContextLayer.SESSION, content="session data")]
        result = adapter.format_context([], session, [])
        assert result[0]["role"] == "user"
        assert "[Session Context]" in result[0]["parts"][0]["text"]

    def test_format_ephemeral_with_parts(self):
        from save_your_tokens.adapters.gemini import GeminiAdapter

        adapter = GeminiAdapter()
        ephemeral = [
            ContextBlock(
                id="e1", layer=ContextLayer.EPHEMERAL, content="hi", metadata={"role": "user"}
            ),
        ]
        result = adapter.format_context([], [], ephemeral)
        assert result[0]["role"] == "user"
        assert result[0]["parts"][0]["text"] == "hi"

    def test_format_all_layers(self):
        from save_your_tokens.adapters.gemini import GeminiAdapter

        adapter = GeminiAdapter()
        p = [ContextBlock(id="p1", layer=ContextLayer.PERSISTENT, content="rules")]
        s = [ContextBlock(id="s1", layer=ContextLayer.SESSION, content="ctx")]
        e = [ContextBlock(id="e1", layer=ContextLayer.EPHEMERAL, content="msg")]
        result = adapter.format_context(p, s, e)
        assert len(result) == 3


class TestGeminiModelCompact:
    @patch("save_your_tokens.adapters.gemini.genai")
    def test_model_compact_calls_api(self, mock_genai):
        from save_your_tokens.adapters.gemini import GeminiAdapter

        mock_client = MagicMock()
        mock_response = MagicMock()
        mock_response.text = "summarized"
        mock_client.models.generate_content.return_value = mock_response
        mock_genai.Client.return_value = mock_client

        adapter = GeminiAdapter()
        result = adapter.model_compact("long content", 100)
        assert result == "summarized"

    def test_supports_native_compact(self):
        from save_your_tokens.adapters.gemini import GeminiAdapter

        adapter = GeminiAdapter()
        assert adapter.supports_native_compact is True
```

- [ ] **Step 3: Run tests to verify they fail**

Run: `pytest tests/test_adapters/test_gemini.py -v`
Expected: FAIL

- [ ] **Step 4: Implement Gemini adapter**

Create `src/save_your_tokens/adapters/gemini.py`:

```python
"""Gemini model adapter.

Uses google-genai SDK for token counting and content generation.
Gemini uses a contents/parts message structure, not OpenAI-style messages.
"""

from __future__ import annotations

from typing import Any

from save_your_tokens.adapters.base import ModelAdapter
from save_your_tokens.core.spec import ContextBlock

try:
    from google import genai
except ImportError:
    genai = None  # type: ignore[assignment]

GEMINI_CONTEXT_WINDOWS: dict[str, int] = {
    "gemini-2.0-flash": 1_000_000,
    "gemini-2.5-pro": 1_000_000,
}


class GeminiAdapter(ModelAdapter):
    """Adapter for Google Gemini models."""

    def __init__(self, model: str = "gemini-2.0-flash") -> None:
        self._model = model
        self._context_window = GEMINI_CONTEXT_WINDOWS.get(model, 1_000_000)
        self._client: Any = None

    def _get_client(self) -> Any:
        if self._client is None:
            if genai is None:
                raise ImportError(
                    "google-genai package required. Install with: "
                    "pip install save-your-tokens[gemini]"
                )
            self._client = genai.Client()
        return self._client

    @property
    def model_name(self) -> str:
        return self._model

    @property
    def context_window(self) -> int:
        return self._context_window

    @property
    def recommended_output_reserve(self) -> int:
        """Gemini has large windows; 10% reserve is sufficient."""
        return int(self.context_window * 0.1)

    def count_tokens(self, text: str) -> int:
        """Count tokens using Gemini's native token counting API."""
        client = self._get_client()
        result = client.models.count_tokens(
            model=self._model,
            contents=text,
        )
        return result.total_tokens

    def format_context(
        self,
        persistent: list[ContextBlock],
        session: list[ContextBlock],
        ephemeral: list[ContextBlock],
    ) -> list[dict[str, Any]]:
        """Format context for Gemini's contents/parts structure.

        Persistent -> system instruction (role: system)
        Session -> user context message
        Ephemeral -> user/model messages with parts
        """
        contents: list[dict[str, Any]] = []

        persistent_text = "\n\n".join(b.content for b in persistent if b.content)
        if persistent_text:
            contents.append({
                "role": "system",
                "parts": [{"text": persistent_text}],
            })

        session_text = "\n\n".join(b.content for b in session if b.content)
        if session_text:
            contents.append({
                "role": "user",
                "parts": [{"text": f"[Session Context]\n{session_text}"}],
            })

        for block in ephemeral:
            role = block.metadata.get("role", "user")
            # Gemini uses "model" instead of "assistant"
            if role == "assistant":
                role = "model"
            contents.append({
                "role": role,
                "parts": [{"text": block.content}],
            })

        return contents

    def model_compact(self, content: str, target_tokens: int) -> str | None:
        """Use Gemini API to summarize content."""
        client = self._get_client()
        response = client.models.generate_content(
            model=self._model,
            contents=(
                f"Summarize the following content concisely, "
                f"preserving key information, in under {target_tokens} tokens:\n\n{content}"
            ),
            config={"max_output_tokens": target_tokens},
        )
        return response.text
```

- [ ] **Step 5: Run tests to verify they pass**

Run: `pytest tests/test_adapters/test_gemini.py -v`
Expected: All PASS

- [ ] **Step 6: Update adapter registry**

Edit `src/save_your_tokens/adapters/__init__.py`:

```python
"""Model adapters: token counting, context formatting, optional compaction."""

from save_your_tokens.adapters.base import ModelAdapter

__all__ = [
    "ModelAdapter",
    "ClaudeAdapter",
    "OpenAIAdapter",
    "DeepSeekAdapter",
    "GeminiAdapter",
]


def __getattr__(name: str):
    """Lazy imports to avoid requiring all SDKs."""
    if name == "ClaudeAdapter":
        from save_your_tokens.adapters.claude import ClaudeAdapter
        return ClaudeAdapter
    if name == "OpenAIAdapter":
        from save_your_tokens.adapters.openai import OpenAIAdapter
        return OpenAIAdapter
    if name == "DeepSeekAdapter":
        from save_your_tokens.adapters.deepseek import DeepSeekAdapter
        return DeepSeekAdapter
    if name == "GeminiAdapter":
        from save_your_tokens.adapters.gemini import GeminiAdapter
        return GeminiAdapter
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
```

- [ ] **Step 7: Lint and commit**

Run: `ruff format src/save_your_tokens/adapters/ tests/test_adapters/ && ruff check src/save_your_tokens/adapters/ tests/test_adapters/`

```bash
git add src/save_your_tokens/adapters/gemini.py tests/test_adapters/test_gemini.py src/save_your_tokens/adapters/__init__.py pyproject.toml
git commit -m "feat: add Gemini adapter and update adapter registry with lazy imports"
```

---

## Chunk 2: Independent Compactor Interface

### Task 3: Compactor Refactor + New Backends

**Files:**
- Create: `src/save_your_tokens/reuse/compactor.py`
- Create: `tests/test_reuse/__init__.py`
- Create: `tests/test_reuse/test_compactor.py`
- Modify: `src/save_your_tokens/core/strategy.py` (import from new location)

**Context:** Currently `Compactor` Protocol and `DefaultCompactor` live in `core/strategy.py`. We move them to `reuse/compactor.py` and add `TruncationCompactor`, `LLMCompactor`, `LocalModelCompactor`, and `create_compactor()` factory. The `Compressor` classes in `reuse/compression.py` remain unchanged — `Compactor` wraps them with token-to-ratio conversion.

- [ ] **Step 1: Write failing tests for compactor module**

Create `tests/test_reuse/__init__.py` (empty).

Create `tests/test_reuse/test_compactor.py`:

```python
"""Tests for save_your_tokens.reuse.compactor — token-based compaction interface."""

from unittest.mock import MagicMock, patch

import pytest


class TestDefaultCompactor:
    def test_short_content_unchanged(self):
        from save_your_tokens.reuse.compactor import DefaultCompactor

        compactor = DefaultCompactor()
        result = compactor.compact("short text", target_tokens=1000)
        assert result == "short text"

    def test_long_content_compressed(self):
        from save_your_tokens.reuse.compactor import DefaultCompactor

        compactor = DefaultCompactor()
        long_text = "This is a long sentence. " * 100
        result = compactor.compact(long_text, target_tokens=50)
        assert len(result) < len(long_text)


class TestTruncationCompactor:
    def test_short_content_unchanged(self):
        from save_your_tokens.reuse.compactor import TruncationCompactor

        compactor = TruncationCompactor()
        result = compactor.compact("short", target_tokens=1000)
        assert result == "short"

    def test_long_content_truncated(self):
        from save_your_tokens.reuse.compactor import TruncationCompactor

        compactor = TruncationCompactor()
        long_text = "x" * 10000
        result = compactor.compact(long_text, target_tokens=100)
        assert len(result) <= 100 * 4 + 50  # Allow for truncation marker
        assert "[... truncated ...]" in result


class TestLLMCompactor:
    def test_compact_with_supporting_adapter(self):
        from save_your_tokens.reuse.compactor import LLMCompactor

        mock_adapter = MagicMock()
        mock_adapter.supports_native_compact = True
        mock_adapter.model_compact.return_value = "summarized"

        compactor = LLMCompactor(adapter=mock_adapter)
        result = compactor.compact("long content here", target_tokens=50)

        assert result == "summarized"
        mock_adapter.model_compact.assert_called_once_with("long content here", 50)

    def test_compact_raises_if_adapter_unsupported(self):
        from save_your_tokens.reuse.compactor import LLMCompactor

        mock_adapter = MagicMock()
        mock_adapter.supports_native_compact = False

        compactor = LLMCompactor(adapter=mock_adapter)
        with pytest.raises(NotImplementedError, match="does not support native compaction"):
            compactor.compact("content", target_tokens=50)


class TestLocalModelCompactor:
    @patch("save_your_tokens.reuse.compactor.httpx")
    def test_compact_calls_local_endpoint(self, mock_httpx):
        from save_your_tokens.reuse.compactor import LocalModelCompactor

        mock_response = MagicMock()
        mock_response.json.return_value = {
            "choices": [{"message": {"content": "compact result"}}]
        }
        mock_response.raise_for_status = MagicMock()
        mock_httpx.post.return_value = mock_response

        compactor = LocalModelCompactor(endpoint="http://localhost:11434", model="llama3")
        result = compactor.compact("long text", target_tokens=100)

        assert result == "compact result"
        mock_httpx.post.assert_called_once()

    def test_custom_token_estimator_stored(self):
        from save_your_tokens.reuse.compactor import LocalModelCompactor

        estimator = lambda text: len(text) // 3
        compactor = LocalModelCompactor(
            endpoint="http://localhost:11434",
            model="llama3",
            token_estimator=estimator,
        )
        assert compactor._token_estimator is estimator
        assert compactor._token_estimator("hello world") == 3


class TestCreateCompactor:
    def test_create_extractive(self):
        from save_your_tokens.reuse.compactor import DefaultCompactor, create_compactor

        compactor = create_compactor("extractive")
        assert isinstance(compactor, DefaultCompactor)

    def test_create_truncation(self):
        from save_your_tokens.reuse.compactor import TruncationCompactor, create_compactor

        compactor = create_compactor("truncation")
        assert isinstance(compactor, TruncationCompactor)

    def test_create_llm(self):
        from save_your_tokens.reuse.compactor import LLMCompactor, create_compactor

        mock_adapter = MagicMock()
        compactor = create_compactor("llm", adapter=mock_adapter)
        assert isinstance(compactor, LLMCompactor)

    def test_create_local(self):
        from save_your_tokens.reuse.compactor import LocalModelCompactor, create_compactor

        compactor = create_compactor("local", endpoint="http://localhost:11434", model="llama3")
        assert isinstance(compactor, LocalModelCompactor)

    def test_create_unknown_raises(self):
        from save_your_tokens.reuse.compactor import create_compactor

        with pytest.raises(ValueError, match="Unknown compactor backend"):
            create_compactor("unknown")
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/test_reuse/test_compactor.py -v`
Expected: FAIL

- [ ] **Step 3: Add httpx optional dep to pyproject.toml**

In `pyproject.toml`, add under `[project.optional-dependencies]`:
```toml
local = ["httpx>=0.27"]
```
Update `all`:
```toml
all = ["save-your-tokens[claude,openai,deepseek,gemini,langfuse,langchain,local]"]
```

- [ ] **Step 4: Implement compactor module**

Create `src/save_your_tokens/reuse/compactor.py`:

```python
"""Token-based compaction interface.

Higher-level than compression.py's ratio-based Compressor ABC.
Compactors accept target_tokens and handle the token-to-ratio conversion internally.

Backends:
- DefaultCompactor: wraps ExtractiveCompressor (no deps)
- TruncationCompactor: wraps TruncationCompressor
- LLMCompactor: uses ModelAdapter.model_compact() (requires API)
- LocalModelCompactor: uses local model via HTTP (Ollama/vLLM)
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Callable, Protocol

if TYPE_CHECKING:
    from save_your_tokens.adapters.base import ModelAdapter


class Compactor(Protocol):
    """Protocol for token-based content compaction."""

    def compact(self, content: str, target_tokens: int) -> str: ...


class DefaultCompactor:
    """Wraps ExtractiveCompressor with token-to-ratio conversion."""

    def __init__(self) -> None:
        from save_your_tokens.reuse.compression import ExtractiveCompressor

        self._compressor = ExtractiveCompressor()

    def compact(self, content: str, target_tokens: int) -> str:
        target_chars = target_tokens * 4
        if len(content) <= target_chars:
            return content
        target_ratio = target_chars / len(content)
        return self._compressor.compress(content, target_ratio=target_ratio)


class TruncationCompactor:
    """Wraps TruncationCompressor with token-to-ratio conversion."""

    def __init__(self) -> None:
        from save_your_tokens.reuse.compression import TruncationCompressor

        self._compressor = TruncationCompressor()

    def compact(self, content: str, target_tokens: int) -> str:
        target_chars = target_tokens * 4
        if len(content) <= target_chars:
            return content
        target_ratio = target_chars / len(content)
        return self._compressor.compress(content, target_ratio=target_ratio)


class LLMCompactor:
    """Uses a ModelAdapter's native compaction for API-based summarization.

    Requires an adapter that supports model_compact(). Raises NotImplementedError
    if the adapter does not support native compaction.
    """

    def __init__(self, adapter: ModelAdapter) -> None:
        self._adapter = adapter

    def compact(self, content: str, target_tokens: int) -> str:
        if not self._adapter.supports_native_compact:
            raise NotImplementedError(
                f"Adapter {self._adapter.model_name!r} does not support native compaction. "
                "Use DefaultCompactor or LocalModelCompactor instead."
            )
        result = self._adapter.model_compact(content, target_tokens)
        if result is None:
            raise RuntimeError("Adapter model_compact returned None unexpectedly")
        return result


class LocalModelCompactor:
    """Uses a local model (Ollama/vLLM) for compaction via OpenAI-compatible API.

    No cloud API keys needed. Sends HTTP requests to a local inference server.
    """

    def __init__(
        self,
        endpoint: str = "http://localhost:11434",
        model: str = "llama3",
        token_estimator: Callable[[str], int] | None = None,
    ) -> None:
        self._endpoint = endpoint.rstrip("/")
        self._model = model
        self._token_estimator = token_estimator or (lambda text: len(text) // 4)

    def compact(self, content: str, target_tokens: int) -> str:
        import httpx

        response = httpx.post(
            f"{self._endpoint}/v1/chat/completions",
            json={
                "model": self._model,
                "max_tokens": target_tokens,
                "messages": [
                    {
                        "role": "user",
                        "content": (
                            f"Summarize the following content concisely, "
                            f"preserving key information, in under "
                            f"{target_tokens} tokens:\n\n{content}"
                        ),
                    }
                ],
            },
            timeout=60.0,
        )
        response.raise_for_status()
        data: dict[str, Any] = response.json()
        return data["choices"][0]["message"]["content"]


def create_compactor(backend: str = "extractive", **kwargs: Any) -> Compactor:
    """Factory for compactor backends.

    Args:
        backend: One of "extractive", "truncation", "llm", "local".
        **kwargs: Backend-specific configuration.
            - "llm": requires `adapter` (ModelAdapter with model_compact support)
            - "local": accepts `endpoint`, `model`, `token_estimator`

    Returns:
        A Compactor instance.

    Raises:
        ValueError: If backend is unknown.
    """
    match backend:
        case "extractive":
            return DefaultCompactor()
        case "truncation":
            return TruncationCompactor()
        case "llm":
            return LLMCompactor(adapter=kwargs["adapter"])
        case "local":
            return LocalModelCompactor(
                endpoint=kwargs.get("endpoint", "http://localhost:11434"),
                model=kwargs.get("model", "llama3"),
                token_estimator=kwargs.get("token_estimator"),
            )
        case _:
            raise ValueError(f"Unknown compactor backend: {backend!r}")
```

- [ ] **Step 5: Run tests to verify they pass**

Run: `pytest tests/test_reuse/test_compactor.py -v`
Expected: All PASS

- [ ] **Step 6: Update strategy.py to import from new location**

Edit `src/save_your_tokens/core/strategy.py`:

**Remove these lines** (the `Compactor` Protocol and `DefaultCompactor` class, approximately lines 22-50):
```python
class Compactor(Protocol):
    ...
class DefaultCompactor:
    ...
```

**Add this import** after the existing imports at the top of the file:
```python
from save_your_tokens.reuse.compactor import Compactor, DefaultCompactor
```

Also remove `Protocol` from the `typing` import if it's no longer needed locally. Keep the `StrategyEngine` class unchanged.

- [ ] **Step 7: Run full test suite to verify backward compatibility**

Run: `pytest -v`
Expected: All existing tests still PASS

- [ ] **Step 8: Lint and commit**

```bash
ruff format src/save_your_tokens/reuse/compactor.py src/save_your_tokens/core/strategy.py tests/test_reuse/ && ruff check src/save_your_tokens/reuse/ src/save_your_tokens/core/strategy.py tests/test_reuse/
git add src/save_your_tokens/reuse/compactor.py tests/test_reuse/ src/save_your_tokens/core/strategy.py pyproject.toml
git commit -m "refactor: extract Compactor to reuse/compactor.py, add LLM and local backends"
```

---

## Chunk 3: Langfuse Observability

### Task 4: Expand Observer and Wire into Lifecycle

**Files:**
- Modify: `src/save_your_tokens/reuse/observability.py`
- Modify: `src/save_your_tokens/core/lifecycle.py`
- Modify: `src/save_your_tokens/core/strategy.py`
- Create: `tests/test_reuse/test_observability.py`

**Context:** `observability.py` already has `Observer` Protocol, `NoOpObserver`, `LangfuseObserver`, and `create_observer()`. We expand `LangfuseObserver` with structured event methods and wire `Observer` into `LifecycleManager` and `StrategyEngine`.

- [ ] **Step 1: Write failing tests for expanded observer**

Create `tests/test_reuse/test_observability.py`:

```python
"""Tests for save_your_tokens.reuse.observability — observer wiring."""

from unittest.mock import MagicMock, call

from save_your_tokens.core.budget import BudgetEngine
from save_your_tokens.core.lifecycle import LifecycleManager
from save_your_tokens.core.spec import (
    PROFILE_AGENTIC,
    ContextBlock,
    ContextLayer,
    ContextUsage,
)
from save_your_tokens.core.strategy import StrategyEngine
from save_your_tokens.reuse.observability import NoOpObserver


class TestNoOpObserver:
    def test_track_usage_noop(self):
        observer = NoOpObserver()
        observer.track_usage({"type": "test"})  # Should not raise

    def test_track_compaction_noop(self):
        observer = NoOpObserver()
        observer.track_compaction(before_tokens=100, after_tokens=50, method="extractive")

    def test_track_budget_warning_noop(self):
        observer = NoOpObserver()
        usage = MagicMock(spec=ContextUsage)
        observer.track_budget_warning(usage=usage, threshold="warn")

    def test_flush_noop(self):
        observer = NoOpObserver()
        observer.flush()  # Should not raise


class TestLifecycleObserverWiring:
    def test_lifecycle_accepts_observer(self):
        engine = BudgetEngine(context_window=200_000, profile=PROFILE_AGENTIC)
        observer = MagicMock()
        lm = LifecycleManager(budget_engine=engine, observer=observer)
        assert lm._observer is observer

    def test_lifecycle_defaults_to_noop(self):
        engine = BudgetEngine(context_window=200_000, profile=PROFILE_AGENTIC)
        lm = LifecycleManager(budget_engine=engine)
        assert isinstance(lm._observer, NoOpObserver)

    def test_post_turn_emits_event(self):
        engine = BudgetEngine(context_window=200_000, profile=PROFILE_AGENTIC)
        observer = MagicMock()
        lm = LifecycleManager(budget_engine=engine, observer=observer)
        lm.start_session()
        lm.post_turn()
        observer.track_usage.assert_called_once()
        event = observer.track_usage.call_args[0][0]
        assert event["type"] == "turn_complete"
        assert "turn_number" in event


class TestStrategyObserverWiring:
    def test_strategy_accepts_observer(self):
        engine = BudgetEngine(context_window=200_000, profile=PROFILE_AGENTIC)
        observer = MagicMock()
        se = StrategyEngine(budget_engine=engine, observer=observer)
        assert se._observer is observer

    def test_strategy_defaults_to_noop(self):
        engine = BudgetEngine(context_window=200_000, profile=PROFILE_AGENTIC)
        se = StrategyEngine(budget_engine=engine)
        assert isinstance(se._observer, NoOpObserver)
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/test_reuse/test_observability.py -v`
Expected: FAIL

- [ ] **Step 3: Expand observability.py**

Edit `src/save_your_tokens/reuse/observability.py` — add `track_compaction` and `track_budget_warning` methods to the `Observer` Protocol, `NoOpObserver`, and `LangfuseObserver`:

Add to `Observer` Protocol:
```python
def track_compaction(self, before_tokens: int, after_tokens: int, method: str) -> None: ...
def track_budget_warning(self, usage: Any, threshold: str) -> None: ...
```

Add to `NoOpObserver`:
```python
def track_compaction(self, before_tokens: int, after_tokens: int, method: str) -> None:
    pass

def track_budget_warning(self, usage: Any, threshold: str) -> None:
    pass
```

Add to `LangfuseObserver`:
```python
def track_compaction(self, before_tokens: int, after_tokens: int, method: str) -> None:
    self._client.trace(
        name="syt-compaction",
        metadata={
            "before_tokens": before_tokens,
            "after_tokens": after_tokens,
            "method": method,
            "savings_pct": round((1 - after_tokens / max(before_tokens, 1)) * 100, 1),
        },
    )

def track_budget_warning(self, usage: Any, threshold: str) -> None:
    self._client.trace(
        name="syt-budget-warning",
        metadata={"threshold": threshold, "usage": str(usage)},
    )
```

- [ ] **Step 4: Wire observer into LifecycleManager**

Edit `src/save_your_tokens/core/lifecycle.py`:

Change `__init__` signature:
```python
def __init__(self, budget_engine: BudgetEngine, observer: Any = None) -> None:
    self._engine = budget_engine
    self._phase = SessionPhase.INITIALIZING
    self._turn_history: list[TurnResult] = []
    self._stale_max_age: int = 10
    self._compact_interval: int = 0
    if observer is None:
        from save_your_tokens.reuse.observability import NoOpObserver
        observer = NoOpObserver()
    self._observer = observer
```

At end of `post_turn()`, before `return result`, add:
```python
self._observer.track_usage({
    "type": "turn_complete",
    "turn_number": result.turn_number,
    "needs_compaction": result.needs_compaction,
    "stale_blocks": len(result.stale_block_ids),
})
```

- [ ] **Step 5: Wire observer into StrategyEngine**

Edit `src/save_your_tokens/core/strategy.py`:

Change `__init__` signature:
```python
def __init__(
    self,
    budget_engine: BudgetEngine,
    compactor: Compactor | None = None,
    observer: Any = None,
) -> None:
    self._engine = budget_engine
    self._compactor = compactor or DefaultCompactor()
    if observer is None:
        from save_your_tokens.reuse.observability import NoOpObserver
        observer = NoOpObserver()
    self._observer = observer
```

In `execute_action()`, after the match/case block, add:
```python
self._observer.track_usage({"type": "compact_action", "action": action.value, "affected": len(result)})
```
(where `result` is the return value of the match arm)

- [ ] **Step 6: Run tests to verify they pass**

Run: `pytest tests/test_reuse/test_observability.py -v`
Expected: All PASS

- [ ] **Step 7: Run full test suite**

Run: `pytest -v`
Expected: All existing tests still PASS (NoOpObserver is the default, so no behavior change)

- [ ] **Step 8: Lint and commit**

```bash
ruff format src/save_your_tokens/reuse/observability.py src/save_your_tokens/core/lifecycle.py src/save_your_tokens/core/strategy.py tests/test_reuse/test_observability.py && ruff check src/save_your_tokens/reuse/ src/save_your_tokens/core/
git add src/save_your_tokens/reuse/observability.py src/save_your_tokens/core/lifecycle.py src/save_your_tokens/core/strategy.py tests/test_reuse/test_observability.py
git commit -m "feat: expand observer with structured events, wire into lifecycle and strategy"
```

---

## Chunk 4: Framework Integrations

### Task 5: Raw SDK Integration

**Files:**
- Create: `src/save_your_tokens/integrations/raw_sdk.py`
- Create: `tests/test_integrations/test_raw_sdk.py`

**Context:** Two-layer design: `RawSDKIntegration(FrameworkIntegration)` inner layer + `SYTWrapper` outer layer. Look at `integrations/claude_code.py` for the FrameworkIntegration pattern.

- [ ] **Step 1: Write failing tests**

Create `tests/test_integrations/test_raw_sdk.py`:

```python
"""Tests for save_your_tokens.integrations.raw_sdk — Raw SDK wrapper."""

from save_your_tokens.core.budget import BudgetEngine
from save_your_tokens.core.lifecycle import LifecycleManager
from save_your_tokens.core.spec import (
    PROFILE_AGENTIC,
    ContextBlock,
    ContextLayer,
)
from save_your_tokens.core.strategy import StrategyEngine
from save_your_tokens.integrations.base import FrameworkIntegration


def _make_engines():
    engine = BudgetEngine(context_window=200_000, profile=PROFILE_AGENTIC)
    lifecycle = LifecycleManager(budget_engine=engine)
    strategy = StrategyEngine(budget_engine=engine)
    return engine, lifecycle, strategy


class TestRawSDKIntegration:
    def test_extends_framework_integration(self):
        from save_your_tokens.integrations.raw_sdk import RawSDKIntegration

        assert issubclass(RawSDKIntegration, FrameworkIntegration)

    def test_setup_is_noop(self):
        from save_your_tokens.integrations.raw_sdk import RawSDKIntegration

        engine, lifecycle, strategy = _make_engines()
        integration = RawSDKIntegration(engine, lifecycle, strategy)
        integration.setup({})  # Should not raise

    def test_teardown_is_noop(self):
        from save_your_tokens.integrations.raw_sdk import RawSDKIntegration

        engine, lifecycle, strategy = _make_engines()
        integration = RawSDKIntegration(engine, lifecycle, strategy)
        integration.teardown()  # Should not raise

    def test_intercept_context_returns_messages(self):
        from save_your_tokens.integrations.raw_sdk import RawSDKIntegration

        engine, lifecycle, strategy = _make_engines()
        integration = RawSDKIntegration(engine, lifecycle, strategy)
        messages = [{"role": "user", "content": "hello"}]
        result = integration.intercept_context(messages)
        assert isinstance(result, list)

    def test_on_response_runs_post_turn(self):
        from save_your_tokens.integrations.raw_sdk import RawSDKIntegration

        engine, lifecycle, strategy = _make_engines()
        integration = RawSDKIntegration(engine, lifecycle, strategy)
        lifecycle.start_session()
        integration.on_response({"content": "response"})
        assert lifecycle.current_turn == 1


class TestSYTWrapper:
    def test_prepare_context_returns_list(self):
        from save_your_tokens.integrations.raw_sdk import SYTWrapper

        engine, lifecycle, strategy = _make_engines()
        wrapper = SYTWrapper(
            adapter=None,  # Not needed for basic prepare
            budget_engine=engine,
            lifecycle=lifecycle,
            strategy=strategy,
        )
        lifecycle.start_session()
        messages = [{"role": "user", "content": "hello"}]
        result = wrapper.prepare_context(messages)
        assert isinstance(result, list)

    def test_on_response_returns_summary(self):
        from save_your_tokens.integrations.raw_sdk import SYTWrapper

        engine, lifecycle, strategy = _make_engines()
        wrapper = SYTWrapper(
            adapter=None,
            budget_engine=engine,
            lifecycle=lifecycle,
            strategy=strategy,
        )
        lifecycle.start_session()
        result = wrapper.on_response({"content": "hello"})
        assert "turn" in result
        assert "usage" in result
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/test_integrations/test_raw_sdk.py -v`
Expected: FAIL

- [ ] **Step 3: Implement raw_sdk.py**

Create `src/save_your_tokens/integrations/raw_sdk.py`:

```python
"""Raw SDK integration — thin wrapper for direct SDK usage.

Two-layer design:
- RawSDKIntegration(FrameworkIntegration): inner layer with standard lifecycle
- SYTWrapper: outer layer providing ergonomic prepare_context/on_response API
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from save_your_tokens.core.spec import ContextBlock, ContextLayer
from save_your_tokens.integrations.base import FrameworkIntegration

if TYPE_CHECKING:
    from save_your_tokens.adapters.base import ModelAdapter
    from save_your_tokens.core.budget import BudgetEngine
    from save_your_tokens.core.lifecycle import LifecycleManager
    from save_your_tokens.core.strategy import StrategyEngine


class RawSDKIntegration(FrameworkIntegration):
    """Inner layer: implements FrameworkIntegration for raw SDK usage."""

    def setup(self, config: dict[str, Any]) -> None:
        """No-op — raw SDK doesn't need hook setup."""

    def teardown(self) -> None:
        """No-op — raw SDK doesn't need cleanup."""

    def intercept_context(self, messages: list[dict[str, Any]]) -> list[dict[str, Any]]:
        """Register messages as ephemeral blocks and return them."""
        for i, msg in enumerate(messages):
            content = msg.get("content", "")
            block = ContextBlock(
                id=f"msg:{i}",
                layer=ContextLayer.EPHEMERAL,
                content=content if isinstance(content, str) else str(content),
                token_count=len(str(content)) // 4,
                source=f"message:{i}",
                metadata={"role": msg.get("role", "user")},
            )
            self._budget.remove_block(block.id)
            self._budget.add_block(block)
        return messages

    def on_response(self, response: dict[str, Any]) -> None:
        """Run post-turn lifecycle."""
        self.run_post_turn()


class SYTWrapper:
    """Ergonomic wrapper for raw SDK usage.

    Usage:
        wrapper = SYTWrapper(adapter=ClaudeAdapter(), ...)
        messages = wrapper.prepare_context(my_messages)
        response = client.messages.create(messages=messages)
        summary = wrapper.on_response(response)
    """

    def __init__(
        self,
        adapter: ModelAdapter | None,
        budget_engine: BudgetEngine,
        lifecycle: LifecycleManager,
        strategy: StrategyEngine,
    ) -> None:
        self._adapter = adapter
        self._integration = RawSDKIntegration(budget_engine, lifecycle, strategy)

    def prepare_context(self, messages: list[dict[str, Any]]) -> list[dict[str, Any]]:
        """Classify, budget check, compact, and return ready-to-send messages."""
        return self._integration.intercept_context(messages)

    def on_response(self, response: dict[str, Any]) -> dict[str, Any]:
        """Post-turn lifecycle. Returns usage summary.

        Note: on_response() already calls run_post_turn() internally,
        so we just call it once and return the usage snapshot.
        """
        self._integration.on_response(response)
        return {
            "turn": self._integration._lifecycle.current_turn,
            "needs_compaction": False,
            "usage": self._integration.get_usage().model_dump(),
        }
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/test_integrations/test_raw_sdk.py -v`
Expected: All PASS

- [ ] **Step 5: Lint and commit**

```bash
ruff format src/save_your_tokens/integrations/raw_sdk.py tests/test_integrations/test_raw_sdk.py && ruff check src/save_your_tokens/integrations/ tests/test_integrations/
git add src/save_your_tokens/integrations/raw_sdk.py tests/test_integrations/test_raw_sdk.py
git commit -m "feat: add Raw SDK integration with SYTWrapper"
```

---

### Task 6: LangChain Integration

**Files:**
- Create: `src/save_your_tokens/integrations/langchain.py`
- Create: `tests/test_integrations/test_langchain.py`
- Modify: `pyproject.toml`

**Context:** Two-layer design: `LangChainIntegration(FrameworkIntegration)` + `SYTRunnable(RunnableSerializable)`. LangChain is an optional dep — all tests mock the langchain imports.

- [ ] **Step 1: Add langchain optional dep**

In `pyproject.toml`:
```toml
langchain = ["langchain-core>=0.3"]
```
Update `all`:
```toml
all = ["save-your-tokens[claude,openai,deepseek,gemini,langfuse,langchain,local]"]
```

- [ ] **Step 2: Write failing tests**

Create `tests/test_integrations/test_langchain.py`:

```python
"""Tests for save_your_tokens.integrations.langchain — LangChain LCEL integration."""

from unittest.mock import MagicMock, patch

from save_your_tokens.core.budget import BudgetEngine
from save_your_tokens.core.lifecycle import LifecycleManager
from save_your_tokens.core.spec import PROFILE_AGENTIC
from save_your_tokens.core.strategy import StrategyEngine
from save_your_tokens.integrations.base import FrameworkIntegration


def _make_engines():
    engine = BudgetEngine(context_window=200_000, profile=PROFILE_AGENTIC)
    lifecycle = LifecycleManager(budget_engine=engine)
    strategy = StrategyEngine(budget_engine=engine)
    return engine, lifecycle, strategy


class TestLangChainIntegration:
    def test_extends_framework_integration(self):
        from save_your_tokens.integrations.langchain import LangChainIntegration

        assert issubclass(LangChainIntegration, FrameworkIntegration)

    def test_setup_is_noop(self):
        from save_your_tokens.integrations.langchain import LangChainIntegration

        engine, lifecycle, strategy = _make_engines()
        integration = LangChainIntegration(engine, lifecycle, strategy)
        integration.setup({})

    def test_intercept_context_returns_messages(self):
        from save_your_tokens.integrations.langchain import LangChainIntegration

        engine, lifecycle, strategy = _make_engines()
        integration = LangChainIntegration(engine, lifecycle, strategy)
        messages = [{"role": "user", "content": "hello"}]
        result = integration.intercept_context(messages)
        assert isinstance(result, list)

    def test_on_response_advances_turn(self):
        from save_your_tokens.integrations.langchain import LangChainIntegration

        engine, lifecycle, strategy = _make_engines()
        integration = LangChainIntegration(engine, lifecycle, strategy)
        lifecycle.start_session()
        integration.on_response({"content": "reply"})
        assert lifecycle.current_turn == 1


class TestSYTRunnable:
    @patch("save_your_tokens.integrations.langchain.RunnableSerializable", MagicMock)
    def test_invoke_delegates_to_integration(self):
        from save_your_tokens.integrations.langchain import SYTRunnable

        engine, lifecycle, strategy = _make_engines()
        lifecycle.start_session()
        runnable = SYTRunnable(
            budget_engine=engine,
            lifecycle=lifecycle,
            strategy=strategy,
        )
        result = runnable.invoke({"messages": [{"role": "user", "content": "hi"}]})
        assert isinstance(result, dict)
        assert "messages" in result
```

- [ ] **Step 3: Run tests to verify they fail**

Run: `pytest tests/test_integrations/test_langchain.py -v`
Expected: FAIL

- [ ] **Step 4: Implement langchain.py**

Create `src/save_your_tokens/integrations/langchain.py`:

```python
"""LangChain LCEL integration.

Two-layer design:
- LangChainIntegration(FrameworkIntegration): inner layer with standard lifecycle
- SYTRunnable: LCEL-native Runnable that delegates to LangChainIntegration

The SYTRunnable can be composed into any LCEL chain:
    chain = SYTRunnable(budget_engine, adapter) | llm | output_parser
"""

from __future__ import annotations

from typing import Any

from save_your_tokens.core.spec import ContextBlock, ContextLayer
from save_your_tokens.integrations.base import FrameworkIntegration

try:
    from langchain_core.runnables import RunnableSerializable
except ImportError:
    # Allow import without langchain installed (for tests and type checking)
    RunnableSerializable = object  # type: ignore[assignment, misc]


class LangChainIntegration(FrameworkIntegration):
    """Inner layer: implements FrameworkIntegration for LangChain."""

    def setup(self, config: dict[str, Any]) -> None:
        """No-op — LCEL doesn't use hook-based setup."""

    def teardown(self) -> None:
        """No-op — LCEL doesn't need cleanup."""

    def intercept_context(self, messages: list[dict[str, Any]]) -> list[dict[str, Any]]:
        """Register messages as ephemeral blocks, run budget check."""
        for i, msg in enumerate(messages):
            content = msg.get("content", "")
            block = ContextBlock(
                id=f"lc-msg:{i}",
                layer=ContextLayer.EPHEMERAL,
                content=content if isinstance(content, str) else str(content),
                token_count=len(str(content)) // 4,
                source=f"langchain:{i}",
                metadata={"role": msg.get("role", "user")},
            )
            self._budget.remove_block(block.id)
            self._budget.add_block(block)
        return messages

    def on_response(self, response: dict[str, Any]) -> None:
        """Run post-turn lifecycle."""
        self.run_post_turn()


class SYTRunnable(RunnableSerializable):
    """LCEL-native Runnable that manages context budget in the chain.

    Wraps LangChainIntegration for composability with other LCEL components.
    """

    def __init__(
        self,
        budget_engine: Any,
        lifecycle: Any,
        strategy: Any,
        **kwargs: Any,
    ) -> None:
        self._integration = LangChainIntegration(budget_engine, lifecycle, strategy)

    def invoke(self, input: Any, config: Any = None) -> dict[str, Any]:
        """Intercept context, run budget management, return processed input."""
        messages = input.get("messages", []) if isinstance(input, dict) else []
        processed = self._integration.intercept_context(messages)
        return {"messages": processed}

    async def ainvoke(self, input: Any, config: Any = None) -> dict[str, Any]:
        """Async version — delegates to sync invoke."""
        return self.invoke(input, config)
```

- [ ] **Step 5: Run tests to verify they pass**

Run: `pytest tests/test_integrations/test_langchain.py -v`
Expected: All PASS

- [ ] **Step 6: Lint and commit**

```bash
ruff format src/save_your_tokens/integrations/langchain.py tests/test_integrations/test_langchain.py && ruff check src/save_your_tokens/integrations/ tests/test_integrations/
git add src/save_your_tokens/integrations/langchain.py tests/test_integrations/test_langchain.py pyproject.toml
git commit -m "feat: add LangChain LCEL integration with SYTRunnable"
```

---

## Chunk 5: Benchmark + Publish

### Task 7: Recorded Session Benchmark

**Files:**
- Create: `benchmarks/recorded_replay.py`
- Create: `benchmarks/fixtures/claude_coding_session.json`
- Create: `benchmarks/fixtures/qa_session.json`
- Create: `benchmarks/fixtures/long_context_session.json`
- Create: `benchmarks/__init__.py`
- Create: `benchmarks/test_recorded_replay.py`

**Context:** Replay engine feeds recorded turns through syt's lifecycle. Uses char-based token estimation (`len(text) // 4`), no live API calls. Reports savings % and compaction events.

- [ ] **Step 1: Create benchmark fixtures directory**

```bash
mkdir -p benchmarks/fixtures
touch benchmarks/__init__.py
```

- [ ] **Step 2: Create sample fixtures**

Create `benchmarks/fixtures/claude_coding_session.json` — a synthetic 50-turn coding session with realistic token counts. Generate user messages that ask coding questions and assistant responses with code. Token counts should be realistic (user: 50-500, assistant: 200-2000).

Create `benchmarks/fixtures/qa_session.json` — a 20-turn Q&A session with shorter exchanges.

Create `benchmarks/fixtures/long_context_session.json` — a 100-turn session with some large file-read responses (5000+ tokens).

Each fixture follows this format:
```json
{
  "metadata": {"model": "claude-sonnet-4-6", "recorded_at": "2026-03-22", "description": "..."},
  "turns": [{"role": "user", "content": "...", "token_count": 123}, ...]
}
```

- [ ] **Step 3: Write failing test for replay engine**

Create `benchmarks/test_recorded_replay.py`:

```python
"""Tests for the recorded session replay benchmark engine."""

import json
from pathlib import Path

import pytest


class TestReplayEngine:
    def test_load_fixture(self):
        from benchmarks.recorded_replay import load_fixture

        fixture_dir = Path(__file__).parent / "fixtures"
        data = load_fixture(fixture_dir / "qa_session.json")
        assert "metadata" in data
        assert "turns" in data
        assert len(data["turns"]) > 0

    def test_replay_produces_report(self):
        from benchmarks.recorded_replay import replay_session

        fixture_dir = Path(__file__).parent / "fixtures"
        report = replay_session(fixture_dir / "qa_session.json")
        assert "baseline_tokens" in report
        assert "syt_tokens" in report
        assert "savings_pct" in report
        assert report["savings_pct"] >= 0

    def test_replay_coding_session(self):
        from benchmarks.recorded_replay import replay_session

        fixture_dir = Path(__file__).parent / "fixtures"
        report = replay_session(fixture_dir / "claude_coding_session.json")
        assert report["total_turns"] == 50

    def test_replay_long_context_session(self):
        from benchmarks.recorded_replay import replay_session

        fixture_dir = Path(__file__).parent / "fixtures"
        report = replay_session(fixture_dir / "long_context_session.json")
        assert report["total_turns"] >= 100
        assert report["compaction_events"] > 0  # Should trigger compaction
```

- [ ] **Step 4: Run tests to verify they fail**

Run: `pytest benchmarks/test_recorded_replay.py -v`
Expected: FAIL

- [ ] **Step 5: Implement replay engine**

Create `benchmarks/recorded_replay.py`:

```python
"""Recorded session replay benchmark.

Feeds recorded conversation turns through syt's lifecycle to measure
token savings vs. a no-management baseline.

Usage:
    python -m benchmarks.recorded_replay benchmarks/fixtures/qa_session.json
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

from save_your_tokens.core.budget import BudgetEngine
from save_your_tokens.core.lifecycle import LifecycleManager
from save_your_tokens.core.spec import PROFILE_AGENTIC, ContextBlock, ContextLayer
from save_your_tokens.core.strategy import StrategyEngine


def load_fixture(path: Path) -> dict[str, Any]:
    """Load a recorded session fixture from JSON."""
    return json.loads(path.read_text(encoding="utf-8"))


def estimate_tokens(text: str) -> int:
    """Rough token estimate: 1 token ≈ 4 chars."""
    return len(text) // 4


def replay_session(
    fixture_path: Path,
    context_window: int = 200_000,
) -> dict[str, Any]:
    """Replay a recorded session through syt and measure savings.

    Returns a report dict with baseline vs. syt token counts.
    """
    data = load_fixture(fixture_path)
    turns = data["turns"]

    engine = BudgetEngine(context_window=context_window, profile=PROFILE_AGENTIC)
    lifecycle = LifecycleManager(budget_engine=engine)
    strategy = StrategyEngine(budget_engine=engine)

    lifecycle.start_session()

    baseline_tokens = 0
    syt_tokens = 0
    compaction_events = 0
    cumulative_baseline: list[int] = []
    cumulative_syt: list[int] = []

    for i, turn in enumerate(turns):
        token_count = turn.get("token_count", estimate_tokens(turn["content"]))
        baseline_tokens += token_count

        block = ContextBlock(
            id=f"turn:{i}",
            layer=ContextLayer.EPHEMERAL,
            content=turn["content"],
            token_count=token_count,
            source=f"replay:{i}",
            metadata={"role": turn["role"]},
        )
        engine.remove_block(block.id)
        engine.add_block(block)

        result = lifecycle.post_turn()
        if result.needs_compaction:
            strategy.execute_actions(result.recommended_actions)
            compaction_events += 1

        usage = engine.compute_budgets()
        syt_tokens = usage.total_used
        cumulative_baseline.append(baseline_tokens)
        cumulative_syt.append(syt_tokens)

    savings_pct = (
        round((1 - syt_tokens / max(baseline_tokens, 1)) * 100, 1) if baseline_tokens > 0 else 0.0
    )

    return {
        "fixture": fixture_path.name,
        "total_turns": len(turns),
        "baseline_tokens": baseline_tokens,
        "syt_tokens": syt_tokens,
        "savings_pct": savings_pct,
        "compaction_events": compaction_events,
    }


def print_report(report: dict[str, Any]) -> None:
    """Print a markdown-formatted benchmark report."""
    print(f"## Benchmark: {report['fixture']}")
    print(f"- Turns: {report['total_turns']}")
    print(f"- Baseline tokens: {report['baseline_tokens']:,}")
    print(f"- SYT tokens: {report['syt_tokens']:,}")
    print(f"- **Savings: {report['savings_pct']}%**")
    print(f"- Compaction events: {report['compaction_events']}")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python -m benchmarks.recorded_replay <fixture_path>")
        sys.exit(1)
    path = Path(sys.argv[1])
    report = replay_session(path)
    print_report(report)
```

- [ ] **Step 6: Run tests to verify they pass**

Run: `pytest benchmarks/test_recorded_replay.py -v`
Expected: All PASS

- [ ] **Step 7: Run all benchmarks and verify positive savings**

Run: `python -m benchmarks.recorded_replay benchmarks/fixtures/claude_coding_session.json`
Run: `python -m benchmarks.recorded_replay benchmarks/fixtures/qa_session.json`
Run: `python -m benchmarks.recorded_replay benchmarks/fixtures/long_context_session.json`

Expected: Savings > 0% for long_context_session (should trigger compaction), savings may be 0% for short sessions that stay within budget.

- [ ] **Step 8: Lint and commit**

```bash
ruff format benchmarks/ && ruff check benchmarks/
git add benchmarks/
git commit -m "feat: add recorded session replay benchmark with 3 fixtures"
```

---

### Task 8: PyPI Publishing Docs + Version Bump

**Files:**
- Create: `docs/PUBLISHING.md`
- Modify: `pyproject.toml` (version bump)
- Modify: `docs/progress.md`

- [ ] **Step 1: Create publishing guide**

Create `docs/PUBLISHING.md`:

```markdown
# Publishing to PyPI

## Prerequisites

1. [PyPI account](https://pypi.org/account/register/)
2. API token from PyPI (Account Settings → API tokens)
3. `hatch` installed: `pip install hatch`

## Pre-Publish Checklist

- [ ] Version bumped in `pyproject.toml`
- [ ] All tests pass: `pytest -v`
- [ ] Lint clean: `ruff format . && ruff check .`
- [ ] Type check: `mypy src/`
- [ ] Wheel builds: `hatch build`
- [ ] CHANGELOG updated (if applicable)

## Build

```bash
hatch build
```

This produces `dist/save_your_tokens-X.Y.Z-py3-none-any.whl` and `.tar.gz`.

## Upload

### Using hatch (recommended)

```bash
hatch publish
```

### Using twine

```bash
pip install twine
twine upload dist/*
```

You'll be prompted for your PyPI username and API token.

## Verify

```bash
pip install save-your-tokens
python -c "import save_your_tokens; print(save_your_tokens.__version__)"
```

## Version Convention

We follow [semver](https://semver.org/):
- `0.1.0` — Phase 1 (initial release)
- `0.2.0` — Phase 2 (ecosystem expansion)
- `1.0.0` — Production-ready (stable API)

Bump version in `pyproject.toml` before each release.
```

- [ ] **Step 2: Bump version**

Edit `pyproject.toml`: change `version = "0.1.0"` to `version = "0.2.0"`.

- [ ] **Step 3: Update progress.md**

Edit `docs/progress.md` to reflect Phase 2 completion:

```markdown
## Current Status

- Completed: Phase 1 — core engine, skills, CLI, 133 tests, 79% coverage
- Completed: Phase 2 — ecosystem expansion
  - DeepSeek + Gemini model adapters
  - Independent Compactor interface (extractive, truncation, LLM, local)
  - Langfuse observability wired into lifecycle + strategy
  - LangChain LCEL + Raw SDK framework integrations
  - Recorded session replay benchmark (3 fixtures)
  - PyPI publishing documentation
- Package version: 0.2.0
```

- [ ] **Step 4: Run full test suite**

Run: `pytest -v --cov=save_your_tokens --cov-report=term-missing`
Expected: 80%+ coverage, all tests pass

- [ ] **Step 5: Build wheel to verify**

Run: `hatch build`
Expected: Produces `dist/save_your_tokens-0.2.0-py3-none-any.whl`

- [ ] **Step 6: Lint and commit**

```bash
ruff format . && ruff check .
git add docs/PUBLISHING.md docs/progress.md pyproject.toml
git commit -m "docs: add PyPI publishing guide, bump version to 0.2.0"
```

---

## Summary

| Task | Description | Files Created | Files Modified |
|------|-------------|---------------|----------------|
| 1 | DeepSeek adapter | 2 | 1 |
| 2 | Gemini adapter + registry | 2 | 2 |
| 3 | Compactor refactor + backends | 2 | 2 |
| 4 | Observer expansion + lifecycle wiring | 1 | 3 |
| 5 | Raw SDK integration | 2 | 0 |
| 6 | LangChain integration | 2 | 1 |
| 7 | Recorded session benchmark | 6 | 0 |
| 8 | PyPI docs + version bump | 1 | 2 |

**Total: 18 new files, 11 modifications, 8 commits**
