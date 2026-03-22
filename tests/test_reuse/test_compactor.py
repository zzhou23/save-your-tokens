"""Tests for save_your_tokens.reuse.compactor module."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from save_your_tokens.reuse.compactor import (
    DefaultCompactor,
    LLMCompactor,
    LocalModelCompactor,
    TruncationCompactor,
    create_compactor,
)


class TestDefaultCompactor:
    def test_short_content_unchanged(self) -> None:
        compactor = DefaultCompactor()
        content = "Short text."
        result = compactor.compact(content, target_tokens=1000)
        assert result == content

    def test_long_content_compressed(self) -> None:
        compactor = DefaultCompactor()
        # 1000 chars > 20 tokens * 4 chars = 80 chars
        content = "This is a sentence. " * 50  # ~1000 chars
        result = compactor.compact(content, target_tokens=20)
        assert len(result) < len(content)


class TestTruncationCompactor:
    def test_short_content_unchanged(self) -> None:
        compactor = TruncationCompactor()
        content = "Short text."
        result = compactor.compact(content, target_tokens=1000)
        assert result == content

    def test_long_content_truncated_with_marker(self) -> None:
        compactor = TruncationCompactor()
        content = "A" * 400  # 400 chars > 20 tokens * 4 = 80 chars
        result = compactor.compact(content, target_tokens=20)
        assert len(result) < len(content)
        assert "[... truncated ...]" in result


class TestLLMCompactor:
    def test_compact_with_supporting_adapter(self) -> None:
        adapter = MagicMock()
        adapter.supports_native_compact = True
        adapter.model_name = "test-model"
        adapter.model_compact.return_value = "compacted result"

        compactor = LLMCompactor(adapter)
        result = compactor.compact("some long content", target_tokens=50)

        assert result == "compacted result"
        adapter.model_compact.assert_called_once_with("some long content", 50)

    def test_raises_not_implemented_if_adapter_not_supporting(self) -> None:
        adapter = MagicMock()
        adapter.supports_native_compact = False
        adapter.model_name = "test-model"

        compactor = LLMCompactor(adapter)
        with pytest.raises(NotImplementedError, match="does not support native compaction"):
            compactor.compact("some content", target_tokens=50)


class TestLocalModelCompactor:
    def test_mock_httpx_post_called(self) -> None:
        mock_response = MagicMock()
        mock_response.json.return_value = {"choices": [{"message": {"content": "summarized text"}}]}

        with patch("httpx.post", return_value=mock_response) as mock_post:
            compactor = LocalModelCompactor(endpoint="http://localhost:11434", model="llama3")
            result = compactor.compact("some content to summarize", target_tokens=50)

        assert result == "summarized text"
        mock_post.assert_called_once()
        call_args = mock_post.call_args
        assert "http://localhost:11434/v1/chat/completions" in call_args[0]

    def test_custom_token_estimator_stored(self) -> None:
        estimator = lambda text: len(text) // 3
        compactor = LocalModelCompactor(token_estimator=estimator)
        assert compactor._token_estimator is estimator

    def test_default_token_estimator(self) -> None:
        compactor = LocalModelCompactor()
        assert compactor._token_estimator("hello world") == len("hello world") // 4


class TestCreateCompactor:
    def test_creates_extractive(self) -> None:
        compactor = create_compactor("extractive")
        assert isinstance(compactor, DefaultCompactor)

    def test_creates_truncation(self) -> None:
        compactor = create_compactor("truncation")
        assert isinstance(compactor, TruncationCompactor)

    def test_creates_llm(self) -> None:
        adapter = MagicMock()
        adapter.supports_native_compact = True
        adapter.model_name = "mock-model"
        compactor = create_compactor("llm", adapter=adapter)
        assert isinstance(compactor, LLMCompactor)

    def test_creates_local(self) -> None:
        compactor = create_compactor("local")
        assert isinstance(compactor, LocalModelCompactor)

    def test_raises_value_error_for_unknown(self) -> None:
        with pytest.raises(ValueError, match="Unknown compactor backend"):
            create_compactor("nonexistent")
