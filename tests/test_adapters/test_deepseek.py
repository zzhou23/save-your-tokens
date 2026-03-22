"""Tests for DeepSeek model adapter."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from save_your_tokens.core.spec import ContextBlock, ContextLayer


class TestDeepSeekAdapterProperties:
    def test_default_model_name(self):
        from save_your_tokens.adapters.deepseek import DeepSeekAdapter

        adapter = DeepSeekAdapter()
        assert adapter.model_name == "deepseek-chat"

    def test_custom_model_name(self):
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

    def test_context_window_unknown_defaults_64k(self):
        from save_your_tokens.adapters.deepseek import DeepSeekAdapter

        adapter = DeepSeekAdapter(model="deepseek-unknown-future-model")
        assert adapter.context_window == 64_000


class TestDeepSeekTokenCounting:
    def test_count_tokens_uses_cl100k_base(self):
        from save_your_tokens.adapters.deepseek import DeepSeekAdapter

        mock_tiktoken = MagicMock()
        mock_encoding = MagicMock()
        mock_encoding.encode.return_value = [1, 2, 3, 4, 5]
        mock_tiktoken.get_encoding.return_value = mock_encoding

        with patch("save_your_tokens.adapters.deepseek.tiktoken", mock_tiktoken):
            adapter = DeepSeekAdapter()
            result = adapter.count_tokens("hello world")

        mock_tiktoken.get_encoding.assert_called_once_with("cl100k_base")
        mock_encoding.encode.assert_called_once_with("hello world")
        assert result == 5

    def test_count_tokens_encoding_cached(self):
        from save_your_tokens.adapters.deepseek import DeepSeekAdapter

        mock_tiktoken = MagicMock()
        mock_encoding = MagicMock()
        mock_encoding.encode.return_value = [1, 2, 3]
        mock_tiktoken.get_encoding.return_value = mock_encoding

        with patch("save_your_tokens.adapters.deepseek.tiktoken", mock_tiktoken):
            adapter = DeepSeekAdapter()
            adapter.count_tokens("first call")
            adapter.count_tokens("second call")

        # get_encoding should only be called once (cached)
        assert mock_tiktoken.get_encoding.call_count == 1

    def test_count_tokens_raises_import_error_when_tiktoken_none(self):
        from save_your_tokens.adapters.deepseek import DeepSeekAdapter

        with patch("save_your_tokens.adapters.deepseek.tiktoken", None):
            adapter = DeepSeekAdapter()
            with pytest.raises(ImportError, match="tiktoken"):
                adapter.count_tokens("hello")


class TestDeepSeekFormatContext:
    def _make_block(
        self, id: str, layer: ContextLayer, content: str, role: str = "user"
    ) -> ContextBlock:
        return ContextBlock(
            id=id,
            layer=layer,
            content=content,
            metadata={"role": role},
        )

    def test_persistent_becomes_system(self):
        from save_your_tokens.adapters.deepseek import DeepSeekAdapter

        adapter = DeepSeekAdapter()
        block = self._make_block("p1", ContextLayer.PERSISTENT, "You are a helpful assistant.")
        messages = adapter.format_context([block], [], [])

        assert len(messages) == 1
        assert messages[0]["role"] == "system"
        assert messages[0]["content"] == "You are a helpful assistant."

    def test_session_becomes_system_with_prefix(self):
        from save_your_tokens.adapters.deepseek import DeepSeekAdapter

        adapter = DeepSeekAdapter()
        block = self._make_block("s1", ContextLayer.SESSION, "User is debugging Python code.")
        messages = adapter.format_context([], [block], [])

        assert len(messages) == 1
        assert messages[0]["role"] == "system"
        assert messages[0]["content"].startswith("[Session Context]")
        assert "User is debugging Python code." in messages[0]["content"]

    def test_ephemeral_preserves_role(self):
        from save_your_tokens.adapters.deepseek import DeepSeekAdapter

        adapter = DeepSeekAdapter()
        user_block = self._make_block("e1", ContextLayer.EPHEMERAL, "What is 2+2?", role="user")
        asst_block = self._make_block("e2", ContextLayer.EPHEMERAL, "It is 4.", role="assistant")
        messages = adapter.format_context([], [], [user_block, asst_block])

        assert len(messages) == 2
        assert messages[0] == {"role": "user", "content": "What is 2+2?"}
        assert messages[1] == {"role": "assistant", "content": "It is 4."}

    def test_all_layers_combined_in_order(self):
        from save_your_tokens.adapters.deepseek import DeepSeekAdapter

        adapter = DeepSeekAdapter()
        persistent = [self._make_block("p1", ContextLayer.PERSISTENT, "System prompt.")]
        session = [self._make_block("s1", ContextLayer.SESSION, "Session info.")]
        ephemeral = [self._make_block("e1", ContextLayer.EPHEMERAL, "User message.", role="user")]

        messages = adapter.format_context(persistent, session, ephemeral)

        assert len(messages) == 3
        assert messages[0]["role"] == "system"
        assert messages[0]["content"] == "System prompt."
        assert messages[1]["role"] == "system"
        assert "[Session Context]" in messages[1]["content"]
        assert messages[2]["role"] == "user"
        assert messages[2]["content"] == "User message."

    def test_empty_blocks_skipped(self):
        from save_your_tokens.adapters.deepseek import DeepSeekAdapter

        adapter = DeepSeekAdapter()
        empty_persistent = [ContextBlock(id="p0", layer=ContextLayer.PERSISTENT, content="")]
        messages = adapter.format_context(empty_persistent, [], [])

        assert messages == []

    def test_multiple_persistent_joined(self):
        from save_your_tokens.adapters.deepseek import DeepSeekAdapter

        adapter = DeepSeekAdapter()
        blocks = [
            self._make_block("p1", ContextLayer.PERSISTENT, "Part A."),
            self._make_block("p2", ContextLayer.PERSISTENT, "Part B."),
        ]
        messages = adapter.format_context(blocks, [], [])

        assert len(messages) == 1
        assert "Part A." in messages[0]["content"]
        assert "Part B." in messages[0]["content"]


class TestDeepSeekModelCompact:
    def test_model_compact_calls_api(self):
        from save_your_tokens.adapters.deepseek import DeepSeekAdapter

        mock_openai_class = MagicMock()
        mock_client = MagicMock()
        mock_openai_class.return_value = mock_client

        mock_response = MagicMock()
        mock_response.choices[0].message.content = "Compacted summary."
        mock_client.chat.completions.create.return_value = mock_response

        with patch("save_your_tokens.adapters.deepseek.OpenAI", mock_openai_class):
            adapter = DeepSeekAdapter()
            result = adapter.model_compact("Long text to compact.", target_tokens=100)

        assert result == "Compacted summary."
        mock_client.chat.completions.create.assert_called_once()
        call_kwargs = mock_client.chat.completions.create.call_args[1]
        assert call_kwargs["model"] == "deepseek-chat"

    def test_model_compact_passes_content_in_messages(self):
        from save_your_tokens.adapters.deepseek import DeepSeekAdapter

        mock_openai_class = MagicMock()
        mock_client = MagicMock()
        mock_openai_class.return_value = mock_client

        mock_response = MagicMock()
        mock_response.choices[0].message.content = "Summary."
        mock_client.chat.completions.create.return_value = mock_response

        with patch("save_your_tokens.adapters.deepseek.OpenAI", mock_openai_class):
            adapter = DeepSeekAdapter()
            adapter.model_compact("Content to summarize.", target_tokens=50)

        call_kwargs = mock_client.chat.completions.create.call_args[1]
        messages = call_kwargs["messages"]
        # Should have a system message and user message with the content
        assert any("Content to summarize." in str(m) for m in messages)

    def test_supports_native_compact_true(self):
        from save_your_tokens.adapters.deepseek import DeepSeekAdapter

        adapter = DeepSeekAdapter()
        assert adapter.supports_native_compact is True
