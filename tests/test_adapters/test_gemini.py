"""Tests for Gemini model adapter."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from save_your_tokens.core.spec import ContextBlock, ContextLayer


class TestGeminiAdapterProperties:
    def test_default_model_name(self):
        from save_your_tokens.adapters.gemini import GeminiAdapter

        adapter = GeminiAdapter()
        assert adapter.model_name == "gemini-2.0-flash"

    def test_custom_model_name(self):
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

    def test_context_window_unknown_defaults_1m(self):
        from save_your_tokens.adapters.gemini import GeminiAdapter

        adapter = GeminiAdapter(model="gemini-future-model")
        assert adapter.context_window == 1_000_000

    def test_recommended_output_reserve_is_10_percent(self):
        from save_your_tokens.adapters.gemini import GeminiAdapter

        adapter = GeminiAdapter()
        assert adapter.recommended_output_reserve == int(1_000_000 * 0.1)


class TestGeminiGetClient:
    def test_get_client_raises_when_genai_none(self):
        from save_your_tokens.adapters.gemini import GeminiAdapter

        with patch("save_your_tokens.adapters.gemini.genai", None):
            adapter = GeminiAdapter()
            with pytest.raises(ImportError, match="google-genai"):
                adapter._get_client()

    def test_get_client_creates_client_without_api_key(self):
        from save_your_tokens.adapters.gemini import GeminiAdapter

        mock_genai = MagicMock()
        mock_client = MagicMock()
        mock_genai.Client.return_value = mock_client

        with patch("save_your_tokens.adapters.gemini.genai", mock_genai):
            adapter = GeminiAdapter()
            client = adapter._get_client()

        mock_genai.Client.assert_called_once_with()
        assert client is mock_client

    def test_get_client_passes_api_key_when_provided(self):
        from save_your_tokens.adapters.gemini import GeminiAdapter

        mock_genai = MagicMock()
        mock_client = MagicMock()
        mock_genai.Client.return_value = mock_client

        with patch("save_your_tokens.adapters.gemini.genai", mock_genai):
            adapter = GeminiAdapter(api_key="test-key-123")
            client = adapter._get_client()

        mock_genai.Client.assert_called_once_with(api_key="test-key-123")
        assert client is mock_client

    def test_get_client_cached_after_first_call(self):
        from save_your_tokens.adapters.gemini import GeminiAdapter

        mock_genai = MagicMock()
        mock_client = MagicMock()
        mock_genai.Client.return_value = mock_client

        with patch("save_your_tokens.adapters.gemini.genai", mock_genai):
            adapter = GeminiAdapter()
            adapter._get_client()
            adapter._get_client()

        assert mock_genai.Client.call_count == 1


class TestGeminiCountTokens:
    def test_count_tokens_calls_api(self):
        from save_your_tokens.adapters.gemini import GeminiAdapter

        mock_genai = MagicMock()
        mock_client = MagicMock()
        mock_genai.Client.return_value = mock_client

        mock_result = MagicMock()
        mock_result.total_tokens = 42
        mock_client.models.count_tokens.return_value = mock_result

        with patch("save_your_tokens.adapters.gemini.genai", mock_genai):
            adapter = GeminiAdapter(model="gemini-2.0-flash")
            result = adapter.count_tokens("hello world")

        mock_client.models.count_tokens.assert_called_once_with(
            model="gemini-2.0-flash", contents="hello world"
        )
        assert result == 42

    def test_count_tokens_returns_total_tokens(self):
        from save_your_tokens.adapters.gemini import GeminiAdapter

        mock_genai = MagicMock()
        mock_client = MagicMock()
        mock_genai.Client.return_value = mock_client

        mock_result = MagicMock()
        mock_result.total_tokens = 100
        mock_client.models.count_tokens.return_value = mock_result

        with patch("save_your_tokens.adapters.gemini.genai", mock_genai):
            adapter = GeminiAdapter()
            result = adapter.count_tokens("some text")

        assert result == 100


class TestGeminiFormatContext:
    def _make_block(
        self, id: str, layer: ContextLayer, content: str, role: str = "user"
    ) -> ContextBlock:
        return ContextBlock(
            id=id,
            layer=layer,
            content=content,
            metadata={"role": role},
        )

    def test_persistent_becomes_user_role_with_system_instructions_prefix(self):
        from save_your_tokens.adapters.gemini import GeminiAdapter

        adapter = GeminiAdapter()
        block = self._make_block("p1", ContextLayer.PERSISTENT, "You are a helpful assistant.")
        messages = adapter.format_context([block], [], [])

        assert len(messages) == 1
        assert messages[0]["role"] == "user"
        assert messages[0]["parts"][0]["text"].startswith("[System Instructions]")
        assert "You are a helpful assistant." in messages[0]["parts"][0]["text"]

    def test_session_becomes_user_with_prefix(self):
        from save_your_tokens.adapters.gemini import GeminiAdapter

        adapter = GeminiAdapter()
        block = self._make_block("s1", ContextLayer.SESSION, "User is debugging Python code.")
        messages = adapter.format_context([], [block], [])

        assert len(messages) == 1
        assert messages[0]["role"] == "user"
        assert messages[0]["parts"][0]["text"].startswith("[Session Context]")
        assert "User is debugging Python code." in messages[0]["parts"][0]["text"]

    def test_ephemeral_user_role_preserved(self):
        from save_your_tokens.adapters.gemini import GeminiAdapter

        adapter = GeminiAdapter()
        block = self._make_block("e1", ContextLayer.EPHEMERAL, "What is 2+2?", role="user")
        messages = adapter.format_context([], [], [block])

        assert len(messages) == 1
        assert messages[0]["role"] == "user"
        assert messages[0]["parts"] == [{"text": "What is 2+2?"}]

    def test_ephemeral_assistant_mapped_to_model(self):
        from save_your_tokens.adapters.gemini import GeminiAdapter

        adapter = GeminiAdapter()
        block = self._make_block("e1", ContextLayer.EPHEMERAL, "It is 4.", role="assistant")
        messages = adapter.format_context([], [], [block])

        assert len(messages) == 1
        assert messages[0]["role"] == "model"
        assert messages[0]["parts"] == [{"text": "It is 4."}]

    def test_all_layers_combined_in_order(self):
        from save_your_tokens.adapters.gemini import GeminiAdapter

        adapter = GeminiAdapter()
        persistent = [self._make_block("p1", ContextLayer.PERSISTENT, "System prompt.")]
        session = [self._make_block("s1", ContextLayer.SESSION, "Session info.")]
        ephemeral = [self._make_block("e1", ContextLayer.EPHEMERAL, "User message.", role="user")]

        messages = adapter.format_context(persistent, session, ephemeral)

        assert len(messages) == 3
        assert messages[0]["role"] == "user"
        assert "[System Instructions]" in messages[0]["parts"][0]["text"]
        assert messages[1]["role"] == "user"
        assert "[Session Context]" in messages[1]["parts"][0]["text"]
        assert messages[2]["role"] == "user"
        assert messages[2]["parts"] == [{"text": "User message."}]

    def test_empty_persistent_skipped(self):
        from save_your_tokens.adapters.gemini import GeminiAdapter

        adapter = GeminiAdapter()
        empty_block = ContextBlock(id="p0", layer=ContextLayer.PERSISTENT, content="")
        messages = adapter.format_context([empty_block], [], [])

        assert messages == []

    def test_multiple_persistent_joined(self):
        from save_your_tokens.adapters.gemini import GeminiAdapter

        adapter = GeminiAdapter()
        blocks = [
            self._make_block("p1", ContextLayer.PERSISTENT, "Part A."),
            self._make_block("p2", ContextLayer.PERSISTENT, "Part B."),
        ]
        messages = adapter.format_context(blocks, [], [])

        assert len(messages) == 1
        assert messages[0]["role"] == "user"
        assert "[System Instructions]" in messages[0]["parts"][0]["text"]
        assert "Part A." in messages[0]["parts"][0]["text"]
        assert "Part B." in messages[0]["parts"][0]["text"]


class TestGeminiModelCompact:
    def test_model_compact_calls_generate_content(self):
        from save_your_tokens.adapters.gemini import GeminiAdapter

        mock_genai = MagicMock()
        mock_client = MagicMock()
        mock_genai.Client.return_value = mock_client

        mock_response = MagicMock()
        mock_response.text = "Compacted summary."
        mock_client.models.generate_content.return_value = mock_response

        with patch("save_your_tokens.adapters.gemini.genai", mock_genai):
            adapter = GeminiAdapter(model="gemini-2.0-flash")
            result = adapter.model_compact("Long text to compact.", target_tokens=100)

        assert result == "Compacted summary."
        mock_client.models.generate_content.assert_called_once()

    def test_model_compact_uses_correct_model(self):
        from save_your_tokens.adapters.gemini import GeminiAdapter

        mock_genai = MagicMock()
        mock_client = MagicMock()
        mock_genai.Client.return_value = mock_client

        mock_response = MagicMock()
        mock_response.text = "Summary."
        mock_client.models.generate_content.return_value = mock_response

        with patch("save_your_tokens.adapters.gemini.genai", mock_genai):
            adapter = GeminiAdapter(model="gemini-2.5-pro")
            adapter.model_compact("Content.", target_tokens=50)

        call_kwargs = mock_client.models.generate_content.call_args[1]
        assert call_kwargs["model"] == "gemini-2.5-pro"

    def test_model_compact_passes_max_output_tokens_in_config(self):
        from save_your_tokens.adapters.gemini import GeminiAdapter

        mock_genai = MagicMock()
        mock_client = MagicMock()
        mock_genai.Client.return_value = mock_client

        mock_response = MagicMock()
        mock_response.text = "Summary."
        mock_client.models.generate_content.return_value = mock_response

        with patch("save_your_tokens.adapters.gemini.genai", mock_genai):
            adapter = GeminiAdapter()
            adapter.model_compact("Content.", target_tokens=200)

        call_kwargs = mock_client.models.generate_content.call_args[1]
        assert call_kwargs["config"]["max_output_tokens"] == 200

    def test_model_compact_returns_response_text(self):
        from save_your_tokens.adapters.gemini import GeminiAdapter

        mock_genai = MagicMock()
        mock_client = MagicMock()
        mock_genai.Client.return_value = mock_client

        mock_response = MagicMock()
        mock_response.text = "Final compacted text."
        mock_client.models.generate_content.return_value = mock_response

        with patch("save_your_tokens.adapters.gemini.genai", mock_genai):
            adapter = GeminiAdapter()
            result = adapter.model_compact("Original content.", target_tokens=50)

        assert result == "Final compacted text."

    def test_supports_native_compact_true(self):
        from save_your_tokens.adapters.gemini import GeminiAdapter

        adapter = GeminiAdapter()
        assert adapter.supports_native_compact is True
