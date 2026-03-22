"""OpenAI model adapter.

Uses tiktoken for token counting.
"""

from __future__ import annotations

from typing import Any

from save_your_tokens.adapters.base import ModelAdapter
from save_your_tokens.core.spec import ContextBlock

OPENAI_CONTEXT_WINDOWS: dict[str, int] = {
    "gpt-4o": 128_000,
    "gpt-4o-mini": 128_000,
    "gpt-4-turbo": 128_000,
    "gpt-4": 8_192,
    "o1": 200_000,
    "o3": 200_000,
    "o3-mini": 200_000,
}


class OpenAIAdapter(ModelAdapter):
    """Adapter for OpenAI models."""

    def __init__(self, model: str = "gpt-4o") -> None:
        self._model = model
        self._context_window = OPENAI_CONTEXT_WINDOWS.get(model, 128_000)
        self._encoding: Any = None

    def _get_encoding(self) -> Any:
        if self._encoding is None:
            try:
                import tiktoken

                self._encoding = tiktoken.encoding_for_model(self._model)
            except ImportError as e:
                raise ImportError(
                    "tiktoken package required. Install with: pip install save-your-tokens[openai]"
                ) from e
        return self._encoding

    @property
    def model_name(self) -> str:
        return self._model

    @property
    def context_window(self) -> int:
        return self._context_window

    def count_tokens(self, text: str) -> int:
        """Count tokens using tiktoken."""
        encoding = self._get_encoding()
        return len(encoding.encode(text))

    def format_context(
        self,
        persistent: list[ContextBlock],
        session: list[ContextBlock],
        ephemeral: list[ContextBlock],
    ) -> list[dict[str, Any]]:
        """Format context for OpenAI's Chat Completions API."""
        messages: list[dict[str, Any]] = []

        # Persistent -> system message
        persistent_text = "\n\n".join(b.content for b in persistent if b.content)
        if persistent_text:
            messages.append({"role": "system", "content": persistent_text})

        # Session -> system message (developer context)
        session_text = "\n\n".join(b.content for b in session if b.content)
        if session_text:
            messages.append({"role": "system", "content": f"[Session Context]\n{session_text}"})

        # Ephemeral -> user/assistant messages
        for block in ephemeral:
            role = block.metadata.get("role", "user")
            messages.append({"role": role, "content": block.content})

        return messages
