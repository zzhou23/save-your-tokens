"""Tokenizer wrappers: unified interface over tiktoken + native SDK tokenizers."""

from __future__ import annotations

from typing import Any


class TokenCounter:
    """Unified token counting interface.

    Wraps tiktoken (OpenAI) and native SDK tokenizers behind a common API.
    Falls back to character-based estimation when no tokenizer is available.
    """

    def __init__(self, backend: str = "estimate") -> None:
        self._backend = backend
        self._encoder: Any = None

    @classmethod
    def for_model(cls, model: str) -> TokenCounter:
        """Create a TokenCounter appropriate for the given model."""
        if model.startswith(("gpt-", "o1", "o3")):
            return cls(backend="tiktoken")
        if model.startswith("claude"):
            return cls(backend="anthropic")
        return cls(backend="estimate")

    def count(self, text: str) -> int:
        """Count tokens in the given text."""
        match self._backend:
            case "tiktoken":
                return self._count_tiktoken(text)
            case "anthropic":
                return self._count_estimate(text)  # Anthropic SDK counting needs API call
            case _:
                return self._count_estimate(text)

    def _count_tiktoken(self, text: str) -> int:
        if self._encoder is None:
            try:
                import tiktoken

                self._encoder = tiktoken.get_encoding("cl100k_base")
            except ImportError:
                return self._count_estimate(text)
        return len(self._encoder.encode(text))

    @staticmethod
    def _count_estimate(text: str) -> int:
        """Rough estimation: ~4 characters per token for English."""
        return max(1, len(text) // 4)
