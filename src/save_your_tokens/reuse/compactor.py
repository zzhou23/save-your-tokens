"""Token-based compaction interface.

Higher-level than compression.py's ratio-based Compressor ABC.
Compactors accept target_tokens and handle the token-to-ratio conversion internally.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import TYPE_CHECKING, Any, Protocol

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
    """Uses a ModelAdapter's native compaction. Raises NotImplementedError if unsupported."""

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
    """Uses a local model (Ollama/vLLM) via OpenAI-compatible HTTP API."""

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
    """Factory for compactor backends."""
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
