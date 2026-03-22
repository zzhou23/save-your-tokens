"""Observability: optional Langfuse integration for tracking token usage.

Phase 2 feature. This module provides the interface and a no-op default.
"""

from __future__ import annotations

from typing import Any, Protocol


class Observer(Protocol):
    """Protocol for observability backends."""

    def track_usage(self, event: dict[str, Any]) -> None: ...
    def flush(self) -> None: ...


class NoOpObserver:
    """Default observer that does nothing."""

    def track_usage(self, event: dict[str, Any]) -> None:
        pass

    def flush(self) -> None:
        pass


def create_observer(backend: str = "noop", **kwargs: Any) -> Observer:
    """Factory for creating observers.

    Args:
        backend: "noop" (default) or "langfuse".
        **kwargs: Backend-specific configuration.
    """
    if backend == "langfuse":
        try:
            from langfuse import Langfuse

            client = Langfuse(**kwargs)
            return LangfuseObserver(client)
        except ImportError as e:
            raise ImportError(
                "langfuse package required. Install with: "
                "pip install save-your-tokens[langfuse]"
            ) from e
    return NoOpObserver()


class LangfuseObserver:
    """Langfuse-based observer for token usage tracking."""

    def __init__(self, client: Any) -> None:
        self._client = client

    def track_usage(self, event: dict[str, Any]) -> None:
        self._client.trace(name="syt-usage", metadata=event)

    def flush(self) -> None:
        self._client.flush()
