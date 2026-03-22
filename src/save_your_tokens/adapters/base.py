"""Abstract ModelAdapter interface.

Q4 decision: adapter provides count_tokens + format_context as required methods,
and an optional model_compact() for models with native summarization capability.
The strategy engine owns the compact flow and calls model_compact() if available.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

from save_your_tokens.core.spec import ContextBlock


class ModelAdapter(ABC):
    """Base class for model-specific adapters.

    Every adapter MUST implement:
    - count_tokens: accurate token counting for the model
    - format_context: assemble context blocks into model-native message format

    Optionally implement:
    - model_compact: use the model's native capability for content compaction
    """

    @property
    @abstractmethod
    def model_name(self) -> str:
        """Human-readable model identifier."""
        ...

    @property
    @abstractmethod
    def context_window(self) -> int:
        """Maximum context window size in tokens."""
        ...

    @property
    def recommended_output_reserve(self) -> int:
        """Recommended output token reservation. Override per model."""
        return int(self.context_window * 0.2)

    @abstractmethod
    def count_tokens(self, text: str) -> int:
        """Count tokens in the given text using the model's tokenizer."""
        ...

    @abstractmethod
    def format_context(
        self,
        persistent: list[ContextBlock],
        session: list[ContextBlock],
        ephemeral: list[ContextBlock],
    ) -> list[dict[str, Any]]:
        """Assemble context blocks into model-native message format.

        Returns a list of message dicts (e.g. [{"role": "system", "content": "..."}]).
        """
        ...

    def model_compact(self, content: str, target_tokens: int) -> str | None:
        """Use model's native capability for compaction (Q4: optional).

        Returns compacted content, or None if this adapter doesn't support
        native compaction (in which case strategy engine uses the default compactor).
        """
        return None

    @property
    def supports_native_compact(self) -> bool:
        """Whether this adapter supports native model compaction."""
        return type(self).model_compact is not ModelAdapter.model_compact
