"""Compression wrappers: extractive summarization utilities.

Wraps community summarization tools behind a common interface.
Phase 1: simple extractive methods (sentence scoring).
Phase 2+: plug in LLM-based or advanced summarizers.
"""

from __future__ import annotations

import re
from abc import ABC, abstractmethod


class Compressor(ABC):
    """Abstract compressor interface."""

    @abstractmethod
    def compress(self, text: str, target_ratio: float) -> str:
        """Compress text to approximately target_ratio of original length.

        Args:
            text: Input text to compress.
            target_ratio: Target compression ratio (0.0 to 1.0). E.g., 0.3 = 30% of original.

        Returns:
            Compressed text.
        """
        ...


class ExtractiveCompressor(Compressor):
    """Simple extractive compressor: keeps the most important sentences.

    Uses sentence position and length as heuristics for importance.
    """

    def compress(self, text: str, target_ratio: float) -> str:
        if target_ratio >= 1.0:
            return text

        sentences = self._split_sentences(text)
        if len(sentences) <= 1:
            target_chars = int(len(text) * target_ratio)
            return text[:target_chars]

        # Score sentences: first/last sentences score higher, longer sentences score higher
        scored = []
        for i, sentence in enumerate(sentences):
            score = len(sentence.split())  # Word count as base score
            if i == 0:
                score *= 2.0  # First sentence bonus
            elif i == len(sentences) - 1:
                score *= 1.5  # Last sentence bonus
            scored.append((score, i, sentence))

        # Sort by score descending, take top N to meet target ratio
        scored.sort(key=lambda x: x[0], reverse=True)
        target_len = int(len(text) * target_ratio)

        selected: list[tuple[int, str]] = []
        current_len = 0
        for _, idx, sentence in scored:
            if current_len + len(sentence) > target_len:
                break
            selected.append((idx, sentence))
            current_len += len(sentence)

        # Restore original order
        selected.sort(key=lambda x: x[0])
        return " ".join(s for _, s in selected)

    @staticmethod
    def _split_sentences(text: str) -> list[str]:
        """Split text into sentences using simple regex."""
        sentences = re.split(r"(?<=[.!?])\s+", text.strip())
        return [s for s in sentences if s.strip()]


class TruncationCompressor(Compressor):
    """Simple truncation with marker."""

    def compress(self, text: str, target_ratio: float) -> str:
        target_chars = int(len(text) * target_ratio)
        if len(text) <= target_chars:
            return text
        return text[:target_chars] + "\n[... truncated ...]"
