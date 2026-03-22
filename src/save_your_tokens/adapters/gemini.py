"""Gemini model adapter.

Uses the google-genai SDK for token counting, context formatting, and compaction.
"""

from __future__ import annotations

from typing import Any

try:
    from google import genai
except ImportError:
    genai = None  # type: ignore[assignment]

from save_your_tokens.adapters.base import ModelAdapter
from save_your_tokens.core.spec import ContextBlock

GEMINI_CONTEXT_WINDOWS: dict[str, int] = {
    "gemini-2.0-flash": 1_000_000,
    "gemini-2.5-pro": 1_000_000,
}

_DEFAULT_CONTEXT_WINDOW = 1_000_000


class GeminiAdapter(ModelAdapter):
    """Adapter for Gemini models via the google-genai SDK."""

    def __init__(
        self,
        model: str = "gemini-2.0-flash",
        api_key: str | None = None,
    ) -> None:
        self._model = model
        self._context_window = GEMINI_CONTEXT_WINDOWS.get(model, _DEFAULT_CONTEXT_WINDOW)
        self._api_key = api_key
        self._client: Any = None

    @property
    def model_name(self) -> str:
        return self._model

    @property
    def context_window(self) -> int:
        return self._context_window

    @property
    def recommended_output_reserve(self) -> int:
        """Override: Gemini has large windows, reserve only 10%."""
        return int(self._context_window * 0.1)

    def _get_client(self) -> Any:
        if genai is None:
            raise ImportError(
                "google-genai package required. Install with: pip install save-your-tokens[gemini]"
            )
        if self._client is None:
            if self._api_key is not None:
                self._client = genai.Client(api_key=self._api_key)
            else:
                self._client = genai.Client()
        return self._client

    def count_tokens(self, text: str) -> int:
        """Count tokens using the Gemini API."""
        client = self._get_client()
        result = client.models.count_tokens(model=self._model, contents=text)
        return result.total_tokens

    def format_context(
        self,
        persistent: list[ContextBlock],
        session: list[ContextBlock],
        ephemeral: list[ContextBlock],
    ) -> list[dict[str, Any]]:
        """Format context for Gemini's native contents/parts structure."""
        messages: list[dict[str, Any]] = []

        # Persistent -> system role
        persistent_text = "\n\n".join(b.content for b in persistent if b.content)
        if persistent_text:
            messages.append({"role": "system", "parts": [{"text": persistent_text}]})

        # Session -> user role with prefix
        session_text = "\n\n".join(b.content for b in session if b.content)
        if session_text:
            messages.append(
                {"role": "user", "parts": [{"text": f"[Session Context]\n{session_text}"}]}
            )

        # Ephemeral -> user/model messages ("assistant" mapped to "model")
        for block in ephemeral:
            raw_role = block.metadata.get("role", "user")
            role = "model" if raw_role == "assistant" else raw_role
            messages.append({"role": role, "parts": [{"text": block.content}]})

        return messages

    def model_compact(self, content: str, target_tokens: int) -> str | None:
        """Use Gemini's API to compact content to approximately target_tokens."""
        client = self._get_client()
        response = client.models.generate_content(
            model=self._model,
            contents=content,
            config={"max_output_tokens": target_tokens},
        )
        return response.text
