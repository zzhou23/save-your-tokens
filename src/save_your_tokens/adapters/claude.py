"""Claude model adapter.

Uses the Anthropic SDK for token counting and supports native compaction
via Claude's summarization capability.
"""

from __future__ import annotations

from typing import Any

from save_your_tokens.adapters.base import ModelAdapter
from save_your_tokens.core.spec import ContextBlock

# Context window sizes for Claude models (as of March 2026)
CLAUDE_CONTEXT_WINDOWS: dict[str, int] = {
    "claude-opus-4-6": 200_000,
    "claude-sonnet-4-6": 200_000,
    "claude-haiku-4-5": 200_000,
}


class ClaudeAdapter(ModelAdapter):
    """Adapter for Anthropic Claude models."""

    def __init__(self, model: str = "claude-sonnet-4-6") -> None:
        self._model = model
        self._context_window = CLAUDE_CONTEXT_WINDOWS.get(model, 200_000)
        self._client: Any = None

    def _get_client(self) -> Any:
        if self._client is None:
            try:
                from anthropic import Anthropic

                self._client = Anthropic()
            except ImportError as e:
                raise ImportError(
                    "anthropic package required. Install with: pip install save-your-tokens[claude]"
                ) from e
        return self._client

    @property
    def model_name(self) -> str:
        return self._model

    @property
    def context_window(self) -> int:
        return self._context_window

    def count_tokens(self, text: str) -> int:
        """Count tokens using Anthropic's token counting API."""
        client = self._get_client()
        result = client.messages.count_tokens(
            model=self._model,
            messages=[{"role": "user", "content": text}],
        )
        return result.input_tokens

    def format_context(
        self,
        persistent: list[ContextBlock],
        session: list[ContextBlock],
        ephemeral: list[ContextBlock],
    ) -> list[dict[str, Any]]:
        """Format context for Claude's Messages API.

        Persistent -> system message
        Session + Ephemeral -> user/assistant messages
        """
        messages: list[dict[str, Any]] = []

        # Persistent context goes into system message
        persistent_text = "\n\n".join(b.content for b in persistent if b.content)
        if persistent_text:
            messages.append({"role": "system", "content": persistent_text})

        # Session context as a context-setting user message
        session_text = "\n\n".join(b.content for b in session if b.content)
        if session_text:
            messages.append(
                {
                    "role": "user",
                    "content": f"[Session Context]\n{session_text}",
                }
            )

        # Ephemeral content as messages
        for block in ephemeral:
            role = block.metadata.get("role", "user")
            messages.append({"role": role, "content": block.content})

        return messages

    def model_compact(self, content: str, target_tokens: int) -> str | None:
        """Use Claude to summarize content to target token count."""
        client = self._get_client()
        response = client.messages.create(
            model=self._model,
            max_tokens=target_tokens,
            messages=[
                {
                    "role": "user",
                    "content": (
                        f"Summarize the following content concisely, "
                        f"preserving key information, in under {target_tokens} tokens:\n\n{content}"
                    ),
                }
            ],
        )
        return response.content[0].text
