"""DeepSeek model adapter.

Uses tiktoken cl100k_base for token counting and the OpenAI-compatible SDK
for API calls (DeepSeek exposes an OpenAI-compatible endpoint).
"""

from __future__ import annotations

from typing import Any

try:
    import tiktoken
except ImportError:
    tiktoken = None  # type: ignore[assignment]

try:
    from openai import OpenAI
except ImportError:
    OpenAI = None  # type: ignore[assignment,misc]

from save_your_tokens.adapters.base import ModelAdapter
from save_your_tokens.core.spec import ContextBlock

DEEPSEEK_BASE_URL = "https://api.deepseek.com"

DEEPSEEK_CONTEXT_WINDOWS: dict[str, int] = {
    "deepseek-chat": 64_000,
    "deepseek-coder": 128_000,
    "deepseek-reasoner": 64_000,
}


class DeepSeekAdapter(ModelAdapter):
    """Adapter for DeepSeek models via the OpenAI-compatible API."""

    def __init__(self, model: str = "deepseek-chat", base_url: str = DEEPSEEK_BASE_URL) -> None:
        self._model = model
        self._context_window = DEEPSEEK_CONTEXT_WINDOWS.get(model, 64_000)
        self._base_url = base_url
        self._encoding: Any = None
        self._client: Any = None

    @property
    def model_name(self) -> str:
        return self._model

    @property
    def context_window(self) -> int:
        return self._context_window

    def _get_encoding(self) -> Any:
        if tiktoken is None:
            raise ImportError(
                "tiktoken package required. Install with: pip install save-your-tokens[deepseek]"
            )
        if self._encoding is None:
            self._encoding = tiktoken.get_encoding("cl100k_base")
        return self._encoding

    def _get_client(self) -> Any:
        if OpenAI is None:
            raise ImportError(
                "openai package required. Install with: pip install save-your-tokens[deepseek]"
            )
        if self._client is None:
            self._client = OpenAI(base_url=self._base_url)
        return self._client

    def count_tokens(self, text: str) -> int:
        """Count tokens using tiktoken cl100k_base encoding."""
        encoding = self._get_encoding()
        return len(encoding.encode(text))

    def format_context(
        self,
        persistent: list[ContextBlock],
        session: list[ContextBlock],
        ephemeral: list[ContextBlock],
    ) -> list[dict[str, Any]]:
        """Format context for DeepSeek's OpenAI-compatible Chat Completions API."""
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

    def model_compact(self, content: str, target_tokens: int) -> str | None:
        """Use DeepSeek's API to compact content to approximately target_tokens."""
        client = self._get_client()
        response = client.chat.completions.create(
            model=self._model,
            messages=[
                {
                    "role": "system",
                    "content": (
                        f"Summarize the following content concisely, targeting approximately "
                        f"{target_tokens} tokens. Preserve key information and structure."
                    ),
                },
                {"role": "user", "content": content},
            ],
        )
        return str(response.choices[0].message.content)
