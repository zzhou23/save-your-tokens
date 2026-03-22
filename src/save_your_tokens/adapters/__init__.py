"""Model adapters: token counting, context formatting, optional compaction."""

from save_your_tokens.adapters.base import ModelAdapter

__all__ = [
    "ModelAdapter",
    "ClaudeAdapter",
    "OpenAIAdapter",
    "DeepSeekAdapter",
    "GeminiAdapter",
]


def __getattr__(name: str):
    """Lazy imports to avoid requiring all SDKs."""
    if name == "ClaudeAdapter":
        from save_your_tokens.adapters.claude import ClaudeAdapter

        return ClaudeAdapter
    if name == "OpenAIAdapter":
        from save_your_tokens.adapters.openai import OpenAIAdapter

        return OpenAIAdapter
    if name == "DeepSeekAdapter":
        from save_your_tokens.adapters.deepseek import DeepSeekAdapter

        return DeepSeekAdapter
    if name == "GeminiAdapter":
        from save_your_tokens.adapters.gemini import GeminiAdapter

        return GeminiAdapter
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
