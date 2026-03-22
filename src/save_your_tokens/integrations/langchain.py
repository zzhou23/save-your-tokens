"""LangChain LCEL integration.

Two-layer: LangChainIntegration(FrameworkIntegration) inner + SYTRunnable outer.
"""

from __future__ import annotations

from typing import Any

from save_your_tokens.core.spec import ContextBlock, ContextLayer
from save_your_tokens.integrations.base import FrameworkIntegration

try:
    from langchain_core.runnables import RunnableSerializable
except ImportError:
    RunnableSerializable = object  # type: ignore[assignment, misc]


class LangChainIntegration(FrameworkIntegration):
    """Inner layer: implements FrameworkIntegration for LangChain."""

    def setup(self, config: dict[str, Any]) -> None:
        """No-op."""

    def teardown(self) -> None:
        """No-op."""

    def intercept_context(self, messages: list[dict[str, Any]]) -> list[dict[str, Any]]:
        """Register messages as ephemeral blocks, run budget check."""
        for i, msg in enumerate(messages):
            content = msg.get("content", "")
            block = ContextBlock(
                id=f"lc-msg:{i}",
                layer=ContextLayer.EPHEMERAL,
                content=content if isinstance(content, str) else str(content),
                token_count=len(str(content)) // 4,
                source=f"langchain:{i}",
                metadata={"role": msg.get("role", "user")},
            )
            self._budget.remove_block(block.id)
            self._budget.add_block(block)
        return messages

    def on_response(self, response: dict[str, Any]) -> None:
        """Run post-turn lifecycle."""
        self.run_post_turn()


class SYTRunnable(RunnableSerializable):
    """LCEL-native Runnable that manages context budget in the chain."""

    def __init__(
        self,
        budget_engine: Any,
        lifecycle: Any,
        strategy: Any,
        **kwargs: Any,
    ) -> None:
        self._integration = LangChainIntegration(budget_engine, lifecycle, strategy)

    def invoke(self, input: Any, config: Any = None) -> dict[str, Any]:
        """Intercept context, run budget management, return processed input."""
        messages = input.get("messages", []) if isinstance(input, dict) else []
        processed = self._integration.intercept_context(messages)
        return {"messages": processed}

    async def ainvoke(self, input: Any, config: Any = None) -> dict[str, Any]:
        """Async version — delegates to sync invoke."""
        return self.invoke(input, config)
