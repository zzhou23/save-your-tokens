"""Raw SDK integration — thin wrapper for direct SDK usage.

Two-layer design:
- RawSDKIntegration(FrameworkIntegration): inner layer with standard lifecycle
- SYTWrapper: outer layer providing ergonomic prepare_context/on_response API
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from save_your_tokens.core.spec import ContextBlock, ContextLayer
from save_your_tokens.integrations.base import FrameworkIntegration

if TYPE_CHECKING:
    from save_your_tokens.adapters.base import ModelAdapter
    from save_your_tokens.core.budget import BudgetEngine
    from save_your_tokens.core.lifecycle import LifecycleManager
    from save_your_tokens.core.strategy import StrategyEngine


class RawSDKIntegration(FrameworkIntegration):
    """Inner layer: implements FrameworkIntegration for raw SDK usage."""

    def setup(self, config: dict[str, Any]) -> None:
        """No-op."""

    def teardown(self) -> None:
        """No-op."""

    def intercept_context(self, messages: list[dict[str, Any]]) -> list[dict[str, Any]]:
        """Register messages as ephemeral blocks and return them."""
        for i, msg in enumerate(messages):
            content = msg.get("content", "")
            block = ContextBlock(
                id=f"msg:{i}",
                layer=ContextLayer.EPHEMERAL,
                content=content if isinstance(content, str) else str(content),
                token_count=len(str(content)) // 4,
                source=f"message:{i}",
                metadata={"role": msg.get("role", "user")},
            )
            self._budget.remove_block(block.id)
            self._budget.add_block(block)
        return messages

    def on_response(self, response: dict[str, Any]) -> None:
        """Run post-turn lifecycle."""
        self.run_post_turn()


class SYTWrapper:
    """Ergonomic wrapper for raw SDK usage."""

    def __init__(
        self,
        adapter: ModelAdapter | None,
        budget_engine: BudgetEngine,
        lifecycle: LifecycleManager,
        strategy: StrategyEngine,
    ) -> None:
        self._adapter = adapter
        self._integration = RawSDKIntegration(budget_engine, lifecycle, strategy)

    def prepare_context(self, messages: list[dict[str, Any]]) -> list[dict[str, Any]]:
        """Classify, budget check, compact, and return ready-to-send messages."""
        return self._integration.intercept_context(messages)

    def on_response(self, response: dict[str, Any]) -> dict[str, Any]:
        """Post-turn lifecycle. Returns usage summary.

        on_response() already calls run_post_turn() internally,
        so we just return the usage snapshot.
        """
        self._integration.on_response(response)
        return {
            "turn": self._integration._lifecycle.current_turn,
            "usage": self._integration.get_usage().model_dump(),
        }
