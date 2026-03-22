"""Abstract FrameworkIntegration interface.

Q5 decision: two-layer design.
- Inner layer: unified context logic (classify, budget check, compact) — shared across frameworks.
- Outer layer: framework-specific glue code (hooks, middleware, etc.) — each integration implements.

This module defines the inner layer interface.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

from save_your_tokens.core.budget import BudgetEngine
from save_your_tokens.core.lifecycle import LifecycleManager
from save_your_tokens.core.spec import CompactAction, ContextUsage
from save_your_tokens.core.strategy import StrategyEngine


class FrameworkIntegration(ABC):
    """Base class for framework integrations.

    Subclasses implement the outer glue layer for specific frameworks.
    The inner context logic (budget check, compact orchestration) is
    provided by this base class via the shared engines.
    """

    def __init__(
        self,
        budget_engine: BudgetEngine,
        lifecycle: LifecycleManager,
        strategy: StrategyEngine,
    ) -> None:
        self._budget = budget_engine
        self._lifecycle = lifecycle
        self._strategy = strategy

    # --- Inner layer: shared context logic ---

    def get_usage(self) -> ContextUsage:
        """Get current context usage snapshot."""
        return self._budget.compute_budgets()

    def run_post_turn(self, referenced_block_ids: list[str] | None = None) -> dict[str, Any]:
        """Run post-turn evaluation and auto-compact if needed.

        Returns a summary of actions taken.
        """
        result = self._lifecycle.post_turn(referenced_block_ids)
        actions_taken: dict[CompactAction, list[str]] = {}

        if result.needs_compaction:
            actions_taken = self._strategy.execute_actions(result.recommended_actions)

        return {
            "turn": result.turn_number,
            "needs_compaction": result.needs_compaction,
            "actions_taken": {k.value: v for k, v in actions_taken.items()},
            "stale_blocks": result.stale_block_ids,
        }

    # --- Outer layer: framework-specific glue (subclass implements) ---

    @abstractmethod
    def setup(self, config: dict[str, Any]) -> None:
        """Set up the integration (install hooks, register middleware, etc.)."""
        ...

    @abstractmethod
    def teardown(self) -> None:
        """Clean up the integration."""
        ...

    @abstractmethod
    def intercept_context(self, messages: list[dict[str, Any]]) -> list[dict[str, Any]]:
        """Intercept and transform context before sending to the model.

        This is the main hook point. Each framework calls this differently:
        - Claude Code: via settings.json hooks
        - LangChain: via middleware chain
        - Raw SDK: direct function call
        """
        ...

    @abstractmethod
    def on_response(self, response: dict[str, Any]) -> None:
        """Handle model response (update usage tracking, etc.)."""
        ...
