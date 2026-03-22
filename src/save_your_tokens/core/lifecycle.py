"""Lifecycle manager: session start/end, post-turn evaluation.

Manages the lifecycle of a context-managed LLM session:
- Session initialization (load persistent context, restore session state)
- Post-turn evaluation (update references, check budgets, trigger compaction)
- Session end (persist session state, generate report)
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any

from save_your_tokens.core.budget import BudgetEngine
from save_your_tokens.core.spec import (
    CompactAction,
    ContextBlock,
    ContextLayer,
    OverageLevel,
)


class SessionPhase(str, Enum):
    """Current phase of the session lifecycle."""

    INITIALIZING = "initializing"
    ACTIVE = "active"
    COMPACTING = "compacting"
    ENDING = "ending"


@dataclass
class TurnResult:
    """Result of post-turn evaluation."""

    turn_number: int
    overage_levels: dict[ContextLayer, OverageLevel] = field(default_factory=dict)
    recommended_actions: list[CompactAction] = field(default_factory=list)
    stale_block_ids: list[str] = field(default_factory=list)

    @property
    def needs_compaction(self) -> bool:
        return len(self.recommended_actions) > 0


class LifecycleManager:
    """Manages the lifecycle of a context-managed session."""

    def __init__(self, budget_engine: BudgetEngine) -> None:
        self._engine = budget_engine
        self._phase = SessionPhase.INITIALIZING
        self._turn_history: list[TurnResult] = []
        self._stale_max_age: int = 10  # Default: 10 turns (Q3)
        self._compact_interval: int = 0  # 0 = disabled

    @property
    def phase(self) -> SessionPhase:
        return self._phase

    @property
    def current_turn(self) -> int:
        return len(self._turn_history)

    def configure(
        self,
        stale_max_age: int | None = None,
        compact_interval: int | None = None,
    ) -> None:
        """Configure lifecycle parameters."""
        if stale_max_age is not None:
            self._stale_max_age = stale_max_age
        if compact_interval is not None:
            self._compact_interval = compact_interval

    def start_session(self, persistent_blocks: list[ContextBlock] | None = None) -> None:
        """Initialize session: load persistent context."""
        self._phase = SessionPhase.ACTIVE
        if persistent_blocks:
            for block in persistent_blocks:
                self._engine.add_block(block)

    def post_turn(self, referenced_block_ids: list[str] | None = None) -> TurnResult:
        """Evaluate state after a turn completes.

        Args:
            referenced_block_ids: IDs of blocks referenced in this turn,
                                  used to update staleness tracking.
        """
        self._engine.advance_turn()
        turn_num = self.current_turn

        # Update reference timestamps
        if referenced_block_ids:
            for block in self._engine.get_blocks():
                if block.id in referenced_block_ids:
                    block.last_referenced_turn = turn_num

        # Check overage levels per layer
        overage_levels = {
            layer: self._engine.check_overage(layer) for layer in ContextLayer
        }

        # Find stale blocks (Q3: pure turn counting)
        stale = self._engine.get_stale_blocks(self._stale_max_age)
        stale_ids = [b.id for b in stale]

        # Get recommended actions
        actions = self._engine.recommend_actions()

        # Check interval-based compaction
        if self._compact_interval > 0 and turn_num % self._compact_interval == 0:
            if CompactAction.DROP_STALE_EPHEMERAL not in actions:
                actions.insert(0, CompactAction.DROP_STALE_EPHEMERAL)

        result = TurnResult(
            turn_number=turn_num,
            overage_levels=overage_levels,
            recommended_actions=actions,
            stale_block_ids=stale_ids,
        )
        self._turn_history.append(result)
        return result

    def end_session(self) -> dict[str, Any]:
        """End the session and return summary stats."""
        self._phase = SessionPhase.ENDING
        usage = self._engine.compute_budgets()

        return {
            "total_turns": self.current_turn,
            "final_usage": usage.model_dump(),
            "compaction_events": sum(1 for t in self._turn_history if t.needs_compaction),
            "total_stale_detected": sum(len(t.stale_block_ids) for t in self._turn_history),
        }
