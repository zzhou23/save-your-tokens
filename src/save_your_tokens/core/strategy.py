"""Strategy engine: compact/clear trigger rules and actions.

Decides WHEN to compact (triggers) and orchestrates HOW (actions).
Q4 decision: strategy owns the compact flow, adapter provides optional model_compact().
Phase 1: rule-based. Phase 2+: LLM-assisted decisions.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from save_your_tokens.core.spec import (
    CompactAction,
    ContextBlock,
    ContextLayer,
)
from save_your_tokens.reuse.compactor import Compactor, DefaultCompactor

if TYPE_CHECKING:
    from save_your_tokens.core.budget import BudgetEngine


class StrategyEngine:
    """Orchestrates compaction actions decided by the budget engine."""

    def __init__(
        self,
        budget_engine: BudgetEngine,
        compactor: Compactor | None = None,
        observer: Any = None,
    ) -> None:
        self._engine = budget_engine
        self._compactor = compactor or DefaultCompactor()
        if observer is None:
            from save_your_tokens.reuse.observability import NoOpObserver

            observer = NoOpObserver()
        self._observer = observer

    def execute_action(self, action: CompactAction) -> list[str]:
        """Execute a single compact action. Returns IDs of affected blocks."""
        match action:
            case CompactAction.DROP_STALE_EPHEMERAL:
                result = self._drop_stale_ephemeral()
            case CompactAction.SUMMARIZE_EPHEMERAL:
                result = self._summarize_layer(ContextLayer.EPHEMERAL)
            case CompactAction.COMPACT_SESSION:
                result = self._summarize_layer(ContextLayer.SESSION)
            case CompactAction.FORCE_TRIM_PERSISTENT:
                result = self._force_trim_persistent()
        self._observer.track_usage(
            {
                "type": "compact_action",
                "action": action.value,
                "affected": len(result),
            }
        )
        return result

    def execute_actions(self, actions: list[CompactAction]) -> dict[CompactAction, list[str]]:
        """Execute a list of actions in order. Returns mapping of action -> affected block IDs."""
        results: dict[CompactAction, list[str]] = {}
        for action in actions:
            results[action] = self.execute_action(action)
        return results

    def _drop_stale_ephemeral(self) -> list[str]:
        """Drop ephemeral blocks that haven't been referenced recently."""
        stale = self._engine.get_stale_blocks(max_age=10)
        dropped_ids: list[str] = []
        for block in stale:
            removed = self._engine.remove_block(block.id)
            if removed:
                dropped_ids.append(block.id)
        return dropped_ids

    def _summarize_layer(self, layer: ContextLayer) -> list[str]:
        """Compact blocks in a layer by summarizing their content."""
        blocks = self._engine.get_blocks(layer)
        if not blocks:
            return []

        affected_ids: list[str] = []
        for block in blocks:
            usage = self._engine.compute_budgets()
            match layer:
                case ContextLayer.EPHEMERAL:
                    target = usage.ephemeral_budget
                case ContextLayer.SESSION:
                    target = usage.session_budget
                case _:
                    continue

            # Target per block: proportional share of layer budget
            per_block_target = target // max(len(blocks), 1)
            if block.token_count > per_block_target:
                compacted = self._compactor.compact(block.content, per_block_target)
                updated_block = ContextBlock(
                    id=block.id,
                    layer=block.layer,
                    content=compacted,
                    token_count=per_block_target,
                    source=block.source,
                    tags=block.tags,
                    last_referenced_turn=block.last_referenced_turn,
                    metadata={**block.metadata, "compacted": True},
                )
                self._engine.remove_block(block.id)
                self._engine.add_block(updated_block)
                affected_ids.append(block.id)

        return affected_ids

    def _force_trim_persistent(self) -> list[str]:
        """Last resort: trim persistent layer content."""
        blocks = self._engine.get_blocks(ContextLayer.PERSISTENT)
        if not blocks:
            return []

        # Sort by metadata priority (lower = less important), trim from least important
        sorted_blocks = sorted(blocks, key=lambda b: b.metadata.get("priority", 50))

        affected_ids: list[str] = []
        usage = self._engine.compute_budgets()

        while usage.persistent_used > usage.persistent_budget and sorted_blocks:
            block = sorted_blocks.pop(0)
            self._engine.remove_block(block.id)
            affected_ids.append(block.id)
            usage = self._engine.compute_budgets()

        return affected_ids
