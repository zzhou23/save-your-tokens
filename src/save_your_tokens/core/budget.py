"""Budget engine: allocate, check usage, recommend actions.

Responsibilities:
- Compute per-layer token budgets from a BudgetProfile + context window size
- Track current usage across layers
- Evaluate overage level (Q2: tiered response)
- Recommend actions when budget is exceeded
"""

from __future__ import annotations

from save_your_tokens.core.spec import (
    BudgetProfile,
    CompactAction,
    ContextBlock,
    ContextLayer,
    ContextUsage,
    OverageLevel,
)

# Tiered overage thresholds (Q2)
WARN_THRESHOLD = 0.05  # <5% over -> warn
COMPACT_THRESHOLD = 0.20  # 5-20% over -> auto compact
# >20% over -> reject


class BudgetEngine:
    """Manages token budget allocation and usage tracking."""

    def __init__(self, context_window: int, profile: BudgetProfile) -> None:
        self._context_window = context_window
        self._profile = profile
        self._blocks: dict[str, ContextBlock] = {}
        self._current_turn: int = 0

    @property
    def profile(self) -> BudgetProfile:
        return self._profile

    def compute_budgets(self) -> ContextUsage:
        """Compute per-layer budgets and current usage."""
        output_reserve = int(self._context_window * self._profile.output_reserve_pct)
        available = self._context_window - output_reserve

        persistent_budget = int(available * self._profile.persistent_pct)
        session_budget = int(available * self._profile.session_pct)
        ephemeral_budget = available - persistent_budget - session_budget

        persistent_used = sum(
            b.token_count for b in self._blocks.values() if b.layer == ContextLayer.PERSISTENT
        )
        session_used = sum(
            b.token_count for b in self._blocks.values() if b.layer == ContextLayer.SESSION
        )
        ephemeral_used = sum(
            b.token_count for b in self._blocks.values() if b.layer == ContextLayer.EPHEMERAL
        )

        return ContextUsage(
            context_window=self._context_window,
            output_reserve=output_reserve,
            persistent_used=persistent_used,
            persistent_budget=persistent_budget,
            session_used=session_used,
            session_budget=session_budget,
            ephemeral_used=ephemeral_used,
            ephemeral_budget=ephemeral_budget,
            current_turn=self._current_turn,
        )

    def check_overage(self, layer: ContextLayer) -> OverageLevel:
        """Check overage level for a specific layer (Q2: tiered response)."""
        usage = self.compute_budgets()

        match layer:
            case ContextLayer.PERSISTENT:
                used, budget = usage.persistent_used, usage.persistent_budget
            case ContextLayer.SESSION:
                used, budget = usage.session_used, usage.session_budget
            case ContextLayer.EPHEMERAL:
                used, budget = usage.ephemeral_used, usage.ephemeral_budget

        if budget == 0:
            return OverageLevel.REJECT if used > 0 else OverageLevel.WITHIN

        overage_ratio = (used - budget) / budget
        if overage_ratio <= 0:
            return OverageLevel.WITHIN
        if overage_ratio < WARN_THRESHOLD:
            return OverageLevel.WARN
        if overage_ratio < COMPACT_THRESHOLD:
            return OverageLevel.COMPACT
        return OverageLevel.REJECT

    def recommend_actions(self) -> list[CompactAction]:
        """Recommend compact actions based on current usage."""
        actions: list[CompactAction] = []
        usage = self.compute_budgets()

        # Global utilization check
        if usage.utilization > 0.8:
            actions.append(CompactAction.DROP_STALE_EPHEMERAL)

        # Per-layer checks
        if self.check_overage(ContextLayer.EPHEMERAL) in (
            OverageLevel.COMPACT,
            OverageLevel.REJECT,
        ):
            actions.append(CompactAction.SUMMARIZE_EPHEMERAL)

        if self.check_overage(ContextLayer.SESSION) in (
            OverageLevel.COMPACT,
            OverageLevel.REJECT,
        ):
            actions.append(CompactAction.COMPACT_SESSION)

        if self.check_overage(ContextLayer.PERSISTENT) == OverageLevel.REJECT:
            actions.append(CompactAction.FORCE_TRIM_PERSISTENT)

        return actions

    def add_block(self, block: ContextBlock) -> OverageLevel:
        """Add a context block and return the resulting overage level for its layer."""
        self._blocks[block.id] = block
        overage = self.check_overage(block.layer)
        if overage == OverageLevel.REJECT:
            del self._blocks[block.id]
        return overage

    def remove_block(self, block_id: str) -> ContextBlock | None:
        """Remove a context block by ID."""
        return self._blocks.pop(block_id, None)

    def get_blocks(self, layer: ContextLayer | None = None) -> list[ContextBlock]:
        """Get all blocks, optionally filtered by layer."""
        blocks = list(self._blocks.values())
        if layer is not None:
            blocks = [b for b in blocks if b.layer == layer]
        return blocks

    def advance_turn(self) -> None:
        """Advance the turn counter."""
        self._current_turn += 1

    def get_stale_blocks(self, max_age: int) -> list[ContextBlock]:
        """Get ephemeral blocks not referenced in the last `max_age` turns (Q3: turn counting)."""
        return [
            b
            for b in self._blocks.values()
            if b.layer == ContextLayer.EPHEMERAL
            and (self._current_turn - b.last_referenced_turn) > max_age
        ]
