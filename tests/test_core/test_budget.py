"""Tests for save_your_tokens.core.budget — budget allocation and overage."""

from save_your_tokens.core.budget import BudgetEngine
from save_your_tokens.core.spec import (
    PROFILE_AGENTIC,
    BudgetProfile,
    CompactAction,
    ContextBlock,
    ContextLayer,
    OverageLevel,
)


def _make_engine(window: int = 200_000) -> BudgetEngine:
    return BudgetEngine(context_window=window, profile=PROFILE_AGENTIC)


def _make_block(
    id: str,
    layer: ContextLayer,
    tokens: int,
    last_ref: int = 0,
) -> ContextBlock:
    return ContextBlock(
        id=id,
        layer=layer,
        content="x" * tokens,
        token_count=tokens,
        last_referenced_turn=last_ref,
    )


class TestComputeBudgets:
    def test_budget_allocation_agentic(self):
        engine = _make_engine(200_000)
        usage = engine.compute_budgets()
        # output_reserve = 200k * 0.20 = 40k
        assert usage.output_reserve == 40_000
        # available = 160k
        # persistent = 160k * 0.15 = 24k
        assert usage.persistent_budget == 24_000
        # session = 160k * 0.35 = 56k
        assert usage.session_budget == 56_000
        # ephemeral = 160k - 24k - 56k = 80k
        assert usage.ephemeral_budget == 80_000

    def test_empty_engine_zero_usage(self):
        engine = _make_engine()
        usage = engine.compute_budgets()
        assert usage.persistent_used == 0
        assert usage.session_used == 0
        assert usage.ephemeral_used == 0
        assert usage.total_used == 0

    def test_usage_tracks_blocks(self):
        engine = _make_engine()
        engine.add_block(_make_block("p1", ContextLayer.PERSISTENT, 1000))
        engine.add_block(_make_block("s1", ContextLayer.SESSION, 2000))
        engine.add_block(_make_block("e1", ContextLayer.EPHEMERAL, 3000))
        usage = engine.compute_budgets()
        assert usage.persistent_used == 1000
        assert usage.session_used == 2000
        assert usage.ephemeral_used == 3000

    def test_turn_counter(self):
        engine = _make_engine()
        engine.advance_turn()
        engine.advance_turn()
        usage = engine.compute_budgets()
        assert usage.current_turn == 2


class TestCheckOverage:
    def test_within_budget(self):
        engine = _make_engine()
        engine.add_block(_make_block("p1", ContextLayer.PERSISTENT, 1000))
        assert engine.check_overage(ContextLayer.PERSISTENT) == OverageLevel.WITHIN

    def test_warn_level(self):
        engine = _make_engine()
        # persistent_budget=24000, warn < 5% over -> up to 25199
        level = engine.add_block(_make_block("p1", ContextLayer.PERSISTENT, 24_500))
        assert level == OverageLevel.WARN
        assert engine.check_overage(ContextLayer.PERSISTENT) == OverageLevel.WARN

    def test_compact_level(self):
        engine = _make_engine()
        # 5-20% over: 25200..28799
        level = engine.add_block(_make_block("p1", ContextLayer.PERSISTENT, 26_000))
        assert level == OverageLevel.COMPACT

    def test_reject_level(self):
        engine = _make_engine()
        # >20% over: 28800+. Bypass via _blocks to verify check_overage logic.
        engine._blocks["p1"] = _make_block("p1", ContextLayer.PERSISTENT, 30_000)
        assert engine.check_overage(ContextLayer.PERSISTENT) == OverageLevel.REJECT

    def test_zero_budget_with_usage_rejects(self):
        """If budget is 0 but there's usage, should reject."""
        zero_profile = BudgetProfile(
            name="zero_persistent",
            persistent_pct=0.0,
            session_pct=0.5,
            output_reserve_pct=0.5,
        )
        engine = BudgetEngine(context_window=100_000, profile=zero_profile)
        # add_block auto-removes on REJECT, so bypass it
        engine._blocks["p1"] = _make_block("p1", ContextLayer.PERSISTENT, 100)
        assert engine.check_overage(ContextLayer.PERSISTENT) == OverageLevel.REJECT

    def test_zero_budget_zero_usage_within(self):
        zero_profile = BudgetProfile(
            name="zero_persistent",
            persistent_pct=0.0,
            session_pct=0.5,
            output_reserve_pct=0.5,
        )
        engine = BudgetEngine(context_window=100_000, profile=zero_profile)
        assert engine.check_overage(ContextLayer.PERSISTENT) == OverageLevel.WITHIN


class TestAddRemoveBlock:
    def test_add_block_returns_overage(self):
        engine = _make_engine()
        level = engine.add_block(_make_block("p1", ContextLayer.PERSISTENT, 1000))
        assert level == OverageLevel.WITHIN

    def test_add_block_reject_removes_block(self):
        engine = _make_engine()
        # persistent_budget=24000, >20% over => reject
        level = engine.add_block(_make_block("p1", ContextLayer.PERSISTENT, 30_000))
        assert level == OverageLevel.REJECT
        assert engine.get_blocks(ContextLayer.PERSISTENT) == []

    def test_remove_block(self):
        engine = _make_engine()
        engine.add_block(_make_block("p1", ContextLayer.PERSISTENT, 1000))
        removed = engine.remove_block("p1")
        assert removed is not None
        assert removed.id == "p1"

    def test_remove_nonexistent_returns_none(self):
        engine = _make_engine()
        assert engine.remove_block("nope") is None


class TestGetBlocks:
    def test_get_all_blocks(self):
        engine = _make_engine()
        engine.add_block(_make_block("p1", ContextLayer.PERSISTENT, 100))
        engine.add_block(_make_block("s1", ContextLayer.SESSION, 200))
        assert len(engine.get_blocks()) == 2

    def test_filter_by_layer(self):
        engine = _make_engine()
        engine.add_block(_make_block("p1", ContextLayer.PERSISTENT, 100))
        engine.add_block(_make_block("s1", ContextLayer.SESSION, 200))
        persistent = engine.get_blocks(ContextLayer.PERSISTENT)
        assert len(persistent) == 1
        assert persistent[0].id == "p1"


class TestStaleBlocks:
    def test_detect_stale_blocks(self):
        engine = _make_engine()
        engine.add_block(_make_block("e1", ContextLayer.EPHEMERAL, 100, last_ref=0))
        for _ in range(15):
            engine.advance_turn()
        stale = engine.get_stale_blocks(max_age=10)
        assert len(stale) == 1
        assert stale[0].id == "e1"

    def test_non_stale_not_returned(self):
        engine = _make_engine()
        engine.add_block(_make_block("e1", ContextLayer.EPHEMERAL, 100, last_ref=5))
        for _ in range(10):
            engine.advance_turn()
        stale = engine.get_stale_blocks(max_age=10)
        assert len(stale) == 0

    def test_only_ephemeral_checked(self):
        engine = _make_engine()
        engine.add_block(_make_block("p1", ContextLayer.PERSISTENT, 100, last_ref=0))
        for _ in range(15):
            engine.advance_turn()
        stale = engine.get_stale_blocks(max_age=10)
        assert len(stale) == 0


class TestRecommendActions:
    def test_no_actions_when_within_budget(self):
        engine = _make_engine()
        engine.add_block(_make_block("e1", ContextLayer.EPHEMERAL, 1000))
        assert engine.recommend_actions() == []

    def test_drop_stale_on_high_utilization(self):
        engine = _make_engine()
        # Fill to >80% of total budget (160k), so >128k
        # Bypass add_block to avoid REJECT on ephemeral layer overage
        engine._blocks["e1"] = _make_block("e1", ContextLayer.EPHEMERAL, 70_000)
        engine._blocks["e2"] = _make_block("e2", ContextLayer.EPHEMERAL, 70_000)
        actions = engine.recommend_actions()
        assert CompactAction.DROP_STALE_EPHEMERAL in actions

    def test_summarize_ephemeral_on_layer_overage(self):
        engine = _make_engine()
        # ephemeral_budget=80k, fill to 5-20% over => 84k-96k
        engine.add_block(_make_block("e1", ContextLayer.EPHEMERAL, 45_000))
        engine.add_block(_make_block("e2", ContextLayer.EPHEMERAL, 45_000))
        actions = engine.recommend_actions()
        assert CompactAction.SUMMARIZE_EPHEMERAL in actions

    def test_compact_session_on_session_overage(self):
        engine = _make_engine()
        # session_budget=56k, fill 5-20% over
        engine.add_block(_make_block("s1", ContextLayer.SESSION, 30_000))
        engine.add_block(_make_block("s2", ContextLayer.SESSION, 30_000))
        actions = engine.recommend_actions()
        assert CompactAction.COMPACT_SESSION in actions

    def test_force_trim_persistent_on_reject(self):
        engine = _make_engine()
        # Bypass add_block to simulate post-hoc overage
        engine._blocks["p1"] = _make_block("p1", ContextLayer.PERSISTENT, 15_000)
        engine._blocks["p2"] = _make_block("p2", ContextLayer.PERSISTENT, 15_000)
        # Total=30k, budget=24k, overage=25% -> REJECT level
        actions = engine.recommend_actions()
        assert CompactAction.FORCE_TRIM_PERSISTENT in actions
