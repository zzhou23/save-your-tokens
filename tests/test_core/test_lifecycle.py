"""Tests for save_your_tokens.core.lifecycle — session lifecycle."""

from save_your_tokens.core.budget import BudgetEngine
from save_your_tokens.core.lifecycle import LifecycleManager, SessionPhase, TurnResult
from save_your_tokens.core.spec import (
    PROFILE_AGENTIC,
    CompactAction,
    ContextBlock,
    ContextLayer,
)


def _make_engine(window: int = 200_000) -> BudgetEngine:
    return BudgetEngine(context_window=window, profile=PROFILE_AGENTIC)


def _make_block(id: str, layer: ContextLayer, tokens: int) -> ContextBlock:
    return ContextBlock(id=id, layer=layer, content="x" * tokens, token_count=tokens)


class TestSessionPhase:
    def test_initial_phase(self):
        lm = LifecycleManager(_make_engine())
        assert lm.phase == SessionPhase.INITIALIZING

    def test_start_session_transitions_to_active(self):
        lm = LifecycleManager(_make_engine())
        lm.start_session()
        assert lm.phase == SessionPhase.ACTIVE

    def test_end_session_transitions_to_ending(self):
        lm = LifecycleManager(_make_engine())
        lm.start_session()
        lm.end_session()
        assert lm.phase == SessionPhase.ENDING


class TestStartSession:
    def test_loads_persistent_blocks(self):
        engine = _make_engine()
        lm = LifecycleManager(engine)
        blocks = [
            _make_block("p1", ContextLayer.PERSISTENT, 500),
            _make_block("p2", ContextLayer.PERSISTENT, 300),
        ]
        lm.start_session(persistent_blocks=blocks)
        assert len(engine.get_blocks(ContextLayer.PERSISTENT)) == 2

    def test_start_without_blocks(self):
        engine = _make_engine()
        lm = LifecycleManager(engine)
        lm.start_session()
        assert engine.get_blocks() == []


class TestPostTurn:
    def test_advances_turn(self):
        lm = LifecycleManager(_make_engine())
        lm.start_session()
        result = lm.post_turn()
        # turn_number = len(_turn_history) before append = 0
        assert result.turn_number == 0
        # current_turn = len(_turn_history) after append = 1
        assert lm.current_turn == 1

    def test_multiple_turns(self):
        lm = LifecycleManager(_make_engine())
        lm.start_session()
        lm.post_turn()
        lm.post_turn()
        result = lm.post_turn()
        # 3rd call: turn_number = len(_turn_history) before append = 2
        assert result.turn_number == 2

    def test_overage_levels_reported(self):
        lm = LifecycleManager(_make_engine())
        lm.start_session()
        result = lm.post_turn()
        assert ContextLayer.PERSISTENT in result.overage_levels
        assert ContextLayer.SESSION in result.overage_levels
        assert ContextLayer.EPHEMERAL in result.overage_levels

    def test_no_compaction_when_under_budget(self):
        lm = LifecycleManager(_make_engine())
        lm.start_session()
        result = lm.post_turn()
        assert not result.needs_compaction
        assert result.recommended_actions == []

    def test_stale_blocks_detected(self):
        engine = _make_engine()
        lm = LifecycleManager(engine)
        lm.configure(stale_max_age=3)
        lm.start_session()
        engine.add_block(_make_block("e1", ContextLayer.EPHEMERAL, 100))

        # Advance enough turns without referencing e1
        for _ in range(5):
            lm.post_turn()

        result = lm.post_turn()
        assert "e1" in result.stale_block_ids

    def test_referenced_blocks_not_stale(self):
        engine = _make_engine()
        lm = LifecycleManager(engine)
        lm.configure(stale_max_age=3)
        lm.start_session()
        engine.add_block(_make_block("e1", ContextLayer.EPHEMERAL, 100))

        for _ in range(5):
            lm.post_turn(referenced_block_ids=["e1"])

        result = lm.post_turn(referenced_block_ids=["e1"])
        assert "e1" not in result.stale_block_ids


class TestCompactInterval:
    def test_interval_triggers_drop(self):
        engine = _make_engine()
        lm = LifecycleManager(engine)
        lm.configure(compact_interval=3)
        lm.start_session()

        # turn_num is 0-indexed: 0, 1, 2, ...
        # 0 % 3 == 0 triggers, 1 % 3 != 0, 2 % 3 != 0, 3 % 3 == 0 triggers
        r1 = lm.post_turn()  # turn_num=0, 0%3==0 -> triggers
        assert CompactAction.DROP_STALE_EPHEMERAL in r1.recommended_actions

        r2 = lm.post_turn()  # turn_num=1, 1%3!=0
        assert CompactAction.DROP_STALE_EPHEMERAL not in r2.recommended_actions

        r3 = lm.post_turn()  # turn_num=2, 2%3!=0
        assert CompactAction.DROP_STALE_EPHEMERAL not in r3.recommended_actions

        r4 = lm.post_turn()  # turn_num=3, 3%3==0 -> triggers
        assert CompactAction.DROP_STALE_EPHEMERAL in r4.recommended_actions

    def test_interval_disabled_by_default(self):
        lm = LifecycleManager(_make_engine())
        lm.start_session()
        for _ in range(10):
            result = lm.post_turn()
        # compact_interval=0 means disabled
        assert not result.needs_compaction


class TestEndSession:
    def test_returns_summary(self):
        engine = _make_engine()
        lm = LifecycleManager(engine)
        lm.start_session()
        lm.post_turn()
        lm.post_turn()
        summary = lm.end_session()

        assert summary["total_turns"] == 2
        assert "final_usage" in summary
        assert "compaction_events" in summary
        assert "total_stale_detected" in summary


class TestTurnResult:
    def test_needs_compaction_false(self):
        tr = TurnResult(turn_number=1)
        assert not tr.needs_compaction

    def test_needs_compaction_true(self):
        tr = TurnResult(
            turn_number=1,
            recommended_actions=[CompactAction.DROP_STALE_EPHEMERAL],
        )
        assert tr.needs_compaction
