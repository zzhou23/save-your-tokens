"""Tests for save_your_tokens.core.strategy — compaction actions."""

from save_your_tokens.core.budget import BudgetEngine
from save_your_tokens.core.spec import (
    PROFILE_AGENTIC,
    CompactAction,
    ContextBlock,
    ContextLayer,
)
from save_your_tokens.core.strategy import DefaultCompactor, StrategyEngine


def _make_engine(window: int = 200_000) -> BudgetEngine:
    return BudgetEngine(context_window=window, profile=PROFILE_AGENTIC)


def _make_block(
    id: str,
    layer: ContextLayer,
    tokens: int,
    last_ref: int = 0,
    priority: int = 50,
) -> ContextBlock:
    return ContextBlock(
        id=id,
        layer=layer,
        content="x" * (tokens * 4),
        token_count=tokens,
        last_referenced_turn=last_ref,
        metadata={"priority": priority},
    )


class TestDefaultCompactor:
    def test_short_content_unchanged(self):
        c = DefaultCompactor()
        result = c.compact("short text", target_tokens=100)
        assert result == "short text"

    def test_long_content_compressed(self):
        c = DefaultCompactor()
        content = "First sentence is important. Second sentence adds detail. Third wraps up."
        result = c.compact(content, target_tokens=5)
        # Extractive compressor keeps important sentences, result is shorter
        assert len(result) < len(content)

    def test_exact_boundary(self):
        c = DefaultCompactor()
        content = "a" * 40  # 40 chars = 10 tokens
        result = c.compact(content, target_tokens=10)
        assert result == content


class TestStrategyDropStale:
    def test_drop_stale_ephemeral(self):
        engine = _make_engine()
        engine.add_block(_make_block("e1", ContextLayer.EPHEMERAL, 100, last_ref=0))
        for _ in range(15):
            engine.advance_turn()

        strategy = StrategyEngine(engine)
        dropped = strategy.execute_action(CompactAction.DROP_STALE_EPHEMERAL)
        assert dropped == ["e1"]
        assert engine.get_blocks(ContextLayer.EPHEMERAL) == []

    def test_no_stale_nothing_dropped(self):
        engine = _make_engine()
        engine.add_block(_make_block("e1", ContextLayer.EPHEMERAL, 100, last_ref=0))
        # Only 5 turns, max_age=10, so not stale
        for _ in range(5):
            engine.advance_turn()

        strategy = StrategyEngine(engine)
        dropped = strategy.execute_action(CompactAction.DROP_STALE_EPHEMERAL)
        assert dropped == []


class TestStrategySummarize:
    def test_summarize_ephemeral(self):
        engine = _make_engine()
        # ephemeral_budget=80k, per_block_target=80k/1=80k
        # Block must exceed per_block_target to be compacted, bypass add_block
        engine._blocks["e1"] = _make_block("e1", ContextLayer.EPHEMERAL, 90_000)

        strategy = StrategyEngine(engine)
        affected = strategy.execute_action(CompactAction.SUMMARIZE_EPHEMERAL)
        assert "e1" in affected
        # Block should still exist but with reduced token count
        blocks = engine.get_blocks(ContextLayer.EPHEMERAL)
        assert len(blocks) == 1
        assert blocks[0].metadata.get("compacted") is True

    def test_compact_session(self):
        engine = _make_engine()
        # session_budget=56k, per_block_target=56k/1=56k
        # Block must exceed per_block_target, bypass add_block
        engine._blocks["s1"] = _make_block("s1", ContextLayer.SESSION, 60_000)

        strategy = StrategyEngine(engine)
        affected = strategy.execute_action(CompactAction.COMPACT_SESSION)
        assert "s1" in affected
        blocks = engine.get_blocks(ContextLayer.SESSION)
        assert len(blocks) == 1
        assert blocks[0].metadata.get("compacted") is True

    def test_empty_layer_no_effect(self):
        engine = _make_engine()
        strategy = StrategyEngine(engine)
        affected = strategy.execute_action(CompactAction.SUMMARIZE_EPHEMERAL)
        assert affected == []

    def test_small_block_not_compacted(self):
        engine = _make_engine()
        # ephemeral_budget=80k, one block of 100 tokens < per_block_target
        engine.add_block(_make_block("e1", ContextLayer.EPHEMERAL, 100))
        strategy = StrategyEngine(engine)
        affected = strategy.execute_action(CompactAction.SUMMARIZE_EPHEMERAL)
        assert affected == []


class TestForceTrimPersistent:
    def test_trim_by_priority(self):
        engine = _make_engine()
        # Bypass add_block rejection by writing directly to _blocks
        engine._blocks["p1"] = _make_block("p1", ContextLayer.PERSISTENT, 12_000, priority=10)
        engine._blocks["p2"] = _make_block("p2", ContextLayer.PERSISTENT, 12_000, priority=90)
        engine._blocks["p3"] = _make_block("p3", ContextLayer.PERSISTENT, 6_000, priority=50)
        # Total=30k, budget=24k. Sorted by priority: p1(10), p3(50), p2(90)
        # Removes lowest priority first: p1(12k) -> total=18k < 24k, stops.

        strategy = StrategyEngine(engine)
        trimmed = strategy.execute_action(CompactAction.FORCE_TRIM_PERSISTENT)
        assert "p1" in trimmed
        assert "p2" not in trimmed  # Higher priority preserved

    def test_trim_empty_persistent(self):
        engine = _make_engine()
        strategy = StrategyEngine(engine)
        trimmed = strategy.execute_action(CompactAction.FORCE_TRIM_PERSISTENT)
        assert trimmed == []


class TestExecuteActions:
    def test_execute_multiple_actions(self):
        engine = _make_engine()
        engine.add_block(_make_block("e1", ContextLayer.EPHEMERAL, 100, last_ref=0))
        for _ in range(15):
            engine.advance_turn()

        strategy = StrategyEngine(engine)
        results = strategy.execute_actions(
            [
                CompactAction.DROP_STALE_EPHEMERAL,
                CompactAction.SUMMARIZE_EPHEMERAL,
            ]
        )
        assert CompactAction.DROP_STALE_EPHEMERAL in results
        assert CompactAction.SUMMARIZE_EPHEMERAL in results

    def test_custom_compactor(self):
        class UpperCompactor:
            def compact(self, content: str, target_tokens: int) -> str:
                return content[: target_tokens * 4].upper()

        engine = _make_engine()
        # Block must exceed per_block_target (80k) to trigger compaction
        engine._blocks["e1"] = _make_block("e1", ContextLayer.EPHEMERAL, 90_000)
        strategy = StrategyEngine(engine, compactor=UpperCompactor())
        affected = strategy.execute_action(CompactAction.SUMMARIZE_EPHEMERAL)
        assert "e1" in affected
