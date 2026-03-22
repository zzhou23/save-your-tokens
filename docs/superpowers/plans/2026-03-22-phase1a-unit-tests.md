# Phase 1A Unit Tests Implementation Plan

> **For agentic workers:** REQUIRED: Use superpowers:subagent-driven-development (if subagents available) or superpowers:executing-plans to implement this plan. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Achieve 80%+ test coverage on core/ and skills/ modules with comprehensive pytest unit tests.

**Architecture:** 6 test modules mirroring the 6 source modules. Each test file focuses on one source module. Tests use only stdlib + pytest + pydantic (no mocking of external services needed since core/ has no I/O). Registry tests use `tmp_path` fixture for filesystem operations.

**Tech Stack:** pytest, pytest-cov, pydantic v2, Python 3.10+

**Conventions:**
- Run tests: `pytest tests/ -v`
- Run single file: `pytest tests/test_core/test_spec.py -v`
- Coverage: `pytest --cov=save_your_tokens --cov-report=term-missing`
- Lint before commit: `ruff format . && ruff check .`

---

## Chunk 1: Core Module Tests

### Task 1: test_spec.py — Data Models & Profiles

**Files:**
- Create: `tests/test_core/test_spec.py`

- [ ] **Step 1: Write tests for ContextLayer enum**

```python
"""Tests for save_your_tokens.core.spec — data models and profiles."""

import pytest

from pydantic import ValidationError

from save_your_tokens.core.spec import (
    BUILTIN_PROFILES,
    BudgetProfile,
    CompactAction,
    CompactTrigger,
    ContextBlock,
    ContextLayer,
    ContextUsage,
    OverageLevel,
    PROFILE_AGENTIC,
    PROFILE_CHAT,
    PROFILE_RAG,
)


class TestContextLayer:
    def test_layer_values(self):
        assert ContextLayer.PERSISTENT == "persistent"
        assert ContextLayer.SESSION == "session"
        assert ContextLayer.EPHEMERAL == "ephemeral"

    def test_layer_is_string_enum(self):
        assert isinstance(ContextLayer.PERSISTENT, str)


class TestBudgetProfile:
    def test_agentic_profile(self):
        p = PROFILE_AGENTIC
        assert p.name == "agentic"
        assert p.persistent_pct == 0.15
        assert p.session_pct == 0.35
        assert p.output_reserve_pct == 0.20

    def test_ephemeral_pct_computed(self):
        p = PROFILE_AGENTIC
        expected = 1.0 - 0.15 - 0.35 - 0.20  # 0.30
        assert abs(p.ephemeral_pct - expected) < 1e-9

    def test_chat_profile(self):
        p = PROFILE_CHAT
        assert p.ephemeral_pct == pytest.approx(0.50)

    def test_rag_profile(self):
        p = PROFILE_RAG
        assert p.ephemeral_pct == pytest.approx(0.60)

    def test_builtin_profiles_contains_all(self):
        assert set(BUILTIN_PROFILES.keys()) == {"chat", "agentic", "rag"}

    def test_ephemeral_pct_clamps_to_zero(self):
        p = BudgetProfile(
            name="overcommit",
            persistent_pct=0.5,
            session_pct=0.4,
            output_reserve_pct=0.2,
        )
        assert p.ephemeral_pct == 0.0

    def test_validation_rejects_negative(self):
        with pytest.raises(ValidationError):
            BudgetProfile(
                name="bad", persistent_pct=-0.1, session_pct=0.5, output_reserve_pct=0.2
            )

    def test_validation_rejects_above_one(self):
        with pytest.raises(ValidationError):
            BudgetProfile(
                name="bad", persistent_pct=1.5, session_pct=0.5, output_reserve_pct=0.2
            )


class TestContextBlock:
    def test_create_block(self):
        block = ContextBlock(
            id="test-1",
            layer=ContextLayer.PERSISTENT,
            content="hello",
            token_count=10,
        )
        assert block.id == "test-1"
        assert block.layer == ContextLayer.PERSISTENT
        assert block.token_count == 10
        assert block.tags == []
        assert block.last_referenced_turn == 0
        assert block.metadata == {}

    def test_block_with_metadata(self):
        block = ContextBlock(
            id="test-2",
            layer=ContextLayer.SESSION,
            content="data",
            token_count=50,
            source="file:test.md",
            tags=["important"],
            metadata={"priority": 10},
        )
        assert block.source == "file:test.md"
        assert block.tags == ["important"]
        assert block.metadata["priority"] == 10


class TestContextUsage:
    def test_total_used(self):
        usage = ContextUsage(
            context_window=200_000,
            output_reserve=40_000,
            persistent_used=1000,
            persistent_budget=24_000,
            session_used=5000,
            session_budget=56_000,
            ephemeral_used=2000,
            ephemeral_budget=80_000,
        )
        assert usage.total_used == 8000

    def test_total_budget(self):
        usage = ContextUsage(
            context_window=200_000,
            output_reserve=40_000,
        )
        assert usage.total_budget == 160_000

    def test_utilization(self):
        usage = ContextUsage(
            context_window=200_000,
            output_reserve=40_000,
            persistent_used=80_000,
            persistent_budget=24_000,
            session_used=0,
            session_budget=56_000,
            ephemeral_used=0,
            ephemeral_budget=80_000,
        )
        assert usage.utilization == pytest.approx(0.5)

    def test_utilization_zero_budget(self):
        usage = ContextUsage(context_window=0, output_reserve=0)
        assert usage.utilization == 0.0


class TestEnums:
    def test_overage_levels(self):
        assert OverageLevel.WITHIN == "within"
        assert OverageLevel.WARN == "warn"
        assert OverageLevel.COMPACT == "compact"
        assert OverageLevel.REJECT == "reject"

    def test_compact_triggers(self):
        assert len(CompactTrigger) == 4

    def test_compact_actions_order(self):
        actions = list(CompactAction)
        assert actions[0] == CompactAction.DROP_STALE_EPHEMERAL
        assert actions[-1] == CompactAction.FORCE_TRIM_PERSISTENT
```

- [ ] **Step 2: Run tests to verify they pass**

Run: `pytest tests/test_core/test_spec.py -v`
Expected: All tests PASS

- [ ] **Step 3: Commit**

```bash
git add tests/test_core/test_spec.py
git commit -m "test: add unit tests for core/spec.py data models and profiles"
```

---

### Task 2: test_budget.py — Budget Engine

**Files:**
- Create: `tests/test_core/test_budget.py`

- [ ] **Step 1: Write tests for BudgetEngine**

```python
"""Tests for save_your_tokens.core.budget — budget allocation and overage."""

import pytest

from save_your_tokens.core.budget import COMPACT_THRESHOLD, WARN_THRESHOLD, BudgetEngine
from save_your_tokens.core.spec import (
    CompactAction,
    ContextBlock,
    ContextLayer,
    OverageLevel,
    PROFILE_AGENTIC,
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
        # add_block returns WARN but keeps the block (only REJECT removes)
        level = engine.add_block(_make_block("p1", ContextLayer.PERSISTENT, 24_500))
        assert level == OverageLevel.WARN
        assert engine.check_overage(ContextLayer.PERSISTENT) == OverageLevel.WARN

    def test_compact_level(self):
        engine = _make_engine()
        # 5-20% over: 25200..28799
        # add_block returns COMPACT but keeps the block
        level = engine.add_block(_make_block("p1", ContextLayer.PERSISTENT, 26_000))
        assert level == OverageLevel.COMPACT

    def test_reject_level(self):
        engine = _make_engine()
        # >20% over: 28800+. add_block auto-removes on REJECT.
        # Bypass via _blocks to verify check_overage logic.
        engine._blocks["p1"] = _make_block("p1", ContextLayer.PERSISTENT, 30_000)
        assert engine.check_overage(ContextLayer.PERSISTENT) == OverageLevel.REJECT

    def test_zero_budget_with_usage_rejects(self):
        """If budget is 0 but there's usage, should reject."""
        from save_your_tokens.core.spec import BudgetProfile

        zero_profile = BudgetProfile(
            name="zero_persistent",
            persistent_pct=0.0,
            session_pct=0.5,
            output_reserve_pct=0.5,
        )
        engine = BudgetEngine(context_window=100_000, profile=zero_profile)
        engine.add_block(_make_block("p1", ContextLayer.PERSISTENT, 100))
        assert engine.check_overage(ContextLayer.PERSISTENT) == OverageLevel.REJECT

    def test_zero_budget_zero_usage_within(self):
        from save_your_tokens.core.spec import BudgetProfile

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
        engine.add_block(_make_block("e1", ContextLayer.EPHEMERAL, 70_000))
        engine.add_block(_make_block("e2", ContextLayer.EPHEMERAL, 70_000))
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
        # persistent_budget=24k. add_block auto-rejects at >20% over (>28.8k),
        # so we bypass it by writing directly to _blocks to simulate post-hoc overage.
        engine._blocks["p1"] = _make_block("p1", ContextLayer.PERSISTENT, 15_000)
        engine._blocks["p2"] = _make_block("p2", ContextLayer.PERSISTENT, 15_000)
        # Total=30k, budget=24k, overage=25% -> REJECT level
        actions = engine.recommend_actions()
        assert CompactAction.FORCE_TRIM_PERSISTENT in actions
```

- [ ] **Step 2: Run tests to verify they pass**

Run: `pytest tests/test_core/test_budget.py -v`
Expected: All tests PASS

- [ ] **Step 3: Commit**

```bash
git add tests/test_core/test_budget.py
git commit -m "test: add unit tests for core/budget.py engine and overage tiers"
```

---

### Task 3: test_strategy.py — Strategy Engine

**Files:**
- Create: `tests/test_core/test_strategy.py`

- [ ] **Step 1: Write tests for StrategyEngine and DefaultCompactor**

```python
"""Tests for save_your_tokens.core.strategy — compaction actions."""

import pytest

from save_your_tokens.core.budget import BudgetEngine
from save_your_tokens.core.spec import (
    CompactAction,
    ContextBlock,
    ContextLayer,
    PROFILE_AGENTIC,
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

    def test_long_content_truncated(self):
        c = DefaultCompactor()
        content = "a" * 1000
        result = c.compact(content, target_tokens=10)
        assert len(result) < 1000
        assert result.endswith("[... truncated ...]")

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
        # Add a big ephemeral block
        engine.add_block(_make_block("e1", ContextLayer.EPHEMERAL, 50_000))

        strategy = StrategyEngine(engine)
        affected = strategy.execute_action(CompactAction.SUMMARIZE_EPHEMERAL)
        assert "e1" in affected
        # Block should still exist but with reduced token count
        blocks = engine.get_blocks(ContextLayer.EPHEMERAL)
        assert len(blocks) == 1
        assert blocks[0].metadata.get("compacted") is True

    def test_compact_session(self):
        engine = _make_engine()
        engine.add_block(_make_block("s1", ContextLayer.SESSION, 40_000))

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
        # to simulate over-budget persistent state.
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
        results = strategy.execute_actions([
            CompactAction.DROP_STALE_EPHEMERAL,
            CompactAction.SUMMARIZE_EPHEMERAL,
        ])
        assert CompactAction.DROP_STALE_EPHEMERAL in results
        assert CompactAction.SUMMARIZE_EPHEMERAL in results

    def test_custom_compactor(self):
        class UpperCompactor:
            def compact(self, content: str, target_tokens: int) -> str:
                return content[:target_tokens * 4].upper()

        engine = _make_engine()
        engine.add_block(_make_block("e1", ContextLayer.EPHEMERAL, 50_000))
        strategy = StrategyEngine(engine, compactor=UpperCompactor())
        affected = strategy.execute_action(CompactAction.SUMMARIZE_EPHEMERAL)
        assert "e1" in affected
```

- [ ] **Step 2: Run tests to verify they pass**

Run: `pytest tests/test_core/test_strategy.py -v`
Expected: All tests PASS

- [ ] **Step 3: Commit**

```bash
git add tests/test_core/test_strategy.py
git commit -m "test: add unit tests for core/strategy.py compaction actions"
```

---

### Task 4: test_lifecycle.py — Lifecycle Manager

**Files:**
- Create: `tests/test_core/test_lifecycle.py`

- [ ] **Step 1: Write tests for LifecycleManager**

```python
"""Tests for save_your_tokens.core.lifecycle — session lifecycle."""

import pytest

from save_your_tokens.core.budget import BudgetEngine
from save_your_tokens.core.lifecycle import LifecycleManager, SessionPhase, TurnResult
from save_your_tokens.core.spec import (
    CompactAction,
    ContextBlock,
    ContextLayer,
    OverageLevel,
    PROFILE_AGENTIC,
)


def _make_engine(window: int = 200_000) -> BudgetEngine:
    return BudgetEngine(context_window=window, profile=PROFILE_AGENTIC)


def _make_block(id: str, layer: ContextLayer, tokens: int) -> ContextBlock:
    return ContextBlock(
        id=id, layer=layer, content="x" * tokens, token_count=tokens
    )


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
        assert result.turn_number == 1
        assert lm.current_turn == 1

    def test_multiple_turns(self):
        lm = LifecycleManager(_make_engine())
        lm.start_session()
        lm.post_turn()
        lm.post_turn()
        result = lm.post_turn()
        assert result.turn_number == 3

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

        # Turns 1, 2 — no interval trigger
        r1 = lm.post_turn()
        r2 = lm.post_turn()
        assert CompactAction.DROP_STALE_EPHEMERAL not in r1.recommended_actions
        assert CompactAction.DROP_STALE_EPHEMERAL not in r2.recommended_actions

        # Turn 3 — interval fires
        r3 = lm.post_turn()
        assert CompactAction.DROP_STALE_EPHEMERAL in r3.recommended_actions

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
```

- [ ] **Step 2: Run tests to verify they pass**

Run: `pytest tests/test_core/test_lifecycle.py -v`
Expected: All tests PASS

- [ ] **Step 3: Commit**

```bash
git add tests/test_core/test_lifecycle.py
git commit -m "test: add unit tests for core/lifecycle.py session management"
```

---

## Chunk 2: Skills Module Tests

### Task 5: test_loader.py — Skill Loader

**Files:**
- Create: `tests/test_skills/test_loader.py`

- [ ] **Step 1: Write tests for SkillLoader**

```python
"""Tests for save_your_tokens.skills.loader — budget-aware skill loading."""

import pytest

from save_your_tokens.core.budget import BudgetEngine
from save_your_tokens.core.spec import ContextLayer, OverageLevel, PROFILE_AGENTIC
from save_your_tokens.skills.loader import LoadedSkill, SkillLoader, SkillMetadata


def _make_engine(window: int = 200_000) -> BudgetEngine:
    return BudgetEngine(context_window=window, profile=PROFILE_AGENTIC)


class TestSkillMetadata:
    def test_frozen(self):
        meta = SkillMetadata(name="test")
        with pytest.raises(AttributeError):
            meta.name = "changed"

    def test_defaults(self):
        meta = SkillMetadata(name="test")
        assert meta.description == ""
        assert meta.tags == []
        assert meta.priority == 50
        assert meta.layer == ContextLayer.SESSION
        assert meta.source_path == ""


class TestSkillLoad:
    def test_load_within_budget(self):
        loader = SkillLoader(_make_engine())
        level = loader.load("debug", "debug content", token_count=1000)
        assert level == OverageLevel.WITHIN
        assert "debug" in loader.get_loaded_names()

    def test_load_with_custom_metadata(self):
        loader = SkillLoader(_make_engine())
        meta = SkillMetadata(name="debug", priority=10, tags=["core"])
        level = loader.load("debug", "content", token_count=500, metadata=meta)
        assert level == OverageLevel.WITHIN
        assert loader.loaded_skills["debug"].metadata.priority == 10

    def test_load_reject_not_stored(self):
        engine = _make_engine(10_000)  # Small window
        loader = SkillLoader(engine)
        # session_budget = (10k - 2k) * 0.35 = 2800
        # >20% over = >3360 => reject
        level = loader.load("huge", "x" * 20_000, token_count=5000)
        assert level == OverageLevel.REJECT
        assert "huge" not in loader.get_loaded_names()

    def test_reload_replaces_existing(self):
        loader = SkillLoader(_make_engine())
        loader.load("debug", "v1", token_count=1000)
        loader.load("debug", "v2", token_count=2000)
        assert loader.loaded_skills["debug"].block.content == "v2"
        assert len(loader.get_loaded_names()) == 1


class TestSkillUnload:
    def test_unload_existing(self):
        loader = SkillLoader(_make_engine())
        loader.load("debug", "content", token_count=1000)
        assert loader.unload("debug") is True
        assert "debug" not in loader.get_loaded_names()

    def test_unload_nonexistent(self):
        loader = SkillLoader(_make_engine())
        assert loader.unload("nope") is False


class TestBudgetSummary:
    def test_summary_structure(self):
        loader = SkillLoader(_make_engine())
        loader.load("a", "content-a", token_count=500)
        loader.load("b", "content-b", token_count=300)
        summary = loader.get_budget_summary()
        assert set(summary.keys()) == {"a", "b"}
        assert summary["a"]["token_count"] == 500
        assert summary["a"]["layer"] == "session"
        assert summary["a"]["priority"] == 50

    def test_empty_summary(self):
        loader = SkillLoader(_make_engine())
        assert loader.get_budget_summary() == {}
```

- [ ] **Step 2: Run tests to verify they pass**

Run: `pytest tests/test_skills/test_loader.py -v`
Expected: All tests PASS

- [ ] **Step 3: Commit**

```bash
git add tests/test_skills/test_loader.py
git commit -m "test: add unit tests for skills/loader.py budget-aware loading"
```

---

### Task 6: test_registry.py — Skill Registry

**Files:**
- Create: `tests/test_skills/test_registry.py`

- [ ] **Step 1: Write tests for SkillRegistry and frontmatter parser**

```python
"""Tests for save_your_tokens.skills.registry — skill discovery."""

import json

import pytest

from save_your_tokens.core.spec import ContextLayer
from save_your_tokens.skills.registry import SkillRegistry, _parse_frontmatter


class TestParseFrontmatter:
    def test_no_frontmatter(self):
        meta, body = _parse_frontmatter("just plain content")
        assert meta == {}
        assert body == "just plain content"

    def test_valid_frontmatter(self):
        content = "---\nname: debug\ndescription: A debugging skill\n---\nBody here."
        meta, body = _parse_frontmatter(content)
        assert meta["name"] == "debug"
        assert meta["description"] == "A debugging skill"
        assert body == "Body here."

    def test_unclosed_frontmatter(self):
        content = "---\nname: debug\nno closing marker"
        meta, body = _parse_frontmatter(content)
        assert meta == {}
        assert body == content

    def test_empty_body(self):
        content = "---\nname: test\n---\n"
        meta, body = _parse_frontmatter(content)
        assert meta["name"] == "test"
        assert body == ""


class TestRegistryScanText:
    def test_scan_md_files(self, tmp_path):
        skill_file = tmp_path / "debug.md"
        skill_file.write_text(
            "---\nname: debug\ndescription: Debug skill\ntags: core, dev\npriority: 10\n---\nBody content.",
            encoding="utf-8",
        )

        registry = SkillRegistry()
        registry.add_scan_dir(tmp_path)
        count = registry.scan()

        assert count == 1
        assert "debug" in registry.catalog
        meta = registry.catalog["debug"]
        assert meta.description == "Debug skill"
        assert meta.tags == ["core", "dev"]
        assert meta.priority == 10

    def test_scan_txt_files(self, tmp_path):
        skill_file = tmp_path / "helper.txt"
        skill_file.write_text("plain text skill content", encoding="utf-8")

        registry = SkillRegistry()
        registry.add_scan_dir(tmp_path)
        count = registry.scan()

        assert count == 1
        assert "helper" in registry.catalog  # Uses stem as name

    def test_default_layer_session(self, tmp_path):
        skill_file = tmp_path / "basic.md"
        skill_file.write_text("no frontmatter", encoding="utf-8")

        registry = SkillRegistry()
        registry.add_scan_dir(tmp_path)
        registry.scan()

        assert registry.catalog["basic"].layer == ContextLayer.SESSION

    def test_custom_layer(self, tmp_path):
        skill_file = tmp_path / "sys.md"
        skill_file.write_text("---\nname: sys\nlayer: persistent\n---\nSystem skill.", encoding="utf-8")

        registry = SkillRegistry()
        registry.add_scan_dir(tmp_path)
        registry.scan()

        assert registry.catalog["sys"].layer == ContextLayer.PERSISTENT

    def test_invalid_layer_defaults_session(self, tmp_path):
        skill_file = tmp_path / "bad.md"
        skill_file.write_text("---\nname: bad\nlayer: bogus\n---\nContent.", encoding="utf-8")

        registry = SkillRegistry()
        registry.add_scan_dir(tmp_path)
        registry.scan()

        assert registry.catalog["bad"].layer == ContextLayer.SESSION


class TestRegistryScanJson:
    def test_scan_json_skill(self, tmp_path):
        skill_data = {
            "name": "json-skill",
            "description": "A JSON skill",
            "content": "JSON body",
            "tags": ["test"],
            "priority": 20,
            "layer": "ephemeral",
        }
        skill_file = tmp_path / "skill.json"
        skill_file.write_text(json.dumps(skill_data), encoding="utf-8")

        registry = SkillRegistry()
        registry.add_scan_dir(tmp_path)
        count = registry.scan()

        assert count == 1
        meta = registry.catalog["json-skill"]
        assert meta.description == "A JSON skill"
        assert meta.tags == ["test"]
        assert meta.priority == 20
        assert meta.layer == ContextLayer.EPHEMERAL

    def test_invalid_json_skipped(self, tmp_path):
        skill_file = tmp_path / "broken.json"
        skill_file.write_text("{invalid json", encoding="utf-8")

        registry = SkillRegistry()
        registry.add_scan_dir(tmp_path)
        count = registry.scan()

        assert count == 0


class TestRegistryGetContent:
    def test_get_skill_content(self, tmp_path):
        skill_file = tmp_path / "test.md"
        skill_file.write_text("---\nname: test\n---\nThe body.", encoding="utf-8")

        registry = SkillRegistry()
        registry.add_scan_dir(tmp_path)
        registry.scan()

        result = registry.get_skill_content("test")
        assert result is not None
        meta, content = result
        assert meta.name == "test"
        assert content == "The body."

    def test_get_nonexistent_returns_none(self):
        registry = SkillRegistry()
        assert registry.get_skill_content("nope") is None


class TestRegistryEdgeCases:
    def test_nonexistent_dir_ignored(self):
        registry = SkillRegistry()
        registry.add_scan_dir("/this/does/not/exist")
        assert registry.scan() == 0

    def test_rescan_clears_catalog(self, tmp_path):
        skill_file = tmp_path / "a.md"
        skill_file.write_text("content a", encoding="utf-8")

        registry = SkillRegistry()
        registry.add_scan_dir(tmp_path)
        registry.scan()
        assert len(registry.catalog) == 1

        skill_file.unlink()
        registry.scan()
        assert len(registry.catalog) == 0

    def test_recursive_scan(self, tmp_path):
        subdir = tmp_path / "nested" / "deep"
        subdir.mkdir(parents=True)
        (subdir / "inner.md").write_text("nested content", encoding="utf-8")

        registry = SkillRegistry()
        registry.add_scan_dir(tmp_path)
        count = registry.scan()

        assert count == 1
        assert "inner" in registry.catalog
```

- [ ] **Step 2: Run tests to verify they pass**

Run: `pytest tests/test_skills/test_registry.py -v`
Expected: All tests PASS

- [ ] **Step 3: Commit**

```bash
git add tests/test_skills/test_registry.py
git commit -m "test: add unit tests for skills/registry.py discovery and frontmatter"
```

---

## Chunk 3: Coverage Verification

### Task 7: Run Full Coverage Report

- [ ] **Step 1: Run full test suite with coverage**

Run: `pytest tests/ -v --cov=save_your_tokens --cov-report=term-missing`
Expected: All tests PASS, core/ modules at 80%+

- [ ] **Step 2: If coverage < 80% on any core module, add targeted tests**

Check `term-missing` output for uncovered lines and add tests as needed.

- [ ] **Step 3: Final commit with any gap-filling tests**

```bash
git add tests/
git commit -m "test: Phase 1A complete — core/ and skills/ unit tests at 80%+ coverage"
```
