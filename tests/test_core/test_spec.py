"""Tests for save_your_tokens.core.spec — data models and profiles."""

import pytest
from pydantic import ValidationError

from save_your_tokens.core.spec import (
    BUILTIN_PROFILES,
    PROFILE_AGENTIC,
    PROFILE_CHAT,
    PROFILE_RAG,
    BudgetProfile,
    CompactAction,
    CompactTrigger,
    ContextBlock,
    ContextLayer,
    ContextUsage,
    OverageLevel,
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
            BudgetProfile(name="bad", persistent_pct=-0.1, session_pct=0.5, output_reserve_pct=0.2)

    def test_validation_rejects_above_one(self):
        with pytest.raises(ValidationError):
            BudgetProfile(name="bad", persistent_pct=1.5, session_pct=0.5, output_reserve_pct=0.2)


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
