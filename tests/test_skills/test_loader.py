"""Tests for save_your_tokens.skills.loader — budget-aware skill loading."""

import pytest

from save_your_tokens.core.budget import BudgetEngine
from save_your_tokens.core.spec import PROFILE_AGENTIC, ContextLayer, OverageLevel
from save_your_tokens.skills.loader import SkillLoader, SkillMetadata


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
