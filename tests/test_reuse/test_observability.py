"""Tests for observer expansion and lifecycle/strategy wiring."""

from unittest.mock import MagicMock

from save_your_tokens.core.budget import BudgetEngine
from save_your_tokens.core.lifecycle import LifecycleManager
from save_your_tokens.core.spec import PROFILE_AGENTIC, ContextUsage
from save_your_tokens.core.strategy import StrategyEngine
from save_your_tokens.reuse.observability import NoOpObserver


class TestNoOpObserver:
    def test_track_usage_noop(self):
        observer = NoOpObserver()
        observer.track_usage({"type": "test"})

    def test_track_compaction_noop(self):
        observer = NoOpObserver()
        observer.track_compaction(before_tokens=100, after_tokens=50, method="extractive")

    def test_track_budget_warning_noop(self):
        observer = NoOpObserver()
        usage = MagicMock(spec=ContextUsage)
        observer.track_budget_warning(usage=usage, threshold="warn")

    def test_flush_noop(self):
        observer = NoOpObserver()
        observer.flush()


class TestLifecycleObserverWiring:
    def test_lifecycle_accepts_observer(self):
        engine = BudgetEngine(context_window=200_000, profile=PROFILE_AGENTIC)
        observer = MagicMock()
        lm = LifecycleManager(budget_engine=engine, observer=observer)
        assert lm._observer is observer

    def test_lifecycle_defaults_to_noop(self):
        engine = BudgetEngine(context_window=200_000, profile=PROFILE_AGENTIC)
        lm = LifecycleManager(budget_engine=engine)
        assert isinstance(lm._observer, NoOpObserver)

    def test_post_turn_emits_event(self):
        engine = BudgetEngine(context_window=200_000, profile=PROFILE_AGENTIC)
        observer = MagicMock()
        lm = LifecycleManager(budget_engine=engine, observer=observer)
        lm.start_session()
        lm.post_turn()
        observer.track_usage.assert_called_once()
        event = observer.track_usage.call_args[0][0]
        assert event["type"] == "turn_complete"
        assert "turn_number" in event


class TestStrategyObserverWiring:
    def test_strategy_accepts_observer(self):
        engine = BudgetEngine(context_window=200_000, profile=PROFILE_AGENTIC)
        observer = MagicMock()
        se = StrategyEngine(budget_engine=engine, observer=observer)
        assert se._observer is observer

    def test_strategy_defaults_to_noop(self):
        engine = BudgetEngine(context_window=200_000, profile=PROFILE_AGENTIC)
        se = StrategyEngine(budget_engine=engine)
        assert isinstance(se._observer, NoOpObserver)
