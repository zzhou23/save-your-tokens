"""Tests for Raw SDK integration."""

from save_your_tokens.core.budget import BudgetEngine
from save_your_tokens.core.lifecycle import LifecycleManager
from save_your_tokens.core.spec import PROFILE_AGENTIC
from save_your_tokens.core.strategy import StrategyEngine
from save_your_tokens.integrations.base import FrameworkIntegration


def _make_engines():
    engine = BudgetEngine(context_window=200_000, profile=PROFILE_AGENTIC)
    lifecycle = LifecycleManager(budget_engine=engine)
    strategy = StrategyEngine(budget_engine=engine)
    return engine, lifecycle, strategy


class TestRawSDKIntegration:
    def test_extends_framework_integration(self):
        from save_your_tokens.integrations.raw_sdk import RawSDKIntegration

        assert issubclass(RawSDKIntegration, FrameworkIntegration)

    def test_setup_is_noop(self):
        from save_your_tokens.integrations.raw_sdk import RawSDKIntegration

        engine, lifecycle, strategy = _make_engines()
        integration = RawSDKIntegration(engine, lifecycle, strategy)
        integration.setup({})

    def test_teardown_is_noop(self):
        from save_your_tokens.integrations.raw_sdk import RawSDKIntegration

        engine, lifecycle, strategy = _make_engines()
        integration = RawSDKIntegration(engine, lifecycle, strategy)
        integration.teardown()

    def test_intercept_context_returns_messages(self):
        from save_your_tokens.integrations.raw_sdk import RawSDKIntegration

        engine, lifecycle, strategy = _make_engines()
        integration = RawSDKIntegration(engine, lifecycle, strategy)
        messages = [{"role": "user", "content": "hello"}]
        result = integration.intercept_context(messages)
        assert isinstance(result, list)

    def test_on_response_runs_post_turn(self):
        from save_your_tokens.integrations.raw_sdk import RawSDKIntegration

        engine, lifecycle, strategy = _make_engines()
        integration = RawSDKIntegration(engine, lifecycle, strategy)
        lifecycle.start_session()
        integration.on_response({"content": "response"})
        assert lifecycle.current_turn == 1


class TestSYTWrapper:
    def test_prepare_context_returns_list(self):
        from save_your_tokens.integrations.raw_sdk import SYTWrapper

        engine, lifecycle, strategy = _make_engines()
        wrapper = SYTWrapper(
            adapter=None, budget_engine=engine, lifecycle=lifecycle, strategy=strategy
        )
        lifecycle.start_session()
        result = wrapper.prepare_context([{"role": "user", "content": "hello"}])
        assert isinstance(result, list)

    def test_on_response_returns_summary(self):
        from save_your_tokens.integrations.raw_sdk import SYTWrapper

        engine, lifecycle, strategy = _make_engines()
        wrapper = SYTWrapper(
            adapter=None, budget_engine=engine, lifecycle=lifecycle, strategy=strategy
        )
        lifecycle.start_session()
        result = wrapper.on_response({"content": "hello"})
        assert "turn" in result
        assert "usage" in result
