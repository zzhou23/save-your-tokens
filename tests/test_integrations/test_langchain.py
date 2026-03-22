"""Tests for LangChain LCEL integration."""

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


class TestLangChainIntegration:
    def test_extends_framework_integration(self):
        from save_your_tokens.integrations.langchain import LangChainIntegration

        assert issubclass(LangChainIntegration, FrameworkIntegration)

    def test_setup_is_noop(self):
        from save_your_tokens.integrations.langchain import LangChainIntegration

        engine, lifecycle, strategy = _make_engines()
        integration = LangChainIntegration(engine, lifecycle, strategy)
        integration.setup({})

    def test_intercept_context_returns_messages(self):
        from save_your_tokens.integrations.langchain import LangChainIntegration

        engine, lifecycle, strategy = _make_engines()
        integration = LangChainIntegration(engine, lifecycle, strategy)
        messages = [{"role": "user", "content": "hello"}]
        result = integration.intercept_context(messages)
        assert isinstance(result, list)

    def test_on_response_advances_turn(self):
        from save_your_tokens.integrations.langchain import LangChainIntegration

        engine, lifecycle, strategy = _make_engines()
        integration = LangChainIntegration(engine, lifecycle, strategy)
        lifecycle.start_session()
        integration.on_response({"content": "reply"})
        assert lifecycle.current_turn == 1


class TestSYTRunnable:
    def test_invoke_returns_processed_messages(self):
        from save_your_tokens.integrations.langchain import SYTRunnable

        engine, lifecycle, strategy = _make_engines()
        lifecycle.start_session()
        runnable = SYTRunnable(budget_engine=engine, lifecycle=lifecycle, strategy=strategy)
        result = runnable.invoke({"messages": [{"role": "user", "content": "hi"}]})
        assert isinstance(result, dict)
        assert "messages" in result
