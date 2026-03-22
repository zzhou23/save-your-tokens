"""Tests for save_your_tokens.integrations.claude_code — Claude Code integration."""

import json

from save_your_tokens.core.budget import BudgetEngine
from save_your_tokens.core.lifecycle import LifecycleManager, SessionPhase
from save_your_tokens.core.spec import PROFILE_AGENTIC, ContextLayer
from save_your_tokens.core.strategy import StrategyEngine
from save_your_tokens.integrations.claude_code import (
    CONTEXT_FILE_LAYERS,
    ClaudeCodeIntegration,
)


def _make_integration(tmp_path):
    """Create a ClaudeCodeIntegration with test fixtures."""
    engine = BudgetEngine(context_window=200_000, profile=PROFILE_AGENTIC)
    lifecycle = LifecycleManager(engine)
    strategy = StrategyEngine(engine)
    integration = ClaudeCodeIntegration(engine, lifecycle, strategy, project_dir=tmp_path)
    return integration, engine, lifecycle


class TestContextFileLayers:
    def test_claude_md_is_persistent(self):
        assert CONTEXT_FILE_LAYERS["CLAUDE.md"] == ContextLayer.PERSISTENT

    def test_progress_md_is_session(self):
        assert CONTEXT_FILE_LAYERS["docs/progress.md"] == ContextLayer.SESSION

    def test_todo_md_is_session(self):
        assert CONTEXT_FILE_LAYERS["todo.md"] == ContextLayer.SESSION


class TestScanAndLoadContextFiles:
    def test_scans_claude_md(self, tmp_path):
        (tmp_path / "CLAUDE.md").write_text("# Project\nSome context here.", encoding="utf-8")
        integration, engine, _ = _make_integration(tmp_path)
        integration.setup({})

        blocks = engine.get_blocks(ContextLayer.PERSISTENT)
        assert len(blocks) == 1
        assert blocks[0].id == "file:CLAUDE.md"
        assert "Project" in blocks[0].content

    def test_scans_progress_md(self, tmp_path):
        docs = tmp_path / "docs"
        docs.mkdir()
        (docs / "progress.md").write_text("## Status\nDone.", encoding="utf-8")
        integration, engine, _ = _make_integration(tmp_path)
        integration.setup({})

        blocks = engine.get_blocks(ContextLayer.SESSION)
        assert len(blocks) == 1
        assert blocks[0].id == "file:docs/progress.md"

    def test_missing_files_ignored(self, tmp_path):
        integration, engine, _ = _make_integration(tmp_path)
        integration.setup({})
        assert engine.get_blocks() == []

    def test_rescan_updates_content(self, tmp_path):
        (tmp_path / "CLAUDE.md").write_text("v1", encoding="utf-8")
        integration, engine, _ = _make_integration(tmp_path)
        integration.setup({})

        # Update file and rescan via intercept_context
        (tmp_path / "CLAUDE.md").write_text("v2 updated content", encoding="utf-8")
        integration.intercept_context([])

        blocks = engine.get_blocks(ContextLayer.PERSISTENT)
        assert len(blocks) == 1
        assert "v2" in blocks[0].content

    def test_token_count_estimated(self, tmp_path):
        content = "a" * 400  # 400 chars ≈ 100 tokens
        (tmp_path / "CLAUDE.md").write_text(content, encoding="utf-8")
        integration, engine, _ = _make_integration(tmp_path)
        integration.setup({})

        blocks = engine.get_blocks(ContextLayer.PERSISTENT)
        assert blocks[0].token_count == 100


class TestSetupTeardown:
    def test_setup_starts_session(self, tmp_path):
        integration, _, lifecycle = _make_integration(tmp_path)
        integration.setup({})
        assert lifecycle.phase == SessionPhase.ACTIVE

    def test_teardown_ends_session(self, tmp_path):
        integration, _, lifecycle = _make_integration(tmp_path)
        integration.setup({})
        integration.teardown()
        assert lifecycle.phase == SessionPhase.ENDING


class TestOnResponse:
    def test_on_response_runs_post_turn(self, tmp_path):
        integration, _, lifecycle = _make_integration(tmp_path)
        integration.setup({})
        integration.on_response({})
        assert lifecycle.current_turn == 1


class TestInterceptContext:
    def test_returns_messages_unchanged(self, tmp_path):
        integration, _, _ = _make_integration(tmp_path)
        integration.setup({})
        messages = [{"role": "user", "content": "hello"}]
        result = integration.intercept_context(messages)
        assert result == messages


class TestHooksConfig:
    def test_generate_hooks_config(self, tmp_path):
        integration, _, _ = _make_integration(tmp_path)
        config = integration.generate_hooks_config()
        assert "hooks" in config
        assert "PostToolUse" in config["hooks"]
        assert "PreCompact" in config["hooks"]
        assert config["hooks"]["PostToolUse"][0]["command"] == "syt analyze --quiet"
        assert config["hooks"]["PreCompact"][0]["command"] == "syt compact --auto"

    def test_write_hooks_config_creates_file(self, tmp_path):
        integration, _, _ = _make_integration(tmp_path)
        path = integration.write_hooks_config()
        assert path.exists()
        data = json.loads(path.read_text(encoding="utf-8"))
        assert "hooks" in data
        assert "PostToolUse" in data["hooks"]

    def test_write_hooks_config_merges_existing(self, tmp_path):
        settings_dir = tmp_path / ".claude"
        settings_dir.mkdir()
        settings_path = settings_dir / "settings.json"
        settings_path.write_text(json.dumps({"existing_key": "value"}), encoding="utf-8")

        integration, _, _ = _make_integration(tmp_path)
        integration.write_hooks_config(settings_path)

        data = json.loads(settings_path.read_text(encoding="utf-8"))
        assert data["existing_key"] == "value"
        assert "hooks" in data

    def test_write_hooks_config_custom_path(self, tmp_path):
        custom_path = tmp_path / "custom" / "settings.json"
        integration, _, _ = _make_integration(tmp_path)
        result = integration.write_hooks_config(custom_path)
        assert result == custom_path
        assert custom_path.exists()
