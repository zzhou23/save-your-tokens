"""Claude Code integration.

Q6 decision: Hooks + file management combined.
- Hooks: settings.json pre/post hooks trigger syt analyze/compact
- File management: syt maintains CLAUDE.md, progress.md as budget-managed context files

This integration provides:
1. Hook configuration generator for Claude Code settings.json
2. Context file manager for CLAUDE.md, progress.md, etc.
3. Glue between Claude Code's lifecycle and syt's budget engine
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from save_your_tokens.core.budget import BudgetEngine
from save_your_tokens.core.lifecycle import LifecycleManager
from save_your_tokens.core.spec import ContextBlock, ContextLayer
from save_your_tokens.core.strategy import StrategyEngine
from save_your_tokens.integrations.base import FrameworkIntegration

# Convention-based classification (Q1: hybrid mode defaults)
CONTEXT_FILE_LAYERS: dict[str, ContextLayer] = {
    "CLAUDE.md": ContextLayer.PERSISTENT,
    ".claude/CLAUDE.md": ContextLayer.PERSISTENT,
    "progress.md": ContextLayer.SESSION,
    "docs/progress.md": ContextLayer.SESSION,
    "todo.md": ContextLayer.SESSION,
}


class ClaudeCodeIntegration(FrameworkIntegration):
    """Integration for Claude Code CLI."""

    def __init__(
        self,
        budget_engine: BudgetEngine,
        lifecycle: LifecycleManager,
        strategy: StrategyEngine,
        project_dir: str | Path = ".",
    ) -> None:
        super().__init__(budget_engine, lifecycle, strategy)
        self._project_dir = Path(project_dir).resolve()

    def setup(self, config: dict[str, Any]) -> None:
        """Scan project directory and load context files into budget engine."""
        self._scan_and_load_context_files()
        self._lifecycle.start_session(
            persistent_blocks=self._budget.get_blocks(ContextLayer.PERSISTENT)
        )

    def teardown(self) -> None:
        """End session and optionally write report."""
        self._lifecycle.end_session()

    def intercept_context(self, messages: list[dict[str, Any]]) -> list[dict[str, Any]]:
        """For Claude Code, interception happens via file management, not message transform.

        This method re-scans context files and returns messages unchanged.
        """
        self._scan_and_load_context_files()
        return messages

    def on_response(self, response: dict[str, Any]) -> None:
        """Post-response: run turn evaluation."""
        self.run_post_turn()

    def _scan_and_load_context_files(self) -> None:
        """Scan project for known context files and register them."""
        for rel_path, layer in CONTEXT_FILE_LAYERS.items():
            full_path = self._project_dir / rel_path
            if full_path.exists():
                content = full_path.read_text(encoding="utf-8")
                block_id = f"file:{rel_path}"

                # Remove old version if exists
                self._budget.remove_block(block_id)

                block = ContextBlock(
                    id=block_id,
                    layer=layer,
                    content=content,
                    token_count=len(content) // 4,  # Rough estimate until adapter counts
                    source=f"file:{rel_path}",
                    tags=["context-file", "auto-detected"],
                    metadata={"path": str(full_path), "convention_classified": True},
                )
                self._budget.add_block(block)

    def generate_hooks_config(self) -> dict[str, Any]:
        """Generate Claude Code hooks configuration for settings.json.

        Users add this to their .claude/settings.json to enable auto-analysis.
        """
        return {
            "hooks": {
                "PostToolUse": [
                    {
                        "matcher": ".*",
                        "command": "syt analyze --quiet",
                    }
                ],
                "PreCompact": [
                    {
                        "matcher": ".*",
                        "command": "syt compact --auto",
                    }
                ],
            }
        }

    def write_hooks_config(self, settings_path: str | Path | None = None) -> Path:
        """Write hooks configuration to Claude Code settings file."""
        if settings_path is None:
            settings_path = self._project_dir / ".claude" / "settings.json"

        path = Path(settings_path)
        path.parent.mkdir(parents=True, exist_ok=True)

        existing: dict[str, Any] = {}
        if path.exists():
            existing = json.loads(path.read_text(encoding="utf-8"))

        hooks_config = self.generate_hooks_config()
        existing.setdefault("hooks", {}).update(hooks_config["hooks"])

        path.write_text(json.dumps(existing, indent=2) + "\n", encoding="utf-8")
        return path
