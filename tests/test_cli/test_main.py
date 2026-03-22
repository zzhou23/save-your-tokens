"""Tests for save_your_tokens.cli.main — CLI end-to-end tests."""

import json

from click.testing import CliRunner

from save_your_tokens.cli.main import cli


def _runner():
    return CliRunner()


class TestInit:
    def test_init_default_profile(self, tmp_path):
        result = _runner().invoke(cli, ["init", "--dir", str(tmp_path)])
        assert result.exit_code == 0
        assert "Initialized syt" in result.output

        config_path = tmp_path / ".syt" / "config.json"
        assert config_path.exists()
        config = json.loads(config_path.read_text(encoding="utf-8"))
        assert config["profile"] == "agentic"

    def test_init_chat_profile(self, tmp_path):
        result = _runner().invoke(cli, ["init", "--profile", "chat", "--dir", str(tmp_path)])
        assert result.exit_code == 0
        config = json.loads((tmp_path / ".syt" / "config.json").read_text(encoding="utf-8"))
        assert config["profile"] == "chat"

    def test_init_creates_syt_dir(self, tmp_path):
        _runner().invoke(cli, ["init", "--dir", str(tmp_path)])
        assert (tmp_path / ".syt").is_dir()


class TestAnalyzeDirectory:
    def test_analyze_empty_dir(self, tmp_path):
        result = _runner().invoke(cli, ["analyze", "--dir", str(tmp_path)])
        assert result.exit_code == 0
        assert "Context files found: 0" in result.output

    def test_analyze_with_claude_md(self, tmp_path):
        (tmp_path / "CLAUDE.md").write_text("# Project context" * 100, encoding="utf-8")
        result = _runner().invoke(cli, ["analyze", "--dir", str(tmp_path)])
        assert result.exit_code == 0
        assert "Context files found: 1" in result.output
        assert "persistent" in result.output

    def test_analyze_quiet_mode(self, tmp_path):
        (tmp_path / "CLAUDE.md").write_text("context", encoding="utf-8")
        result = _runner().invoke(cli, ["analyze", "--dir", str(tmp_path), "--quiet"])
        assert result.exit_code == 0
        assert "syt:" in result.output
        assert "used" in result.output

    def test_analyze_with_profile(self, tmp_path):
        result = _runner().invoke(cli, ["analyze", "--dir", str(tmp_path), "--profile", "rag"])
        assert result.exit_code == 0
        assert "rag" in result.output


class TestAnalyzeLog:
    def test_analyze_json_log(self, tmp_path):
        log = [
            {"role": "user", "content": "Hello world"},
            {"role": "assistant", "content": "Hi there!"},
        ]
        log_path = tmp_path / "chat.json"
        log_path.write_text(json.dumps(log), encoding="utf-8")

        result = _runner().invoke(cli, ["analyze", "--log", str(log_path)])
        assert result.exit_code == 0
        assert "Messages: 2" in result.output

    def test_analyze_jsonl_log(self, tmp_path):
        lines = [
            json.dumps({"role": "user", "content": "msg1"}),
            json.dumps({"role": "assistant", "content": "msg2"}),
        ]
        log_path = tmp_path / "chat.jsonl"
        log_path.write_text("\n".join(lines), encoding="utf-8")

        result = _runner().invoke(cli, ["analyze", "--log", str(log_path)])
        assert result.exit_code == 0
        assert "Messages: 2" in result.output

    def test_analyze_log_quiet(self, tmp_path):
        log_path = tmp_path / "chat.json"
        log_path.write_text(json.dumps([{"content": "test"}]), encoding="utf-8")

        result = _runner().invoke(cli, ["analyze", "--log", str(log_path), "--quiet"])
        assert result.exit_code == 0
        assert "syt:" in result.output

    def test_analyze_missing_log(self, tmp_path):
        result = _runner().invoke(cli, ["analyze", "--log", str(tmp_path / "nope.json")])
        assert result.exit_code == 1


class TestCompact:
    def test_compact_no_action_needed(self, tmp_path):
        (tmp_path / "CLAUDE.md").write_text("small", encoding="utf-8")
        result = _runner().invoke(cli, ["compact", str(tmp_path)])
        assert result.exit_code == 0
        assert "No compaction needed" in result.output

    def test_compact_dry_run(self, tmp_path):
        # Create a large enough file to trigger compaction recommendation
        (tmp_path / "CLAUDE.md").write_text("x" * 200_000, encoding="utf-8")
        result = _runner().invoke(cli, ["compact", str(tmp_path), "--dry-run"])
        assert result.exit_code == 0
        # Either "Would execute" or "No compaction needed"
        assert "Would execute" in result.output or "No compaction needed" in result.output

    def test_compact_single_file(self, tmp_path):
        target = tmp_path / "test.md"
        target.write_text("First sentence here. Second sentence here. Third one.", encoding="utf-8")
        result = _runner().invoke(cli, ["compact", str(target)])
        assert result.exit_code == 0
        assert "Compacted" in result.output

    def test_compact_single_file_dry_run(self, tmp_path):
        target = tmp_path / "test.md"
        target.write_text("Some content to compress.", encoding="utf-8")
        result = _runner().invoke(cli, ["compact", str(target), "--dry-run"])
        assert result.exit_code == 0
        assert "Would compact" in result.output

    def test_compact_missing_target(self):
        result = _runner().invoke(cli, ["compact", "/nonexistent/path"])
        assert result.exit_code == 1

    def test_compact_auto_mode(self, tmp_path):
        (tmp_path / "CLAUDE.md").write_text("small", encoding="utf-8")
        result = _runner().invoke(cli, ["compact", str(tmp_path), "--auto"])
        assert result.exit_code == 0


class TestReport:
    def test_report_placeholder(self):
        result = _runner().invoke(cli, ["report"])
        assert result.exit_code == 0
        assert "not yet implemented" in result.output


class TestVersion:
    def test_version_flag(self):
        result = _runner().invoke(cli, ["--version"])
        assert result.exit_code == 0
        assert "0.1.0" in result.output
