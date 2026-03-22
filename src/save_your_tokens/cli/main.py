"""CLI entry point: syt init, analyze, report, compact.

Q10: `syt compact` defaults to scanning current directory, supports specified input.
Q11: `syt analyze` defaults to scanning current directory, supports conversation logs.
"""

from __future__ import annotations

from pathlib import Path

import click

from save_your_tokens.core.budget import BudgetEngine
from save_your_tokens.core.spec import BUILTIN_PROFILES, ContextLayer


@click.group()
@click.version_option()
def cli() -> None:
    """save-your-tokens: Context budget management for LLM applications."""


@cli.command()
@click.option("--profile", type=click.Choice(list(BUILTIN_PROFILES.keys())), default="agentic")
@click.option("--dir", "project_dir", default=".", help="Project directory")
def init(profile: str, project_dir: str) -> None:
    """Initialize syt configuration for a project."""
    project = Path(project_dir).resolve()
    config_dir = project / ".syt"
    config_dir.mkdir(exist_ok=True)

    config = {
        "profile": profile,
        "context_window": 200000,
        "stale_max_age": 10,
        "compact_interval": 0,
    }

    import json

    config_path = config_dir / "config.json"
    config_path.write_text(json.dumps(config, indent=2) + "\n", encoding="utf-8")
    click.echo(f"Initialized syt in {config_dir} with profile '{profile}'")


@cli.command()
@click.option("--dir", "project_dir", default=".", help="Project directory to analyze")
@click.option("--log", "log_file", default=None, help="Conversation log file (JSON/JSONL)")
@click.option("--quiet", is_flag=True, help="Minimal output for hook usage")
@click.option("--context-window", default=200000, help="Context window size")
@click.option("--profile", type=click.Choice(list(BUILTIN_PROFILES.keys())), default="agentic")
def analyze(
    project_dir: str,
    log_file: str | None,
    quiet: bool,
    context_window: int,
    profile: str,
) -> None:
    """Analyze context usage. Scans directory by default, or a conversation log."""
    budget_profile = BUILTIN_PROFILES[profile]
    engine = BudgetEngine(context_window, budget_profile)

    if log_file:
        _analyze_log(log_file, engine, quiet)
    else:
        _analyze_directory(project_dir, engine, quiet)


def _analyze_directory(project_dir: str, engine: BudgetEngine, quiet: bool) -> None:
    """Analyze context files in a project directory."""
    from save_your_tokens.integrations.claude_code import CONTEXT_FILE_LAYERS

    project = Path(project_dir).resolve()
    found = 0

    for rel_path, layer in CONTEXT_FILE_LAYERS.items():
        full_path = project / rel_path
        if full_path.exists():
            content = full_path.read_text(encoding="utf-8")
            token_est = len(content) // 4
            from save_your_tokens.core.spec import ContextBlock

            block = ContextBlock(
                id=f"file:{rel_path}",
                layer=layer,
                content=content,
                token_count=token_est,
                source=f"file:{rel_path}",
            )
            engine.add_block(block)
            found += 1

    usage = engine.compute_budgets()

    if quiet:
        click.echo(f"syt: {usage.utilization:.0%} used ({usage.total_used}/{usage.total_budget})")
        return

    click.echo("=== Context Budget Analysis ===")
    click.echo(f"Profile: {engine.profile.name}")
    click.echo(f"Context files found: {found}")
    click.echo()

    for layer in ContextLayer:
        match layer:
            case ContextLayer.PERSISTENT:
                used, budget = usage.persistent_used, usage.persistent_budget
            case ContextLayer.SESSION:
                used, budget = usage.session_used, usage.session_budget
            case ContextLayer.EPHEMERAL:
                used, budget = usage.ephemeral_used, usage.ephemeral_budget

        pct = (used / budget * 100) if budget > 0 else 0
        status = "OK" if pct < 100 else "OVER"
        click.echo(f"  {layer.value:12s}: {used:>7,} / {budget:>7,} tokens ({pct:.0f}%) [{status}]")

    click.echo()
    click.echo(f"  Total: {usage.total_used:,} / {usage.total_budget:,} ({usage.utilization:.0%})")
    actions = engine.recommend_actions()
    if actions:
        click.echo()
        click.echo("Recommended actions:")
        for action in actions:
            click.echo(f"  - {action.value}")


def _analyze_log(log_file: str, engine: BudgetEngine, quiet: bool) -> None:
    """Analyze a conversation log file."""
    import json

    path = Path(log_file)
    if not path.exists():
        click.echo(f"Error: log file not found: {log_file}", err=True)
        raise SystemExit(1)

    content = path.read_text(encoding="utf-8")

    # Support JSONL (one message per line) or JSON array
    messages: list[dict] = []
    try:
        data = json.loads(content)
        if isinstance(data, list):
            messages = data
        else:
            messages = [data]
    except json.JSONDecodeError:
        for line in content.strip().split("\n"):
            if line.strip():
                try:
                    messages.append(json.loads(line))
                except json.JSONDecodeError:
                    continue

    total_tokens = 0
    for msg in messages:
        text = msg.get("content", "")
        if isinstance(text, str):
            total_tokens += len(text) // 4

    if quiet:
        click.echo(f"syt: {len(messages)} messages, ~{total_tokens:,} tokens")
    else:
        click.echo(f"=== Conversation Log Analysis ===")
        click.echo(f"Messages: {len(messages)}")
        click.echo(f"Estimated tokens: {total_tokens:,}")


@cli.command()
@click.argument("target", default=".")
@click.option("--auto", "auto_mode", is_flag=True, help="Auto-compact without confirmation")
@click.option("--dry-run", is_flag=True, help="Show what would be compacted")
@click.option("--profile", type=click.Choice(list(BUILTIN_PROFILES.keys())), default="agentic")
@click.option("--context-window", default=200000, help="Context window size")
def compact(
    target: str, auto_mode: bool, dry_run: bool, profile: str, context_window: int
) -> None:
    """Compact context files. Defaults to current directory, or specify a file/stdin."""
    target_path = Path(target)
    budget_profile = BUILTIN_PROFILES[profile]
    engine = BudgetEngine(context_window, budget_profile)

    if target_path.is_dir():
        _compact_directory(target_path, engine, auto_mode, dry_run)
    elif target_path.is_file():
        _compact_file(target_path, engine, dry_run)
    else:
        click.echo(f"Error: target not found: {target}", err=True)
        raise SystemExit(1)


def _compact_directory(
    directory: Path, engine: BudgetEngine, auto_mode: bool, dry_run: bool
) -> None:
    """Compact context files in a directory."""
    from save_your_tokens.integrations.claude_code import CONTEXT_FILE_LAYERS
    from save_your_tokens.core.spec import ContextBlock
    from save_your_tokens.core.strategy import StrategyEngine

    for rel_path, layer in CONTEXT_FILE_LAYERS.items():
        full_path = directory / rel_path
        if full_path.exists():
            content = full_path.read_text(encoding="utf-8")
            block = ContextBlock(
                id=f"file:{rel_path}",
                layer=layer,
                content=content,
                token_count=len(content) // 4,
                source=f"file:{rel_path}",
            )
            engine.add_block(block)

    actions = engine.recommend_actions()
    if not actions:
        click.echo("No compaction needed.")
        return

    if dry_run:
        click.echo("Would execute:")
        for action in actions:
            click.echo(f"  - {action.value}")
        return

    if not auto_mode:
        click.echo("Recommended compaction actions:")
        for action in actions:
            click.echo(f"  - {action.value}")
        if not click.confirm("Proceed?"):
            return

    strategy = StrategyEngine(engine)
    results = strategy.execute_actions(actions)
    for action, block_ids in results.items():
        click.echo(f"  {action.value}: {len(block_ids)} blocks affected")


def _compact_file(path: Path, engine: BudgetEngine, dry_run: bool) -> None:
    """Compact a single file."""
    from save_your_tokens.reuse.compression import ExtractiveCompressor

    content = path.read_text(encoding="utf-8")
    original_tokens = len(content) // 4

    compressor = ExtractiveCompressor()
    compressed = compressor.compress(content, target_ratio=0.5)
    compressed_tokens = len(compressed) // 4

    if dry_run:
        click.echo(f"Would compact {path.name}: {original_tokens:,} -> ~{compressed_tokens:,} tokens")
        return

    path.write_text(compressed, encoding="utf-8")
    click.echo(f"Compacted {path.name}: {original_tokens:,} -> {compressed_tokens:,} tokens")


@cli.command()
def report() -> None:
    """Generate a token usage report for the current project."""
    click.echo("Report generation not yet implemented. Use `syt analyze` for now.")
