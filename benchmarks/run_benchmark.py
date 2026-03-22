"""Benchmark: baseline (cumulative resend) vs syt-managed context.

Loads a synthetic 50-turn agentic session and compares:
- Baseline: every turn resends ALL prior context (how LLMs actually work)
- Managed: syt's BudgetEngine + LifecycleManager + StrategyEngine actively manage context

Metrics: total tokens sent, savings %, compaction events, per-turn utilization.
"""

from __future__ import annotations

import json
import sys
from dataclasses import dataclass, field
from pathlib import Path

# Add src to path for local dev
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from save_your_tokens.core.budget import BudgetEngine
from save_your_tokens.core.lifecycle import LifecycleManager
from save_your_tokens.core.spec import (
    PROFILE_AGENTIC,
    ContextBlock,
    ContextLayer,
)
from save_your_tokens.core.strategy import StrategyEngine
from save_your_tokens.reuse.tokenizers import TokenCounter

CONTEXT_WINDOW = 200_000
LAYER_MAP = {
    "persistent": ContextLayer.PERSISTENT,
    "session": ContextLayer.SESSION,
    "ephemeral": ContextLayer.EPHEMERAL,
}


@dataclass(frozen=True)
class TurnSnapshot:
    """Immutable snapshot of a single turn's metrics."""

    turn: int
    baseline_tokens: int
    managed_tokens: int
    utilization: float
    compaction_triggered: bool
    actions_taken: tuple[str, ...] = ()


@dataclass
class BenchmarkResult:
    """Aggregate benchmark results."""

    total_turns: int = 0
    total_baseline_tokens: int = 0
    total_managed_tokens: int = 0
    compaction_events: int = 0
    actions_by_type: dict[str, int] = field(default_factory=dict)
    snapshots: list[TurnSnapshot] = field(default_factory=list)

    @property
    def savings_pct(self) -> float:
        if self.total_baseline_tokens == 0:
            return 0.0
        return (1 - self.total_managed_tokens / self.total_baseline_tokens) * 100

    @property
    def savings_tokens(self) -> int:
        return self.total_baseline_tokens - self.total_managed_tokens


def load_dataset(path: Path) -> list[dict]:
    """Load JSONL dataset."""
    messages: list[dict] = []
    with path.open(encoding="utf-8") as f:
        for line in f:
            if line.strip():
                messages.append(json.loads(line))
    return messages


def run_baseline(messages: list[dict], counter: TokenCounter) -> list[int]:
    """Baseline: cumulative resend — each turn sends all prior messages.

    Returns per-turn cumulative token counts.
    """
    cumulative: list[int] = []
    total = 0
    for msg in messages:
        tokens = counter.count(msg["content"])
        total += tokens
        # Each turn resends everything up to this point
        cumulative.append(total)
    return cumulative


def run_managed(messages: list[dict], counter: TokenCounter) -> BenchmarkResult:
    """Managed: syt actively manages context budget.

    Uses BudgetEngine + LifecycleManager + StrategyEngine.
    """
    engine = BudgetEngine(context_window=CONTEXT_WINDOW, profile=PROFILE_AGENTIC)
    lifecycle = LifecycleManager(engine)
    lifecycle.configure(stale_max_age=8, compact_interval=10)
    strategy = StrategyEngine(engine)
    lifecycle.start_session()

    result = BenchmarkResult()
    current_turn = 0
    block_counter = 0
    referenced_ids: list[str] = []

    for msg in messages:
        msg_turn = msg["turn"]
        tokens = counter.count(msg["content"])
        layer = LAYER_MAP.get(msg["layer"], ContextLayer.EPHEMERAL)

        # Create a context block for this message
        block_id = f"{msg['type']}:{block_counter}"
        block_counter += 1
        block = ContextBlock(
            id=block_id,
            layer=layer,
            content=msg["content"],
            token_count=tokens,
            source=f"{msg['role']}:{msg['type']}",
            last_referenced_turn=lifecycle.current_turn,
        )
        engine.add_block(block)
        referenced_ids.append(block_id)

        # When turn advances, evaluate
        if msg_turn > current_turn:
            turn_result = lifecycle.post_turn(referenced_block_ids=referenced_ids)
            referenced_ids = []

            # Execute compaction if needed
            actions_taken: list[str] = []
            if turn_result.needs_compaction:
                action_results = strategy.execute_actions(turn_result.recommended_actions)
                for action, block_ids in action_results.items():
                    if block_ids:
                        actions_taken.append(action.value)
                        result.actions_by_type[action.value] = (
                            result.actions_by_type.get(action.value, 0) + 1
                        )
                if actions_taken:
                    result.compaction_events += 1

            # Record snapshot
            usage = engine.compute_budgets()
            result.snapshots.append(
                TurnSnapshot(
                    turn=msg_turn,
                    baseline_tokens=0,  # filled in later
                    managed_tokens=usage.total_used,
                    utilization=usage.utilization,
                    compaction_triggered=bool(actions_taken),
                    actions_taken=tuple(actions_taken),
                )
            )
            current_turn = msg_turn

    # Final turn
    if referenced_ids:
        turn_result = lifecycle.post_turn(referenced_block_ids=referenced_ids)
        if turn_result.needs_compaction:
            strategy.execute_actions(turn_result.recommended_actions)

    usage = engine.compute_budgets()
    result.total_turns = current_turn
    result.total_managed_tokens = sum(s.managed_tokens for s in result.snapshots)

    return result


def print_results(result: BenchmarkResult, baseline_per_turn: list[int]) -> None:
    """Print formatted benchmark results."""
    print("=" * 70)
    print("  save-your-tokens Benchmark Results")
    print("=" * 70)
    print()

    # Fill in baseline tokens per snapshot
    baseline_idx = 0
    for snapshot in result.snapshots:
        # Find matching baseline cumulative token count
        while baseline_idx < len(baseline_per_turn) - 1:
            baseline_idx += 1
            if baseline_idx >= len(baseline_per_turn):
                break
        # Use the last baseline value for this turn
        turn_baseline = baseline_per_turn[min(snapshot.turn - 1, len(baseline_per_turn) - 1)]
        result.total_baseline_tokens += turn_baseline

    print(f"  Context window:      {CONTEXT_WINDOW:>12,} tokens")
    print(f"  Profile:             {'agentic':>12s}")
    print(f"  Turns:               {result.total_turns:>12,}")
    print()
    print("  --- Token Usage ---")
    print(f"  Baseline (resend):   {result.total_baseline_tokens:>12,} tokens")
    print(f"  Managed (syt):       {result.total_managed_tokens:>12,} tokens")
    print(f"  Savings:             {result.savings_tokens:>12,} tokens ({result.savings_pct:.1f}%)")
    print()
    print("  --- Compaction Events ---")
    print(f"  Total events:        {result.compaction_events:>12,}")
    for action, count in sorted(result.actions_by_type.items()):
        print(f"    {action:30s}: {count:>5,}")
    print()

    # Per-turn utilization chart
    print("  --- Per-Turn Utilization (managed) ---")
    print()
    max_bar = 50
    for snapshot in result.snapshots:
        bar_len = int(snapshot.utilization * max_bar)
        bar = "#" * bar_len + "-" * (max_bar - bar_len)
        marker = " *" if snapshot.compaction_triggered else ""
        print(f"  T{snapshot.turn:>3d} |{bar}| {snapshot.utilization:>5.1%}{marker}")

    print()
    print("  * = compaction triggered")
    print()

    # Summary
    print("  --- Summary ---")
    print(f"  syt reduces cumulative token usage by {result.savings_pct:.1f}%")
    print(
        f"  through {result.compaction_events} compaction events across {result.total_turns} turns."
    )
    if result.savings_pct > 11:
        print("  This exceeds the JetBrains Research benchmark (7-11% cost reduction).")
    print()


def export_csv(result: BenchmarkResult, path: Path) -> None:
    """Export per-turn data as CSV for external charting."""
    with path.open("w", encoding="utf-8") as f:
        f.write("turn,managed_tokens,utilization,compaction\n")
        for s in result.snapshots:
            f.write(f"{s.turn},{s.managed_tokens},{s.utilization:.4f},{int(s.compaction_triggered)}\n")
    print(f"  CSV exported to: {path}")


def main() -> None:
    """Run the benchmark."""
    dataset_path = Path(__file__).parent / "data" / "agentic_session.jsonl"

    if not dataset_path.exists():
        print("Dataset not found. Generating...")
        from generate_dataset import main as gen_main

        gen_main()

    messages = load_dataset(dataset_path)
    counter = TokenCounter.for_model("claude-3")

    print(f"\nLoaded {len(messages)} messages from dataset\n")

    # Run baseline
    baseline_cumulative = run_baseline(messages, counter)

    # Run managed
    result = run_managed(messages, counter)

    # Print results
    print_results(result, baseline_cumulative)

    # Export CSV
    csv_path = Path(__file__).parent / "data" / "results.csv"
    export_csv(result, csv_path)


if __name__ == "__main__":
    main()
