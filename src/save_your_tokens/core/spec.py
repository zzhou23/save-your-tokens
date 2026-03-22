"""Context Budget Specification — Pydantic data models for the three-layer protocol.

Decisions from design Q&A:
- Q1: Hybrid classification — convention-based defaults + user override
- Q2: Tiered budget response — warn on small overage, auto-compact on large, reject on extreme
- Q8: Skills are format-agnostic text blocks with metadata
"""

from __future__ import annotations

from enum import Enum
from typing import Any

from pydantic import BaseModel, Field


class ContextLayer(str, Enum):
    """Every piece of context belongs to exactly one layer."""

    PERSISTENT = "persistent"
    SESSION = "session"
    EPHEMERAL = "ephemeral"


class BudgetProfile(BaseModel):
    """Token budget allocation across layers.

    Budget formula: reserve output first, then allocate persistent -> session -> ephemeral.
    """

    name: str
    persistent_pct: float = Field(ge=0, le=1, description="Fraction of window for persistent layer")
    session_pct: float = Field(ge=0, le=1, description="Fraction of window for session layer")
    output_reserve_pct: float = Field(
        ge=0, le=1, description="Fraction of window reserved for output"
    )

    @property
    def ephemeral_pct(self) -> float:
        return max(0.0, 1.0 - self.persistent_pct - self.session_pct - self.output_reserve_pct)


# Predefined profiles
PROFILE_CHAT = BudgetProfile(
    name="chat", persistent_pct=0.05, session_pct=0.20, output_reserve_pct=0.25
)
PROFILE_AGENTIC = BudgetProfile(
    name="agentic", persistent_pct=0.15, session_pct=0.35, output_reserve_pct=0.20
)
PROFILE_RAG = BudgetProfile(
    name="rag", persistent_pct=0.05, session_pct=0.10, output_reserve_pct=0.25
)

BUILTIN_PROFILES: dict[str, BudgetProfile] = {
    "chat": PROFILE_CHAT,
    "agentic": PROFILE_AGENTIC,
    "rag": PROFILE_RAG,
}


class OverageLevel(str, Enum):
    """Tiered response levels for budget overage (Q2 decision)."""

    WITHIN = "within"
    WARN = "warn"  # <5% over
    COMPACT = "compact"  # 5-20% over
    REJECT = "reject"  # >20% over


class ContextBlock(BaseModel):
    """A single piece of context with metadata.

    Format-agnostic (Q8): can represent a skill, a file, a message, etc.
    """

    id: str
    layer: ContextLayer
    content: str
    token_count: int = 0
    source: str = ""  # e.g. "skill:debugging", "file:CLAUDE.md", "message:turn-5"
    tags: list[str] = Field(default_factory=list)
    last_referenced_turn: int = 0  # For staleness tracking (Q3)
    metadata: dict[str, Any] = Field(default_factory=dict)


class ContextUsage(BaseModel):
    """Current token usage snapshot across all layers."""

    context_window: int
    output_reserve: int
    persistent_used: int = 0
    persistent_budget: int = 0
    session_used: int = 0
    session_budget: int = 0
    ephemeral_used: int = 0
    ephemeral_budget: int = 0
    current_turn: int = 0

    @property
    def total_used(self) -> int:
        return self.persistent_used + self.session_used + self.ephemeral_used

    @property
    def total_budget(self) -> int:
        return self.context_window - self.output_reserve

    @property
    def utilization(self) -> float:
        if self.total_budget == 0:
            return 0.0
        return self.total_used / self.total_budget


class CompactTrigger(str, Enum):
    """What triggered a compact action."""

    THRESHOLD = "threshold"  # >80% of window used
    INTERVAL = "interval"  # Every N turns
    STALENESS = "staleness"  # Unreferenced content older than M turns
    QUALITY = "quality"  # Repetition/contradiction detected


class CompactAction(str, Enum):
    """Escalating compact actions."""

    DROP_STALE_EPHEMERAL = "drop_stale_ephemeral"
    SUMMARIZE_EPHEMERAL = "summarize_ephemeral"
    COMPACT_SESSION = "compact_session"
    FORCE_TRIM_PERSISTENT = "force_trim_persistent"  # Last resort
