"""Skill loader: load/unload skills with budget awareness.

Q7: syt doesn't care about skill content, only budget management.
Q8: Skills are format-agnostic text blocks with metadata.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from save_your_tokens.core.budget import BudgetEngine
from save_your_tokens.core.spec import ContextBlock, ContextLayer, OverageLevel


@dataclass(frozen=True)
class SkillMetadata:
    """Metadata for a loadable skill (Q8: format-agnostic)."""

    name: str
    description: str = ""
    tags: list[str] = field(default_factory=list)
    priority: int = 50  # 0=critical, 100=optional
    layer: ContextLayer = ContextLayer.SESSION
    source_path: str = ""  # Original file/directory path, for reference only


@dataclass
class LoadedSkill:
    """A skill currently loaded into the context budget."""

    metadata: SkillMetadata
    block: ContextBlock


class SkillLoader:
    """Budget-aware skill loading and unloading."""

    def __init__(self, budget_engine: BudgetEngine) -> None:
        self._engine = budget_engine
        self._loaded: dict[str, LoadedSkill] = {}

    @property
    def loaded_skills(self) -> dict[str, LoadedSkill]:
        return dict(self._loaded)

    def load(
        self,
        name: str,
        content: str,
        token_count: int,
        metadata: SkillMetadata | None = None,
    ) -> OverageLevel:
        """Load a skill into context. Returns overage level after loading.

        If loading would cause REJECT-level overage, the skill is not loaded.
        """
        if name in self._loaded:
            self.unload(name)

        meta = metadata or SkillMetadata(name=name)
        block = ContextBlock(
            id=f"skill:{name}",
            layer=meta.layer,
            content=content,
            token_count=token_count,
            source=f"skill:{name}",
            tags=meta.tags,
            metadata={"priority": meta.priority, "skill_name": name},
        )

        overage = self._engine.add_block(block)
        if overage != OverageLevel.REJECT:
            self._loaded[name] = LoadedSkill(metadata=meta, block=block)
        return overage

    def unload(self, name: str) -> bool:
        """Unload a skill from context. Returns True if the skill was loaded."""
        skill = self._loaded.pop(name, None)
        if skill is None:
            return False
        self._engine.remove_block(skill.block.id)
        return True

    def get_loaded_names(self) -> list[str]:
        """Get names of all currently loaded skills."""
        return list(self._loaded.keys())

    def get_budget_summary(self) -> dict[str, Any]:
        """Get token usage summary for loaded skills."""
        return {
            name: {
                "token_count": skill.block.token_count,
                "layer": skill.metadata.layer.value,
                "priority": skill.metadata.priority,
            }
            for name, skill in self._loaded.items()
        }
