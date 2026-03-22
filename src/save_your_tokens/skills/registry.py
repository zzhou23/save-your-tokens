"""Skill registry: discover skills from directories.

Scans directories for skill files and provides a catalog for the loader.
Format-agnostic: reads any text file and extracts metadata from frontmatter or config.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from save_your_tokens.core.spec import ContextLayer
from save_your_tokens.skills.loader import SkillMetadata


def _parse_frontmatter(content: str) -> tuple[dict[str, Any], str]:
    """Parse YAML-like frontmatter from a text file.

    Returns (metadata_dict, body_content).
    Supports simple key: value pairs only (no nested YAML).
    """
    if not content.startswith("---"):
        return {}, content

    lines = content.split("\n")
    end_idx = -1
    for i in range(1, len(lines)):
        if lines[i].strip() == "---":
            end_idx = i
            break

    if end_idx == -1:
        return {}, content

    meta: dict[str, Any] = {}
    for line in lines[1:end_idx]:
        if ":" in line:
            key, _, value = line.partition(":")
            meta[key.strip()] = value.strip()

    body = "\n".join(lines[end_idx + 1 :]).strip()
    return meta, body


class SkillRegistry:
    """Discovers and catalogs skills from directories."""

    def __init__(self) -> None:
        self._catalog: dict[str, tuple[SkillMetadata, str]] = {}  # name -> (metadata, content)
        self._scan_dirs: list[Path] = []

    @property
    def catalog(self) -> dict[str, SkillMetadata]:
        return {name: meta for name, (meta, _) in self._catalog.items()}

    def add_scan_dir(self, directory: str | Path) -> None:
        """Add a directory to scan for skills."""
        path = Path(directory)
        if path.is_dir():
            self._scan_dirs.append(path)

    def scan(self) -> int:
        """Scan all registered directories. Returns number of skills found."""
        self._catalog.clear()
        for scan_dir in self._scan_dirs:
            self._scan_directory(scan_dir)
        return len(self._catalog)

    def _scan_directory(self, directory: Path) -> None:
        """Scan a single directory for skill files."""
        for path in directory.rglob("*"):
            if path.is_file() and path.suffix in (".md", ".txt", ".json"):
                self._register_file(path)

    def _register_file(self, path: Path) -> None:
        """Register a single skill file."""
        content = path.read_text(encoding="utf-8")

        if path.suffix == ".json":
            self._register_json_skill(path, content)
        else:
            self._register_text_skill(path, content)

    def _register_text_skill(self, path: Path, content: str) -> None:
        """Register a markdown/text skill file with optional frontmatter."""
        meta_dict, body = _parse_frontmatter(content)
        name = meta_dict.get("name", path.stem)

        layer_str = meta_dict.get("type", meta_dict.get("layer", "session"))
        try:
            layer = ContextLayer(layer_str)
        except ValueError:
            layer = ContextLayer.SESSION

        tags_raw = meta_dict.get("tags", "")
        tags = [t.strip() for t in tags_raw.split(",") if t.strip()] if tags_raw else []

        metadata = SkillMetadata(
            name=name,
            description=meta_dict.get("description", ""),
            tags=tags,
            priority=int(meta_dict.get("priority", 50)),
            layer=layer,
            source_path=str(path),
        )
        self._catalog[name] = (metadata, body or content)

    def _register_json_skill(self, path: Path, content: str) -> None:
        """Register a JSON skill definition."""
        try:
            data = json.loads(content)
        except json.JSONDecodeError:
            return

        name = data.get("name", path.stem)
        body = data.get("content", "")

        layer_str = data.get("layer", "session")
        try:
            layer = ContextLayer(layer_str)
        except ValueError:
            layer = ContextLayer.SESSION

        metadata = SkillMetadata(
            name=name,
            description=data.get("description", ""),
            tags=data.get("tags", []),
            priority=data.get("priority", 50),
            layer=layer,
            source_path=str(path),
        )
        self._catalog[name] = (metadata, body)

    def get_skill_content(self, name: str) -> tuple[SkillMetadata, str] | None:
        """Get metadata and content for a skill by name."""
        return self._catalog.get(name)
