"""Tests for save_your_tokens.skills.registry — skill discovery."""

import json

from save_your_tokens.core.spec import ContextLayer
from save_your_tokens.skills.registry import SkillRegistry, _parse_frontmatter


class TestParseFrontmatter:
    def test_no_frontmatter(self):
        meta, body = _parse_frontmatter("just plain content")
        assert meta == {}
        assert body == "just plain content"

    def test_valid_frontmatter(self):
        content = "---\nname: debug\ndescription: A debugging skill\n---\nBody here."
        meta, body = _parse_frontmatter(content)
        assert meta["name"] == "debug"
        assert meta["description"] == "A debugging skill"
        assert body == "Body here."

    def test_unclosed_frontmatter(self):
        content = "---\nname: debug\nno closing marker"
        meta, body = _parse_frontmatter(content)
        assert meta == {}
        assert body == content

    def test_empty_body(self):
        content = "---\nname: test\n---\n"
        meta, body = _parse_frontmatter(content)
        assert meta["name"] == "test"
        assert body == ""


class TestRegistryScanText:
    def test_scan_md_files(self, tmp_path):
        skill_file = tmp_path / "debug.md"
        skill_file.write_text(
            "---\nname: debug\ndescription: Debug skill\n"
            "tags: core, dev\npriority: 10\n---\nBody content.",
            encoding="utf-8",
        )

        registry = SkillRegistry()
        registry.add_scan_dir(tmp_path)
        count = registry.scan()

        assert count == 1
        assert "debug" in registry.catalog
        meta = registry.catalog["debug"]
        assert meta.description == "Debug skill"
        assert meta.tags == ["core", "dev"]
        assert meta.priority == 10

    def test_scan_txt_files(self, tmp_path):
        skill_file = tmp_path / "helper.txt"
        skill_file.write_text("plain text skill content", encoding="utf-8")

        registry = SkillRegistry()
        registry.add_scan_dir(tmp_path)
        count = registry.scan()

        assert count == 1
        assert "helper" in registry.catalog  # Uses stem as name

    def test_default_layer_session(self, tmp_path):
        skill_file = tmp_path / "basic.md"
        skill_file.write_text("no frontmatter", encoding="utf-8")

        registry = SkillRegistry()
        registry.add_scan_dir(tmp_path)
        registry.scan()

        assert registry.catalog["basic"].layer == ContextLayer.SESSION

    def test_custom_layer(self, tmp_path):
        skill_file = tmp_path / "sys.md"
        skill_file.write_text(
            "---\nname: sys\nlayer: persistent\n---\nSystem skill.", encoding="utf-8"
        )

        registry = SkillRegistry()
        registry.add_scan_dir(tmp_path)
        registry.scan()

        assert registry.catalog["sys"].layer == ContextLayer.PERSISTENT

    def test_invalid_layer_defaults_session(self, tmp_path):
        skill_file = tmp_path / "bad.md"
        skill_file.write_text("---\nname: bad\nlayer: bogus\n---\nContent.", encoding="utf-8")

        registry = SkillRegistry()
        registry.add_scan_dir(tmp_path)
        registry.scan()

        assert registry.catalog["bad"].layer == ContextLayer.SESSION


class TestRegistryScanJson:
    def test_scan_json_skill(self, tmp_path):
        skill_data = {
            "name": "json-skill",
            "description": "A JSON skill",
            "content": "JSON body",
            "tags": ["test"],
            "priority": 20,
            "layer": "ephemeral",
        }
        skill_file = tmp_path / "skill.json"
        skill_file.write_text(json.dumps(skill_data), encoding="utf-8")

        registry = SkillRegistry()
        registry.add_scan_dir(tmp_path)
        count = registry.scan()

        assert count == 1
        meta = registry.catalog["json-skill"]
        assert meta.description == "A JSON skill"
        assert meta.tags == ["test"]
        assert meta.priority == 20
        assert meta.layer == ContextLayer.EPHEMERAL

    def test_invalid_json_skipped(self, tmp_path):
        skill_file = tmp_path / "broken.json"
        skill_file.write_text("{invalid json", encoding="utf-8")

        registry = SkillRegistry()
        registry.add_scan_dir(tmp_path)
        count = registry.scan()

        assert count == 0


class TestRegistryGetContent:
    def test_get_skill_content(self, tmp_path):
        skill_file = tmp_path / "test.md"
        skill_file.write_text("---\nname: test\n---\nThe body.", encoding="utf-8")

        registry = SkillRegistry()
        registry.add_scan_dir(tmp_path)
        registry.scan()

        result = registry.get_skill_content("test")
        assert result is not None
        meta, content = result
        assert meta.name == "test"
        assert content == "The body."

    def test_get_nonexistent_returns_none(self):
        registry = SkillRegistry()
        assert registry.get_skill_content("nope") is None


class TestRegistryEdgeCases:
    def test_nonexistent_dir_ignored(self):
        registry = SkillRegistry()
        registry.add_scan_dir("/this/does/not/exist")
        assert registry.scan() == 0

    def test_rescan_clears_catalog(self, tmp_path):
        skill_file = tmp_path / "a.md"
        skill_file.write_text("content a", encoding="utf-8")

        registry = SkillRegistry()
        registry.add_scan_dir(tmp_path)
        registry.scan()
        assert len(registry.catalog) == 1

        skill_file.unlink()
        registry.scan()
        assert len(registry.catalog) == 0

    def test_recursive_scan(self, tmp_path):
        subdir = tmp_path / "nested" / "deep"
        subdir.mkdir(parents=True)
        (subdir / "inner.md").write_text("nested content", encoding="utf-8")

        registry = SkillRegistry()
        registry.add_scan_dir(tmp_path)
        count = registry.scan()

        assert count == 1
        assert "inner" in registry.catalog
