"""Tests for per-folder settings.yaml."""

from pathlib import Path

import yaml

from lynxkite_core.folder_settings import resolve_flat_section, resolve_settings_section


def test_resolve_settings_section_merges_parent_and_child(tmp_path: Path):
    (tmp_path / "settings.yaml").write_text(
        yaml.dump({"lim": {"Box A": {"memory": "8Gi", "cpu": "2"}}}),
        encoding="utf-8",
    )
    child = tmp_path / "heavy"
    child.mkdir()
    (child / "settings.yaml").write_text(
        yaml.dump({"lim": {"Box A": {"memory": "16Gi"}}}),
        encoding="utf-8",
    )

    merged = resolve_settings_section(tmp_path, "heavy/ws.lynxkite.json", "lim")
    assert merged["Box A"] == {
        "memory": "16Gi",
        "cpu": "2",
    }


def test_resolve_settings_section_missing_returns_empty(tmp_path: Path):
    assert resolve_settings_section(tmp_path, "ws.lynxkite.json", "lim") == {}


def test_resolve_settings_section_returns_dict(tmp_path: Path):
    (tmp_path / "settings.yaml").write_text(
        yaml.dump({"lim": {"Box A": {"memory": "4Gi"}}}),
        encoding="utf-8",
    )
    merged = resolve_settings_section(tmp_path, "ws.lynxkite.json", "lim")
    assert merged["Box A"] == {"memory": "4Gi"}


def test_resolve_settings_section_merges_multiple_levels(tmp_path: Path):
    (tmp_path / "settings.yaml").write_text(
        yaml.dump({"lim": {"Box A": {"cpu": "1", "memory": "4Gi"}}}),
        encoding="utf-8",
    )
    child = tmp_path / "team"
    child.mkdir()
    (child / "settings.yaml").write_text(
        yaml.dump({"lim": {"Box A": {"memory": "8Gi"}}}),
        encoding="utf-8",
    )
    grandchild = child / "project"
    grandchild.mkdir()
    (grandchild / "settings.yaml").write_text(
        yaml.dump({"lim": {"Box A": {"cpu": "3"}}}),
        encoding="utf-8",
    )

    merged = resolve_settings_section(tmp_path, "team/project/ws.lynxkite.json", "lim")
    assert merged["Box A"] == {
        "cpu": "3",
        "memory": "8Gi",
    }


def test_resolve_settings_section_keeps_boxes_isolated(tmp_path: Path):
    (tmp_path / "settings.yaml").write_text(
        yaml.dump(
            {
                "lim": {
                    "Box A": {"cpu": "1"},
                    "Box B": {"cpu": "2"},
                }
            }
        ),
        encoding="utf-8",
    )
    child = tmp_path / "team"
    child.mkdir()
    (child / "settings.yaml").write_text(
        yaml.dump({"lim": {"Box B": {"memory": "16Gi"}}}),
        encoding="utf-8",
    )

    merged = resolve_settings_section(tmp_path, "team/ws.lynxkite.json", "lim")
    assert merged["Box A"] == {"cpu": "1"}
    assert merged["Box B"] == {
        "cpu": "2",
        "memory": "16Gi",
    }


def test_resolve_settings_section_accepts_workspace_path_without_suffix(tmp_path: Path):
    (tmp_path / "settings.yaml").write_text(
        yaml.dump({"lim": {"Box A": {"cpu": "1"}}}),
        encoding="utf-8",
    )
    child = tmp_path / "team"
    child.mkdir()
    (child / "settings.yaml").write_text(
        yaml.dump({"lim": {"Box A": {"memory": "8Gi"}}}),
        encoding="utf-8",
    )

    merged = resolve_settings_section(tmp_path, "team", "lim")
    assert merged["Box A"] == {"cpu": "1", "memory": "8Gi"}


def test_resolve_flat_section_child_overrides_parent(tmp_path: Path):
    (tmp_path / "settings.yaml").write_text(
        yaml.dump({"acl": {"read": ["*"], "write": []}}),
        encoding="utf-8",
    )
    child = tmp_path / "team"
    child.mkdir()
    (child / "settings.yaml").write_text(
        yaml.dump({"acl": {"write": ["group:Eng"]}}),
        encoding="utf-8",
    )

    merged = resolve_flat_section(tmp_path, "team/ws.lynxkite.json", "acl")
    assert merged == {"read": ["*"], "write": ["group:Eng"]}
