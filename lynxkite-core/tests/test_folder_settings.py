"""Tests for per-folder settings.yaml."""

from pathlib import Path

import yaml
from lynxkite_core.folder_settings import (
    ACLSettings,
    FolderSettings,
    LIMBoxConfig,
    get_local_settings,
    get_total_settings,
    set_local_settings,
)


def test_get_total_settings_merges_parent_and_child(tmp_path: Path):
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

    merged = get_total_settings(tmp_path, "heavy/ws.lynxkite.json")
    assert merged.lim["Box A"] == LIMBoxConfig(memory="16Gi", cpu="2")


def test_get_total_settings_missing_returns_defaults(tmp_path: Path):
    assert get_total_settings(tmp_path, "ws.lynxkite.json") == FolderSettings()


def test_get_total_settings_returns_typed_settings(tmp_path: Path):
    (tmp_path / "settings.yaml").write_text(
        yaml.dump({"lim": {"Box A": {"memory": "4Gi"}}}),
        encoding="utf-8",
    )
    merged = get_total_settings(tmp_path, "ws.lynxkite.json")
    assert isinstance(merged, FolderSettings)
    assert merged.lim["Box A"] == LIMBoxConfig(memory="4Gi")


def test_get_total_settings_merges_multiple_levels(tmp_path: Path):
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

    merged = get_total_settings(tmp_path, "team/project/ws.lynxkite.json")
    assert merged.lim["Box A"] == LIMBoxConfig(cpu="3", memory="8Gi")


def test_get_total_settings_keeps_boxes_isolated(tmp_path: Path):
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

    merged = get_total_settings(tmp_path, "team/ws.lynxkite.json")
    assert merged.lim["Box A"] == LIMBoxConfig(cpu="1")
    assert merged.lim["Box B"] == LIMBoxConfig(cpu="2", memory="16Gi")


def test_get_total_settings_accepts_folder_path_without_suffix(tmp_path: Path):
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

    merged = get_total_settings(tmp_path, "team")
    assert merged.lim["Box A"] == LIMBoxConfig(cpu="1", memory="8Gi")


def test_get_total_settings_merges_acl_parent_and_child(tmp_path: Path):
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

    merged = get_total_settings(tmp_path, "team/ws.lynxkite.json")
    assert merged.acl == ACLSettings(read=["*"], write=["group:Eng"])


def test_get_local_settings_excludes_inherited_settings(tmp_path: Path):
    (tmp_path / "settings.yaml").write_text(
        yaml.dump({"acl": {"read": ["*"]}, "lim": {"Box A": {"cpu": "1"}}}),
        encoding="utf-8",
    )
    child = tmp_path / "team"
    child.mkdir()
    (child / "settings.yaml").write_text(
        yaml.dump({"acl": {"write": ["group:Eng"]}}),
        encoding="utf-8",
    )

    local = get_local_settings(tmp_path, "team")
    assert local == FolderSettings(acl=ACLSettings(write=["group:Eng"]))


def test_set_local_settings_writes_settings_for_folder(tmp_path: Path):
    folder = tmp_path / "team"
    folder.mkdir()
    settings = FolderSettings(
        acl=ACLSettings(read=["group:Eng"]),
        lim={"Box A": LIMBoxConfig(cpu="2", memory="8Gi")},
    )

    set_local_settings(tmp_path, "team", settings)

    assert get_local_settings(tmp_path, "team") == settings
