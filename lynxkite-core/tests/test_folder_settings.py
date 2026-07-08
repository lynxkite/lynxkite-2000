"""Tests for per-folder settings.yaml."""

from pathlib import Path

import yaml

from lynxkite_core.folder_settings import resolve_lim_for_box


def test_resolve_lim_for_box_merges_parent_and_child(tmp_path: Path):
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

    assert resolve_lim_for_box(tmp_path, "heavy/ws.lynxkite.json", "Box A") == {
        "memory": "16Gi",
        "cpu": "2",
    }


def test_resolve_lim_for_box_missing_returns_none(tmp_path: Path):
    assert resolve_lim_for_box(tmp_path, "ws.lynxkite.json", "Box A") is None


def test_resolve_lim_for_box_returns_dict(tmp_path: Path):
    (tmp_path / "settings.yaml").write_text(
        yaml.dump({"lim": {"Box A": {"memory": "4Gi"}}}),
        encoding="utf-8",
    )
    assert resolve_lim_for_box(tmp_path, "ws.lynxkite.json", "Box A") == {"memory": "4Gi"}


def test_resolve_lim_for_box_merges_multiple_levels(tmp_path: Path):
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

    assert resolve_lim_for_box(tmp_path, "team/project/ws.lynxkite.json", "Box A") == {
        "cpu": "3",
        "memory": "8Gi",
    }


def test_resolve_lim_for_box_only_returns_target_box(tmp_path: Path):
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

    assert resolve_lim_for_box(tmp_path, "team/ws.lynxkite.json", "Box A") == {"cpu": "1"}
    assert resolve_lim_for_box(tmp_path, "team/ws.lynxkite.json", "Box B") == {
        "cpu": "2",
        "memory": "16Gi",
    }


def test_resolve_lim_for_box_accepts_workspace_path_without_suffix(tmp_path: Path):
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

    assert resolve_lim_for_box(tmp_path, "team", "Box A") == {"cpu": "1", "memory": "8Gi"}
