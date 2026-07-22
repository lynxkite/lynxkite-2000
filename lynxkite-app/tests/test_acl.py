from pathlib import Path

import pytest
import yaml

from lynxkite_app import acl


@pytest.fixture
def data_root(tmp_path, monkeypatch):
    acl.set_data_root(tmp_path)
    yield tmp_path
    acl.set_data_root(Path())


def _write_settings(folder: Path, data: dict) -> None:
    folder.mkdir(parents=True, exist_ok=True)
    (folder / "settings.yaml").write_text(yaml.dump(data), encoding="utf-8")


def test_resolve_folder():
    assert acl.resolve_folder(None) == ""
    assert acl.resolve_folder("") == ""
    assert acl.resolve_folder("team-a/project/foo.lynxkite.json") == "team-a/project/"
    assert acl.resolve_folder("uploads/data.csv") == "uploads/"
    assert acl.resolve_folder("team-a/project") == "team-a/project/"
    assert acl.resolve_folder("root.lynxkite.json") == ""


def test_child_settings_override_parent(data_root):
    _write_settings(
        data_root,
        {"acl": {"read": ["group:A"], "write": []}},
    )
    _write_settings(
        data_root / "team-a" / "project",
        {"acl": {"read": ["group:B"], "write": ["group:B"]}},
    )
    user = {"sub": "u1", "groups": ["B"]}
    assert acl.has_permission(user, "read", "team-a/project/ws.lynxkite.json")
    assert acl.has_permission(user, "write", "team-a/project/ws.lynxkite.json")
    assert not acl.has_permission(user, "write", "team-a/other/ws.lynxkite.json")


def test_write_implies_read(data_root):
    _write_settings(
        data_root / "shared",
        {"acl": {"read": [], "write": ["sub:writer"]}},
    )
    user = {"sub": "writer"}
    path = "shared/ws.lynxkite.json"
    assert acl.has_permission(user, "write", path)
    assert acl.has_permission(user, "read", path)


def test_default_deny(data_root):
    user = {"sub": "u1"}
    assert not acl.has_permission(user, "read", "any/ws.lynxkite.json")


def test_wildcard_requires_authenticated_user(data_root):
    _write_settings(data_root, {"acl": {"read": ["*"], "write": []}})
    assert acl.has_permission({"sub": "u1"}, "read", "x.lynxkite.json")
    assert not acl.has_permission({}, "read", "x.lynxkite.json")


def test_parent_folder_inheritance(data_root):
    _write_settings(
        data_root / "team-a",
        {"acl": {"read": ["group:Team"], "write": []}},
    )
    user = {"sub": "u1", "groups": ["Team"]}
    assert acl.has_permission(user, "read", "team-a/nested/ws.lynxkite.json")


def test_partial_override_keeps_parent_read(data_root):
    _write_settings(data_root, {"acl": {"read": ["*"], "write": []}})
    _write_settings(
        data_root / "team-a",
        {"acl": {"write": ["group:Eng"]}},
    )
    reader = {"sub": "r1"}
    writer = {"sub": "w1", "groups": ["Eng"]}
    assert acl.has_permission(reader, "read", "team-a/ws.lynxkite.json")
    assert not acl.has_permission(reader, "write", "team-a/ws.lynxkite.json")
    assert acl.has_permission(writer, "write", "team-a/ws.lynxkite.json")


def test_invalid_action():
    with pytest.raises(ValueError, match="Invalid action"):
        acl.has_permission({"sub": "u1"}, "execute", "x.lynxkite.json")  # type: ignore[arg-type]


def test_effective_permissions(data_root):
    _write_settings(
        data_root,
        {"acl": {"read": ["*"], "write": ["sub:admin"]}},
    )
    reader = {"sub": "reader"}
    admin = {"sub": "admin"}
    assert acl.effective_permissions(reader, "a.lynxkite.json") == {
        "read": True,
        "write": False,
    }
    assert acl.effective_permissions(admin, "a.lynxkite.json") == {
        "read": True,
        "write": True,
    }
