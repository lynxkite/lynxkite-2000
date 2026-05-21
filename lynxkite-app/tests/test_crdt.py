from enum import Enum

import pycrdt
import pytest

from lynxkite_app.crdt_update import crdt_update


class MyEnum(int, Enum):
    VALUE = 1


def _to_crdt(value):
    if isinstance(value, dict):
        return pycrdt.Map()
    if isinstance(value, list):
        return pycrdt.Array()
    return value


def _populate_map(target_map, source_dict):
    for key, value in source_dict.items():
        if isinstance(value, dict):
            target_map[key] = pycrdt.Map()
            _populate_map(target_map[key], value)
        elif isinstance(value, list):
            target_map[key] = pycrdt.Array()
            _populate_array(target_map[key], value)
        else:
            target_map[key] = value


def _populate_array(target_array, source_list):
    for value in source_list:
        if isinstance(value, dict):
            target_array.append(pycrdt.Map())
            _populate_map(target_array[-1], value)
        elif isinstance(value, list):
            target_array.append(pycrdt.Array())
            _populate_array(target_array[-1], value)
        else:
            target_array.append(value)


def _make_root(value):
    doc = pycrdt.Doc()
    root = _to_crdt(value)
    doc["workspace"] = root
    if isinstance(value, dict):
        _populate_map(root, value)
    elif isinstance(value, list):
        _populate_array(root, value)
    return root


def _assert_update(initial, target, expected=None, *, non_collaborative_fields=()):
    root = _make_root(initial)
    expected_result = target if expected is None else expected
    crdt_update(root, target, set(non_collaborative_fields))
    assert root.to_py() == expected_result
    return root


def test_rejects_invalid_python_object_type():
    root = _make_root({})
    with pytest.raises(ValueError, match="Invalid type"):
        crdt_update(root, "not-a-dict-or-list")  # ty: ignore[invalid-argument-type]


def test_asserts_for_top_level_type_mismatch_dict_to_array():
    root = _make_root([])
    with pytest.raises(AssertionError, match="expected CRDT Map"):
        crdt_update(root, {"k": "v"})


def test_asserts_for_top_level_type_mismatch_list_to_map():
    root = _make_root({})
    with pytest.raises(AssertionError, match="expected CRDT Array"):
        crdt_update(root, ["v"])


def test_dict_deletes_missing_keys_and_updates_scalars():
    _assert_update(
        {"keep": "old", "delete": "me"},
        {"keep": "new", "add": 7},
        {"keep": "new", "add": 7},
    )


def test_dict_creates_nested_structures_and_converts_enum_values():
    _assert_update(
        {},
        {
            "name": "x",
            "nested": {"arr": ["v", MyEnum.VALUE], "enum": MyEnum.VALUE},
        },
        {
            "name": "x",
            "nested": {"arr": ["v", "1"], "enum": "1"},
        },
    )


def test_dict_updates_nested_map_and_array_in_place():
    root = _make_root({"node": {"items": [1, 2], "flag": "on"}})
    crdt_update(root, {"node": {"items": [1, 3, 4], "flag": "off"}})

    assert root.to_py() == {"node": {"items": [1, 3, 4], "flag": "off"}}
    assert isinstance(root["node"], pycrdt.Map)
    assert isinstance(root["node"]["items"], pycrdt.Array)


def test_dict_non_collaborative_fields_are_replaced_as_black_boxes():
    _assert_update(
        {"meta": {"old": 1, "keep": 2}},
        {"meta": {"new": 3}},
        {"meta": {"new": 3}},
        non_collaborative_fields={"meta"},
    )


def test_dict_handles_existing_falsey_values():
    _assert_update(
        {"zero": 0, "empty": "", "false": False},
        {"zero": {"nested": 1}, "empty": "filled", "false": True},
        {"zero": {"nested": 1}, "empty": "filled", "false": True},
    )


def test_dict_existing_map_with_non_container_target_raises_value_error():
    root = _make_root({"nested": {"x": 1}})
    with pytest.raises(ValueError, match="Invalid type"):
        crdt_update(root, {"nested": 5})


def test_dict_existing_array_with_non_container_target_raises_value_error():
    root = _make_root({"nested": [1, 2]})
    with pytest.raises(ValueError, match="Invalid type"):
        crdt_update(root, {"nested": 5})


def test_list_index_based_update_append_truncate_and_enum_conversion():
    _assert_update(
        [
            "old",
            {"k": "v"},
            [0],
            "drop",
        ],
        [
            "new",
            {"k": "updated", "new": 1},
            [0, 1],
            MyEnum.VALUE,
        ],
        [
            "new",
            {"k": "updated", "new": 1},
            [0, 1],
            "1",
        ],
    )


def test_list_index_based_replaces_mismatched_container_types():
    _assert_update(
        [{"k": 1}, [1, 2]],
        [[9], {"z": 2}],
        [[9], {"z": 2}],
    )


def test_list_id_based_matching_updates_deletes_and_appends():
    _assert_update(
        [
            {"id": "a", "v": 1},
            {"id": "b", "v": 2},
            {"id": "drop1", "v": 0},
            {"id": "drop2", "v": 0},
        ],
        [
            {"id": "b", "v": 20},
            {"id": "a", "v": 10},
            {"id": "c", "v": 30},
        ],
        [
            {"id": "b", "v": 20},
            {"id": "a", "v": 10},
            {"id": "c", "v": 30},
        ],
    )


def test_list_id_based_matching_with_duplicate_ids_updates_last_instance():
    _assert_update(
        [{"id": "dup", "v": 1}, {"id": "dup", "v": 2}],
        [{"id": "dup", "v": 3}],
        [{"id": "dup", "v": 3}],
    )


def test_list_inserting_non_dict_items_goes_to_index_based_update():
    _assert_update(
        [{"id": "a", "v": 1}],
        [{"id": "a", "v": 2}, "not-a-dict"],
        [{"id": "a", "v": 2}, "not-a-dict"],
    )
