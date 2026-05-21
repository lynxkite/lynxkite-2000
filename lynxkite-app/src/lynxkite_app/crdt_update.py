"""For updating a pycrdt object to match a Python object."""

import enum
from typing import Any
import pycrdt


def crdt_update(
    crdt_obj: pycrdt.Map[Any] | pycrdt.Array[Any],
    python_obj: dict | list,
    non_collaborative_fields: set[str] = set(),
):
    """Update a CRDT object to match a Python object.

    The types between the CRDT object and the Python object must match. If the Python object
    is a dict, the CRDT object must be a Map. If the Python object is a list, the CRDT object
    must be an Array.

    Args:
        crdt_obj: The CRDT object, that will be updated to match the Python object.
        python_obj: The Python object to update with.
        non_collaborative_fields: List of fields to treat as a black box. Black boxes are
            updated as a whole, instead of having a fine-grained data structure to edit
            collaboratively. Useful for complex fields that contain auto-generated data or
            metadata.
            The default is an empty set.

    Raises:
        ValueError: If the Python object provided is not a dict or list.
    """
    if isinstance(python_obj, dict):
        assert isinstance(crdt_obj, pycrdt.Map), f"expected CRDT Map, got {type(crdt_obj)}"
        _crdt_update_dict(crdt_obj, python_obj, non_collaborative_fields)
    elif isinstance(python_obj, list):
        assert isinstance(crdt_obj, pycrdt.Array), f"expected CRDT Array, got {type(crdt_obj)}"
        _crdt_update_list(crdt_obj, python_obj, non_collaborative_fields)
    else:
        raise ValueError("Invalid type:", python_obj)


def _crdt_update_dict(
    crdt_obj: pycrdt.Map[Any], python_obj: dict, non_collaborative_fields: set[str]
):
    crdt_py = crdt_obj.to_py()
    if crdt_py == python_obj:
        return
    # Delete keys that are gone.
    for key in list(crdt_obj.keys()):
        if key not in python_obj:
            del crdt_obj[key]
    for key, value in python_obj.items():
        if isinstance(value, enum.Enum):
            value = str(value.value)
        crdt_value = crdt_obj.get(key)
        if crdt_value == value:
            # No change, skip.
            pass
        elif key in non_collaborative_fields:
            # Non-collaborative field, update as black box.
            crdt_obj[key] = value
        elif not crdt_value:
            # New key, add it.
            if isinstance(value, dict):
                crdt_obj[key] = pycrdt.Map()
                crdt_update(crdt_obj[key], value, non_collaborative_fields)
            elif isinstance(value, list):
                crdt_obj[key] = pycrdt.Array()
                crdt_update(crdt_obj[key], value, non_collaborative_fields)
            else:
                crdt_obj[key] = value
        elif isinstance(crdt_value, (pycrdt.Array, pycrdt.Map)):
            # Existing Map or Array, update it.
            crdt_update(
                crdt_obj[key],
                value,  # ty: ignore[invalid-argument-type]
                non_collaborative_fields,
            )
        else:
            # Existing non-CRDT value, update as black box.
            crdt_obj[key] = value


def _crdt_update_list(
    crdt_obj: pycrdt.Array[Any],
    python_obj: list,
    non_collaborative_fields: set[str],
    key="id",
):
    crdt_py = crdt_obj.to_py()
    if crdt_py == python_obj:
        return
    if all(isinstance(v, dict) and key in v for v in python_obj):
        _crdt_update_list_by_key(crdt_obj, python_obj, non_collaborative_fields, key)
    else:
        _crdt_update_list_by_index(crdt_obj, python_obj, non_collaborative_fields)


def _crdt_update_list_by_key(
    crdt_obj: pycrdt.Array[Any],
    python_obj: list,
    non_collaborative_fields: set[str],
    key: str,
):
    # Key-based matching: use the key field to track additions/deletions.
    new_keys = {v[key] for v in python_obj if isinstance(v, dict) and key in v}
    current_keys = [v.get(key) if isinstance(v, (dict, pycrdt.Map)) else None for v in crdt_obj]
    # Delete items no longer present and duplicates.
    for i in range(len(crdt_obj) - 1, -1, -1):
        if current_keys[i] not in new_keys or current_keys[i] in current_keys[:i]:
            del crdt_obj[i]
            del current_keys[i]
    # Update existing items or append new ones.
    for i, value in enumerate(python_obj):
        value_key = value.get(key)
        current_index = current_keys.index(value_key) if value_key in current_keys else None
        if current_index is not None:
            if current_index != i:
                crdt_obj.move(current_index, i)
                current_keys.insert(i, current_keys.pop(current_index))
            crdt_update(crdt_obj[i], value, non_collaborative_fields)
        else:
            new_map = pycrdt.Map()
            crdt_obj.insert(i, new_map)
            current_keys.insert(i, value_key)
            crdt_update(new_map, value, non_collaborative_fields)


def _crdt_update_list_by_index(
    crdt_obj: pycrdt.Array[Any],
    python_obj: list,
    non_collaborative_fields: set[str],
):
    # Index-based matching.
    for i, value in enumerate(python_obj):
        if isinstance(value, enum.Enum):
            value = str(value.value)
        if i < len(crdt_obj):
            # Update in place.
            if isinstance(value, dict):
                if not isinstance(crdt_obj[i], pycrdt.Map):
                    crdt_obj[i] = pycrdt.Map()  # ty: ignore[invalid-assignment]
                crdt_update(crdt_obj[i], value, non_collaborative_fields)
            elif isinstance(value, list):
                if not isinstance(crdt_obj[i], pycrdt.Array):
                    crdt_obj[i] = pycrdt.Array()  # ty: ignore[invalid-assignment]
                crdt_update(crdt_obj[i], value, non_collaborative_fields)
            else:
                if crdt_obj[i] != value:
                    crdt_obj[i] = value  # ty: ignore[invalid-assignment]
        else:
            # Append new item.
            if isinstance(value, dict):
                crdt_obj.append(pycrdt.Map())  # ty: ignore[invalid-argument-type]
                crdt_update(crdt_obj[i], value, non_collaborative_fields)
            elif isinstance(value, list):
                crdt_obj.append(pycrdt.Array())  # ty: ignore[invalid-argument-type]
                crdt_update(crdt_obj[i], value, non_collaborative_fields)
            else:
                crdt_obj.append(value)  # ty: ignore[invalid-argument-type]
    # Delete items that are no longer present.
    while len(crdt_obj) > len(python_obj):
        del crdt_obj[-1]
