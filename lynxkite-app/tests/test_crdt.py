from enum import Enum
import pycrdt
import pytest
from lynxkite_app.crdt import crdt_update


@pytest.fixture
def empty_dict_workspace():
    ydoc = pycrdt.Doc()
    ydoc["workspace"] = ws = pycrdt.Map()
    yield ws


@pytest.fixture
def empty_list_workspace():
    ydoc = pycrdt.Doc()
    ydoc["workspace"] = ws = pycrdt.Array()
    yield ws


class MyEnum(int, Enum):
    VALUE = 1


@pytest.mark.parametrize(
    "python_obj,expected",
    [
        (
            {
                "key1": "value1",
                "key2": {
                    "nested_key1": "nested_value1",
                    "nested_key2": ["nested_value2"],
                    "nested_key3": MyEnum.VALUE,
                },
            },
            {
                "key1": "value1",
                "key2": {
                    "nested_key1": "nested_value1",
                    "nested_key2": ["nested_value2"],
                    "nested_key3": "1",
                },
            },
        )
    ],
)
def test_crdt_update_with_dict(empty_dict_workspace, python_obj, expected):
    crdt_update(empty_dict_workspace, python_obj)
    assert empty_dict_workspace.to_py() == expected


@pytest.mark.parametrize(
    "python_obj,expected",
    [
        (
            [
                "value1",
                {"nested_key1": "nested_value1", "nested_key2": ["nested_value2"]},
                MyEnum.VALUE,
            ],
            [
                "value1",
                {"nested_key1": "nested_value1", "nested_key2": ["nested_value2"]},
                "1",
            ],
        ),
    ],
)
def test_crdt_update_with_list(empty_list_workspace, python_obj, expected):
    crdt_update(empty_list_workspace, python_obj)
    assert empty_list_workspace.to_py() == expected
