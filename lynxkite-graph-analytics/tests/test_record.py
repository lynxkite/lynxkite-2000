"""Unit tests for row records used by elementwise operations."""

from __future__ import annotations

import pandas as pd
import pytest
from lynxkite_graph_analytics.record import Record


def test_item_access_and_assignment_track_updates():
    series = pd.Series({"name": "Ada", "score": 10})
    record = Record(series, index=3)

    assert record["name"] == "Ada"
    record["score"] = 11
    record["active"] = True

    assert record["score"] == 11
    assert record["active"] is True
    assert series["score"] == 10
    assert record.get_updates() == {"score": 11, "active": True}


def test_keys_iteration_and_length_include_new_columns():
    record = Record(pd.Series({"first": 1, "second": 2}), index=0)
    record["second"] = 3
    record["third"] = 4

    assert record.keys() == ["first", "second", "third"]
    assert list(record) == ["first", "second", "third"]
    assert len(record) == 3


def test_attribute_access_and_assignment_use_columns():
    record = Record(pd.Series({"name": "Ada"}), index=1)

    assert record.name == "Ada"
    record.name = "Grace"
    record.active = True

    assert record.name == "Grace"
    assert record.active is True
    assert record.get_updates() == {"name": "Grace", "active": True}


def test_missing_column_raises_key_error_or_attribute_error():
    record = Record(pd.Series({"name": "Ada"}), index=0)

    with pytest.raises(KeyError):
        record["missing"]
    with pytest.raises(AttributeError, match="Record has no column 'missing'"):
        _ = record.missing


def test_index_and_get_updates():
    record = Record(pd.Series({"name": "Ada"}), index=("people", 4))
    record["name"] = "Grace"

    updates = record.get_updates()
    assert record.index == ("people", 4)
    assert updates == {"name": "Grace"}

    updates["name"] = "Katherine"
    updates["other"] = 1
    assert record.get_updates() == {"name": "Grace"}


@pytest.mark.parametrize(
    "key",
    [
        ("people", "name"),
        ["people", "name"],
    ],
)
def test_qualified_column_keys_are_normalized(key):
    record = Record(pd.Series({"name": "Ada"}), index=0)

    assert record[key] == "Ada"
    record[key] = "Grace"
    assert record.get_updates() == {"name": "Grace"}


def test_tuple_column_key_without_string_table_name_is_not_normalized():
    column = (1, "name")
    series = pd.Series(["Ada"], index=pd.Index([column], tupleize_cols=False))
    record = Record(series, index=0)

    assert record[column] == "Ada"


def test_record_turns_into_dict():
    record = Record(pd.Series({"name": "Ada", "score": 10}), index=0)
    record["score"] = 11
    record["active"] = True

    record_dict = dict(record)
    assert record_dict == {"name": "Ada", "score": 11, "active": True}
