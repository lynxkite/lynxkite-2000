"""Row record wrapper used by elementwise operations."""

from __future__ import annotations

import typing

import pandas as pd

ColumnKey = typing.Hashable
RowIndex = typing.Hashable


class Record:
    """Row wrapper with attribute access and mutation tracking."""

    def __init__(self, series: pd.Series, index: RowIndex):
        object.__setattr__(self, "_data", series)
        object.__setattr__(self, "_index", index)
        object.__setattr__(self, "_updates", {})

    def __getitem__(self, key: ColumnKey) -> typing.Any:
        normalized_key = self._normalize_column_key(key)
        if normalized_key in self._updates:
            return self._updates[normalized_key]
        return self._data[normalized_key]

    def __setitem__(self, key: ColumnKey, value: typing.Any) -> None:
        normalized_key = self._normalize_column_key(key)
        self._updates[normalized_key] = value

    def __getattr__(self, key: str) -> typing.Any:
        if key.startswith("_"):
            return object.__getattribute__(self, key)
        try:
            return self[key]
        except KeyError:
            raise AttributeError(f"Record has no column '{key}'")

    def __setattr__(self, key: str, value: typing.Any) -> None:
        if key.startswith("_"):
            object.__setattr__(self, key, value)
        else:
            self[key] = value

    @property
    def index(self) -> RowIndex:
        return self._index

    def get_updates(self) -> dict[ColumnKey, typing.Any]:
        return self._updates.copy()

    @staticmethod
    def _normalize_column_key(key: ColumnKey) -> ColumnKey:
        if isinstance(key, (tuple, list)) and len(key) == 2 and isinstance(key[0], str):
            return key[1]
        return key
