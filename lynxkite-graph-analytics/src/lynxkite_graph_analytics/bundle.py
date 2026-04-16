"""Bundle data structures and related helpers."""

import dataclasses
import enum
import typing

import networkx as nx
import pandas as pd
import polars as pl

from .relation_definition import RelationDefinition


class BundleMergeMode(enum.StrEnum):
    """How `Bundle.merge` handles name collisions.

    - `must_be_unique`: all DataFrame, relation, and `other` names must be unique
      across inputs.
    - `rename_non_unique`: colliding names are made unique by suffixing `_1`, `_2`, ...
      (for example `records_1`, `records_2`).
    - `concatenate_non_unique`: DataFrames with the same name are concatenated.
      Relations and `other` entries with the same name must be equal.
    """

    must_be_unique = "must be unique"
    rename_non_unique = "rename non-unique"
    concatenate_non_unique = "concatenate non-unique"


@dataclasses.dataclass
class Bundle:
    """A collection of DataFrames and other data.

    Can efficiently represent a knowledge graph (homogeneous or heterogeneous) or tabular data.

    By convention, if it contains a single DataFrame, it is called `df`.
    If it contains a homogeneous graph, it is represented as two DataFrames called `nodes` and
    `edges`.

    Attributes:
        dfs: Named DataFrames.
        relations: Metadata that describes the roles of each DataFrame.
            Can be empty, if the bundle is just one or more DataFrames.
        other: Other data, such as a trained model.
    """

    dfs: dict[str, pd.DataFrame] = dataclasses.field(default_factory=dict)
    relations: list[RelationDefinition] = dataclasses.field(default_factory=list)
    other: dict[str, typing.Any] = dataclasses.field(default_factory=dict)

    @classmethod
    def from_nx(cls, graph: nx.Graph):
        edges = nx.to_pandas_edgelist(graph)
        d = dict(graph.nodes(data=True))
        nodes = pd.DataFrame(d.values(), index=list(d.keys()))
        nodes["id"] = nodes.index
        if "index" in nodes.columns:
            nodes.drop(columns=["index"], inplace=True)
        return cls(
            dfs={"edges": edges, "nodes": nodes},
            relations=[
                RelationDefinition(
                    name="edges",
                    df="edges",
                    source_column="source",
                    target_column="target",
                    source_table="nodes",
                    target_table="nodes",
                    source_key="id",
                    target_key="id",
                )
            ],
        )

    @classmethod
    def from_df(cls, df: pd.DataFrame):
        return cls(dfs={"df": df})

    def to_nx(self):
        # TODO: Use relations.
        graph = nx.DiGraph()
        if "nodes" in self.dfs:
            df = self.dfs["nodes"]
            if df.index.name != "id":
                df = df.set_index("id")
            graph.add_nodes_from(df.to_dict("index").items())
        if "edges" in self.dfs:
            edges = self.dfs["edges"]
            graph.add_edges_from(
                [
                    (
                        e["source"],
                        e["target"],
                        {k: e[k] for k in edges.columns if k not in ["source", "target"]},
                    )
                    for e in edges.to_records()
                ]
            )
        return graph

    def copy(self):
        """
        Returns a shallow copy of the bundle. The Bundle and its containers are new, but
        the DataFrames and RelationDefinitions are shared. (The contents of `other` are also shared.)
        """
        return Bundle(
            dfs=dict(self.dfs),
            relations=list(self.relations),
            other=dict(self.other),
        )

    def metadata(self):
        """JSON-serializable information about the bundle, metadata only."""
        return {
            "dataframes": {
                name: {
                    "key": name,
                    "columns": sorted(str(c) for c in df.columns),
                }
                for name, df in self.dfs.items()
            },
            "relations": [dataclasses.asdict(relation) for relation in self.relations],
            "other": {
                k: {"key": k, **getattr(v, "metadata", lambda: {})()} for k, v in self.other.items()
            },
        }

    def to_table_view(self, limit: int = 100):
        """Converts the bundle to a format suitable for display as tables in the frontend."""
        return BundleTableView.from_bundle(self, limit=limit)


def merge_bundles(
    bundles: list["Bundle"], merge_mode: BundleMergeMode = BundleMergeMode.must_be_unique
):
    """Merges multiple bundles into a new bundle. Does not modify the original bundles."""
    match merge_mode:
        case BundleMergeMode.must_be_unique:
            return _merge_bundles_must_be_unique(bundles)
        case BundleMergeMode.rename_non_unique:
            return _merge_bundles_rename_non_unique(bundles)
        case BundleMergeMode.concatenate_non_unique:
            return _merge_bundles_concatenate_non_unique(bundles)
        case _:
            raise ValueError(f"Unknown merge mode: {merge_mode}")


def _merge_bundles_must_be_unique(bundles: list["Bundle"]) -> Bundle:
    """Merge bundles while requiring all names to be unique across inputs."""
    result = Bundle()
    non_unique_dfs = _find_overlaps(name for b in bundles for name in b.dfs.keys())
    non_unique_relations = _find_overlaps(
        relation.name for b in bundles for relation in b.relations
    )
    non_unique_other = _find_overlaps(k for b in bundles for k in b.other.keys())
    assert not non_unique_dfs, f"Non-unique tables: {non_unique_dfs}"
    assert not non_unique_relations, f"Non-unique relations: {non_unique_relations}"
    assert not non_unique_other, f"Non-unique other: {non_unique_other}"

    for b in bundles:
        result.dfs.update(b.dfs)
        result.relations.extend(b.relations)
        result.other.update(b.other)
    return result


def _merge_bundles_rename_non_unique(bundles: list["Bundle"]) -> Bundle:
    """Merge bundles by suffixing colliding names with incrementing counters."""
    result = Bundle()
    non_unique_dfs = _find_overlaps(name for b in bundles for name in b.dfs.keys())
    non_unique_relations = _find_overlaps(
        relation.name for b in bundles for relation in b.relations
    )
    non_unique_other = _find_overlaps(k for b in bundles for k in b.other.keys())

    df_counters: dict[str, int] = {}
    relation_counters: dict[str, int] = {}
    other_counters: dict[str, int] = {}

    for b in bundles:
        table_name_map: dict[str, str] = {}
        for name, df in b.dfs.items():
            if name in non_unique_dfs:
                i = df_counters.get(name, 0) + 1
                df_counters[name] = i
                new_name = f"{name}_{i}"
            else:
                new_name = name
            table_name_map[name] = new_name
            result.dfs[new_name] = df

        for relation in b.relations:
            if relation.name in non_unique_relations:
                i = relation_counters.get(relation.name, 0) + 1
                relation_counters[relation.name] = i
                new_relation_name = f"{relation.name}_{i}"
            else:
                new_relation_name = relation.name

            result.relations.append(
                dataclasses.replace(
                    relation,
                    name=new_relation_name,
                    df=table_name_map.get(relation.df, relation.df),
                    source_table=table_name_map.get(relation.source_table, relation.source_table),
                    target_table=table_name_map.get(relation.target_table, relation.target_table),
                )
            )

        for name, value in b.other.items():
            if name in non_unique_other:
                i = other_counters.get(name, 0) + 1
                other_counters[name] = i
                new_name = f"{name}_{i}"
            else:
                new_name = name
            result.other[new_name] = value
    return result


def _merge_bundles_concatenate_non_unique(bundles: list["Bundle"]) -> Bundle:
    """Merge bundles by concatenating same-named DataFrames and validating metadata equality."""
    result = Bundle()
    dfs_by_name: dict[str, list[pd.DataFrame]] = {}
    relation_by_name: dict[str, RelationDefinition] = {}

    for b in bundles:
        for name, df in b.dfs.items():
            dfs_by_name.setdefault(name, []).append(df)
        for relation in b.relations:
            existing = relation_by_name.get(relation.name)
            if existing is None:
                relation_by_name[relation.name] = relation
            else:
                assert existing == relation, (
                    f"Conflicting relation definitions for {relation.name!r}: "
                    f"{existing!r} != {relation!r}"
                )
        for name, value in b.other.items():
            if name in result.other:
                assert result.other[name] == value, (
                    f"Conflicting values in other for {name!r}: {result.other[name]!r} != {value!r}"
                )
            else:
                result.other[name] = value

    for name, frames in dfs_by_name.items():
        if len(frames) == 1:
            result.dfs[name] = frames[0]
        else:
            result.dfs[name] = pd.concat(frames, ignore_index=True)
    result.relations = list(relation_by_name.values())
    return result


def _find_overlaps(iterable: typing.Iterable[str]) -> set[str]:
    seen = set()
    overlaps = set()
    for x in iterable:
        if x in seen:
            overlaps.add(x)
        else:
            seen.add(x)
    return overlaps


@dataclasses.dataclass
class SingleTableView:
    """A JSON-serializable view of a table in the bundle, for use in the frontend.

    Attributes:
        columns: The columns to display.
        data: A list of rows, where each row is a list of values.
    """

    columns: list[str]
    data: list[list[typing.Any]]

    @staticmethod
    def from_df(df: pd.DataFrame, limit: int = 100):
        columns = [str(c) for c in df.columns][:limit]
        df = df[columns]
        data = df_for_frontend(df, limit).values.tolist()
        return SingleTableView(columns=columns, data=data)


@dataclasses.dataclass
class BundleTableView:
    """A JSON-serializable tabular view of a bundle, for use in the frontend."""

    dataframes: dict[str, SingleTableView]
    relations: list[RelationDefinition]
    other: dict[str, typing.Any]

    @staticmethod
    def from_bundle(bundle: Bundle, limit: int = 100):
        dataframes = {name: SingleTableView.from_df(df, limit) for name, df in bundle.dfs.items()}
        other = {k: str(v)[:limit] for k, v in bundle.other.items()}
        return BundleTableView(dataframes=dataframes, relations=bundle.relations, other=other)


def df_for_frontend(df: pd.DataFrame, limit: int) -> pd.DataFrame:
    """Returns a DataFrame with values that are safe to send to the frontend."""
    df = df[:limit]
    if isinstance(df, pl.LazyFrame):
        df = df.collect()
    if isinstance(df, pl.DataFrame):
        df = df.to_pandas()
    # Convert non-numeric columns to strings.
    for c in df.columns:
        if not pd.api.types.is_numeric_dtype(df[c]):
            df[c] = df[c].astype(str)
    return df
