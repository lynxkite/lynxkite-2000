import pandas as pd
import pytest

from lynxkite_graph_analytics.bundle import (
    Bundle,
    BundleMergeMode,
    RelationDefinition,
    merge_bundles,
)


def _relation(
    name: str,
    df: str = "edges",
    source_table: str = "nodes",
    target_table: str = "nodes",
    source_key: str = "id",
    target_key: str = "id",
):
    return RelationDefinition(
        df=df,
        source_column="source",
        target_column="target",
        source_table=source_table,
        target_table=target_table,
        source_key=source_key,
        target_key=target_key,
        name=name,
    )


def test_merge_bundles_must_be_unique_merges_all_parts():
    left = Bundle(
        dfs={"left": pd.DataFrame({"x": [1]})},
        relations=[_relation("left_rel", df="left", source_table="left", target_table="left")],
        other={"left_model": "a"},
    )
    right = Bundle(
        dfs={"right": pd.DataFrame({"y": [2]})},
        relations=[_relation("right_rel", df="right", source_table="right", target_table="right")],
        other={"right_model": "b"},
    )

    merged = merge_bundles([left, right], merge_mode=BundleMergeMode.must_be_unique)

    assert set(merged.dfs.keys()) == {"left", "right"}
    assert [r.name for r in merged.relations] == ["left_rel", "right_rel"]
    assert merged.other == {"left_model": "a", "right_model": "b"}


def test_merge_bundles_must_be_unique_raises_on_collisions():
    left = Bundle(dfs={"same": pd.DataFrame({"x": [1]})})
    right = Bundle(dfs={"same": pd.DataFrame({"x": [2]})})

    with pytest.raises(AssertionError, match="Non-unique tables"):
        merge_bundles([left, right], merge_mode=BundleMergeMode.must_be_unique)


def test_merge_bundles_rename_non_unique_renames_and_rewrites_relations():
    left = Bundle(
        dfs={
            "nodes": pd.DataFrame({"id": [1]}),
            "edges": pd.DataFrame({"source": [1], "target": [1]}),
        },
        relations=[_relation("rel", df="edges", source_table="nodes", target_table="nodes")],
        other={"model": "left"},
    )
    right = Bundle(
        dfs={
            "nodes": pd.DataFrame({"id": [2]}),
            "edges": pd.DataFrame({"source": [2], "target": [2]}),
        },
        relations=[_relation("rel", df="edges", source_table="nodes", target_table="nodes")],
        other={"model": "right"},
    )

    merged = merge_bundles([left, right], merge_mode=BundleMergeMode.rename_non_unique)

    assert set(merged.dfs.keys()) == {"nodes_1", "edges_1", "nodes_2", "edges_2"}
    assert set(merged.other.keys()) == {"model_1", "model_2"}
    assert merged.other["model_1"] == "left"
    assert merged.other["model_2"] == "right"

    relations_by_name = {r.name: r for r in merged.relations}
    assert set(relations_by_name.keys()) == {"rel_1", "rel_2"}
    assert relations_by_name["rel_1"].df == "edges_1"
    assert relations_by_name["rel_1"].source_table == "nodes_1"
    assert relations_by_name["rel_1"].target_table == "nodes_1"
    assert relations_by_name["rel_2"].df == "edges_2"
    assert relations_by_name["rel_2"].source_table == "nodes_2"
    assert relations_by_name["rel_2"].target_table == "nodes_2"


def test_merge_bundles_concatenate_non_unique_concatenates_dataframes_and_keeps_equal_metadata():
    relation = _relation("edges_rel", df="edges", source_table="nodes", target_table="nodes")
    left = Bundle(
        dfs={
            "nodes": pd.DataFrame({"id": [1]}),
            "edges": pd.DataFrame({"source": [1], "target": [1]}),
        },
        relations=[relation],
        other={"config": "same"},
    )
    right = Bundle(
        dfs={
            "nodes": pd.DataFrame({"id": [2]}),
            "edges": pd.DataFrame({"source": [2], "target": [2]}),
        },
        relations=[relation],
        other={"config": "same"},
    )

    merged = merge_bundles([left, right], merge_mode=BundleMergeMode.concatenate_non_unique)

    pd.testing.assert_frame_equal(
        merged.dfs["nodes"].reset_index(drop=True),
        pd.DataFrame({"id": [1, 2]}),
    )
    pd.testing.assert_frame_equal(
        merged.dfs["edges"].reset_index(drop=True),
        pd.DataFrame({"source": [1, 2], "target": [1, 2]}),
    )
    assert merged.relations == [relation]
    assert merged.other == {"config": "same"}


def test_merge_bundles_concatenate_non_unique_raises_on_conflicting_relations():
    left = Bundle(
        dfs={"edges": pd.DataFrame({"source": [1], "target": [1]})},
        relations=[_relation("rel", source_key="id")],
    )
    conflicting = RelationDefinition(
        df="edges",
        source_column="source",
        target_column="target",
        source_table="nodes",
        target_table="nodes",
        source_key="node_id",
        target_key="id",
        name="rel",
    )
    right = Bundle(
        dfs={"edges": pd.DataFrame({"source": [2], "target": [2]})},
        relations=[conflicting],
    )

    with pytest.raises(AssertionError, match="Conflicting relation definitions"):
        merge_bundles([left, right], merge_mode=BundleMergeMode.concatenate_non_unique)


def test_merge_bundles_concatenate_non_unique_raises_on_conflicting_other_values():
    left = Bundle(dfs={"a": pd.DataFrame({"x": [1]})}, other={"model": "v1"})
    right = Bundle(dfs={"a": pd.DataFrame({"x": [2]})}, other={"model": "v2"})

    with pytest.raises(AssertionError, match="Conflicting values in other"):
        merge_bundles([left, right], merge_mode=BundleMergeMode.concatenate_non_unique)
