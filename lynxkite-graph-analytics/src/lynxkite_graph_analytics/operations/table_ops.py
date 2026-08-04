"""Operations for tables."""

import enum
import polars as pl

from lynxkite_core import ops
import pandas as pd
import io
from .. import core

op = ops.op_registration(core.ENV, "Table operations")


@op("Sample table", icon="filter-filled")
def sample_table(b: core.Bundle, *, table_name: core.TableName = "meta", fraction: float = 0.1):
    b = b.copy()
    b.dfs[table_name] = b.dfs[table_name].sample(frac=fraction)
    return b


@op("Filter with formula", icon="filter-filled")
def filter_with_formula(
    b: core.Bundle, *, table_name: core.TableName, formula: ops.LongStr
) -> core.Bundle:
    """Removes all rows where the formula(https://numexpr.readthedocs.io/en/latest/user_guide.html#supported-functions) evaluates to false"""
    b = b.copy()
    df = b.dfs[table_name]
    b.dfs[table_name] = df.query(formula)
    return b


class OrderType(enum.StrEnum):
    asc = "ascending"
    desc = "descending"


@op("Vector from attributes", icon="brackets-contain")
def vector_from_attributes(
    b: core.Bundle,
    *,
    table_name: core.TableName,
    attributes: core.MultiColumnNameByTableName,
    vector_name: str,
) -> core.Bundle:
    """Creates a new column with vectors that contain the selected attributes in the selected order"""
    b = b.copy()
    df = b.dfs[table_name].copy()
    df[vector_name] = list(zip(*(df[col] for col in attributes)))
    b.dfs[table_name] = df
    return b


@op("Add rank attribute", icon="sort-descending")
def add_rank(
    b: core.Bundle,
    *,
    table_column: core.TableColumn,
    rank_name: str,
    order: OrderType,
):
    """Sorts the rows by the given attribute in the given order and creates a new column with the rank of the row

    Parameters
    ----------
    table_column : core.TableColumn
        The table and column to rank
    rank_name : str
        The name of the new rank column
    order : OrderType
        The order in which to rank the rows either 'ascending' or 'descending'

    Returns
    -------
    output : core.Bundle
        The updated bundle with the new rank column
    """
    table, column = table_column
    b = b.copy()
    df = b.dfs[table]

    df = df.sort_values(by=column, ascending=(order == OrderType.asc))
    df[rank_name] = range(len(df))

    b.dfs[table] = df
    return b


@op("Drop tables", color="orange", icon="table-minus")
def drop_tables(b: core.Bundle, *, keep_selected: bool, tables: core.MultiTableName) -> core.Bundle:
    """
    Keeps/removes the selected tables based on the value of drop_selected
    :param b: the bundle
    :param keep_selected: if False, removes the selected tables, otherwise the unselected ones
    :param tables: the tables to keep/remove
    """
    b = b.copy()

    b.dfs = {k: v for k, v in b.dfs.items() if (k in tables) == keep_selected}
    b.relations = [r for r in b.relations if r.source_table in b.dfs and r.target_table in b.dfs]
    return b


@op("Fill attributes with default values", icon="table-column")
def fill_with_default(
    b: core.Bundle, *, table_name: core.TableName, adder: core.DropdownTextAdderByTableName
) -> core.Bundle:
    """
    An attribute may not be defined everywhere. This operation sets the provided values for the rows of the specified attributes where they are not defined.
    :param b: the bundle
    :param table_name: the table to operate on
    :param adder: the attributes and the values to set
    """
    b = b.copy()
    df = b.dfs[table_name].copy()
    for column, default_value in adder:
        df[column] = df[column].fillna(default_value)
    b.dfs[table_name] = df
    return b


@op("Rename table", color="orange", icon="writing")
def rename_table(b: core.Bundle, *, old_name: core.TableName, new_name: str) -> core.Bundle:
    """Assigns a new name to the table"""
    b = b.copy()
    b.dfs[new_name] = b.dfs.pop(old_name)
    relations = []
    for r in b.relations:
        r = r.copy()
        if r.source_table == old_name:
            r.source_table = new_name
        if r.target_table == old_name:
            r.target_table = new_name
        relations.append(r)
    b.relations = relations
    return b


@op("Derive property", icon="arrow-big-right-lines")
def derive_property(
    b: core.Bundle, *, table_name: core.TableName, formula: ops.LongStr
) -> core.Bundle:
    b = b.copy()
    df = b.dfs[table_name]
    b.dfs[table_name] = df.eval(formula)
    return b


@op("Derive with SQL", icon="database-plus")
def derive_with_sql(
    b: core.Bundle,
    *,
    table_name: core.TableName,
    formula: ops.LongStr,
    name: str,
) -> core.Bundle:
    """
    Derives a new column with a SQL expression and stores it in the same table.
    :param b: the bundle.
    :param table_name: the name of the table to derive the column in.
    :param formula: the formula to derive the column with.
    :param name: the name of the derived column.
    """
    b = b.copy()
    query = f"select *, {formula} as {name} from {table_name}"
    b.dfs[table_name] = pl.SQLContext(b.dfs).execute(query).collect().to_pandas()
    return b


@op("Enter table data", color="green", icon="table-filled")
def enter_table_data(
    *,
    table_name: str,
    data: ops.LongStr,
):
    """Enter table data as CSV. The first row should contain column names."""
    b = core.Bundle()
    b.dfs[table_name] = pd.read_csv(io.StringIO(data))
    return b


class TableMergeMode(enum.StrEnum):
    merge_second = "Merge, prefer the first table's version"
    merge_first = "Merge, prefer the second table's version"
    report_conflict = "Merge, report error on conflict"
    use_first = "Keep the first table's version"
    use_second = "Use the second table's version"
    only_matching = "Only keep rows with matching values"
    both = "Keep both values with suffixes"
    disallow = "Disallow this"


@op("Join tables", icon="table-plus")
def join_tables(
    b: core.Bundle,
    *,
    table1_column: core.TableColumn,
    table2_column: core.TableColumn,
    merge_mode: TableMergeMode,
) -> core.Bundle:
    """
    Adds data from the second table to the first table.
    :param b: the bundle
    :param table1_column: the first table and its column to join on
    :param table2_column: the second table and its column to join on
    :param merge_mode: determines what happens if a column is in both tables
    Merge, prefer the second table’s version: Where the second table defines values, those will be used. Elsewhere, the first table's values are used.
    Merge, prefer the first table’s version: Where the first table defines values, those will be used. Elsewhere, the second table's values are used.
    Merge, report error on conflict: An assertion is made to ensure that the values in the two tables are equal. If they are not, an error is raised.
    Use the first table’s version: The data in the second table is ignored.
    Use the second table’s version: The data in the first table is ignored.
    Only keep rows with matching values: Only rows that have matching values in both tables are kept.
    Keep both values with suffixes: Both values are kept, with suffixes added to the column names to distinguish them.
    Disallow this: A name conflict is treated as an error.
    """
    b = b.copy()
    primary_table = b.dfs[table1_column[0]].copy()
    secondary_table = b.dfs[table2_column[0]].copy()

    how = "inner" if merge_mode == TableMergeMode.only_matching else "left"

    merged = primary_table.merge(
        secondary_table,
        how=how,
        left_on=table1_column[1],
        right_on=table2_column[1],
        suffixes=("_1", "_2"),
    )

    for column in secondary_table.columns:
        if (
            column == table2_column[1]
            or f"{column}_1" not in merged.columns
            or merge_mode == TableMergeMode.both
        ):
            continue

        if merge_mode == TableMergeMode.merge_second:
            merged[column] = merged[f"{column}_1"].combine_first(merged[f"{column}_2"])
        elif merge_mode == TableMergeMode.merge_first:
            merged[column] = merged[f"{column}_2"].combine_first(merged[f"{column}_1"])
        elif merge_mode == TableMergeMode.report_conflict:
            conflict = merged[f"{column}_1"] != merged[f"{column}_2"]
            if conflict.any():
                raise ValueError(f"Conflict in column {column}: {merged[conflict]}")
            merged[column] = merged[f"{column}_1"].combine_first(merged[f"{column}_2"])
        elif merge_mode == TableMergeMode.use_second:
            merged[column] = merged[f"{column}_2"]
        elif merge_mode == TableMergeMode.use_first:
            merged[column] = merged[f"{column}_1"]
        elif merge_mode == TableMergeMode.only_matching:
            merged = merged[merged[f"{column}_1"] == merged[f"{column}_2"]]
            merged[column] = merged[f"{column}_1"]
        elif merge_mode == TableMergeMode.disallow:
            raise ValueError(f"Both tables have '{column}' column.")

        merged.drop(columns=[f"{column}_1", f"{column}_2"], inplace=True)

    b.dfs[table1_column[0]] = merged
    return b
