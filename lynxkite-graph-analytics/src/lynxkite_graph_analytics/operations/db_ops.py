"""Operations for reading from databases."""

from lynxkite_core import ops
from .. import core

import pandas as pd
import ibis
import json
import enum
import os

op = ops.op_registration(core.ENV, "Database operations")

class DatabaseType(enum.StrEnum):
    postgresql = "postgresql"
# TODO: add support for other databases in the future
    # mysql = "mysql"
    # sqlite = "sqlite"
    # oracle = "oracle"
    # mssql = "mssql"

@op("Import from Database with SQL", color="green", icon="database", slow=True)
def import_from_database_with_SQL(
    *, database_type: DatabaseType = DatabaseType.postgresql, query="<enter query>"
) -> core.Bundle:
    """Import data from a database using an SQL query"""


    # database = "L"
    # user = "postgres"
    # password = "database"
    # host = "localhost"
    # port = 5432

    database = os.getenv("PGDATABASE") or os.getenv("POSTGRES_DB")
    user = os.getenv("PGUSER") or os.getenv("POSTGRES_USER", "postgres")
    password = os.getenv("PGPASSWORD") or os.getenv("POSTGRES_PASSWORD")
    host = os.getenv("PGHOST") or os.getenv("POSTGRES_HOST", "localhost")
    port_raw = os.getenv("PGPORT") or os.getenv("POSTGRES_PORT", "5432")

    if not database:
        raise ValueError(
            "PostgreSQL database name is missing. Set PGDATABASE or POSTGRES_DB."
        )
    if password is None:
        raise ValueError(
            "PostgreSQL password is missing. Set PGPASSWORD or POSTGRES_PASSWORD."
        )

    try:
        port = int(port_raw)
    except ValueError as e:
        raise ValueError(
            f"Invalid PostgreSQL port '{port_raw}'. Set PGPORT or POSTGRES_PORT to an integer."
        ) from e

    conn = ibis.postgres.connect(
        database=database,
        user=user,
        password=password,
        host=host,
        port=port
    )

    executable = conn.sql(query)
    df = executable.execute()
    return core.Bundle(dfs={"database_data": df})

@op("Import all tables from a Database", color="green", icon="database", slow=True)
def import_all_tables_from_database(
    *, database_type: DatabaseType = DatabaseType.postgresql
) -> core.Bundle:
    """Import all tables from a database"""


    database = os.getenv("PGDATABASE") or os.getenv("POSTGRES_DB")
    user = os.getenv("PGUSER") or os.getenv("POSTGRES_USER", "postgres")
    password = os.getenv("PGPASSWORD") or os.getenv("POSTGRES_PASSWORD")
    host = os.getenv("PGHOST") or os.getenv("POSTGRES_HOST", "localhost")
    port_raw = os.getenv("PGPORT") or os.getenv("POSTGRES_PORT", "5432")

    if not database:
        raise ValueError(
            "PostgreSQL database name is missing. Set PGDATABASE or POSTGRES_DB."
        )
    if password is None:
        raise ValueError(
            "PostgreSQL password is missing. Set PGPASSWORD or POSTGRES_PASSWORD."
        )

    try:
        port = int(port_raw)
    except ValueError as e:
        raise ValueError(
            f"Invalid PostgreSQL port '{port_raw}'. Set PGPORT or POSTGRES_PORT to an integer."
        ) from e

    conn = ibis.postgres.connect(
        database=database,
        user=user,
        password=password,
        host=host,
        port=port
    )
    query = "SELECT table_name FROM information_schema.tables WHERE table_schema='public';"
    executable = conn.sql(query)
    tables = executable.execute()
    tables_df = pd.DataFrame(tables, columns=["table_name"])
    dfs = {}
    
    for table in tables_df["table_name"]:
        table_query = f"SELECT * FROM {table};"
        table_executable = conn.sql(table_query)
        dfs[table] = pd.DataFrame(table_executable.execute())

    return core.Bundle(dfs=dfs)


@op("Explode Database Table", icon = "bomb")
def explode_table(b: core.Bundle, *,column_divide_by = "<column name>"):
    """Divide a database table into multiple tables, by the values of a column"""
    if b is None:
        return core.Bundle(dfs={})
    dfs = {}
    for k in b.dfs:
        df = b.dfs[k]
        if column_divide_by not in df.columns:
            dfs[k] = df
            continue
        unique_values = df[column_divide_by].unique()
        for value in unique_values:
            dfs[f"{k}_{column_divide_by}_{value}"] = df[df[column_divide_by] == value]
    return core.Bundle(dfs=dfs)


@op("Explode Column", icon = "bomb")
def explode_column(b: core.Bundle, *, added_columns_from="<column name>"):
    """Add columns from a tables columns to the main table, by the values of the column"""

    """
    The values of the column is in json format, and the columns are added to the main
    table, with the keys of the json as the column names, and the values of the json as the values of the columns.
    Therefore deletes the original column after adding the new columns.
    Applies for all tables in the bundle, and returns a new bundle with the exploded tables.

    If the table does not have the column, it skips it and returns the table as is.
    If the column is not in json format, it raises an error.
    """

    if b is None:
        return core.Bundle(dfs={})
    dfs = {}

    def _parse(x):
        if isinstance(x, dict):
            return x
        # Avoid pd.isna on lists/arrays — check for None/float NaN only
        if x is None or (isinstance(x, float) and pd.isna(x)):
            return {}
        return json.loads(x)

    for k in b.dfs:
        df = b.dfs[k]
        if added_columns_from not in df.columns:
            # skip this table if the column is not found
            dfs[k] = df
            continue
        parsed = df[added_columns_from].apply(_parse).tolist()
        # Build one list per key; avoids broadcast assignment that fails on list values
        all_keys = dict.fromkeys(key for d in parsed for key in d)
        json_df = pd.DataFrame(
            {key: pd.Series([d.get(key) for d in parsed], index=df.index, dtype=object) for key in all_keys},
        )
        result_df = pd.concat([df.drop(columns=[added_columns_from]), json_df], axis=1)
        # Ensure we're storing a DataFrame, not a list
        assert isinstance(result_df, pd.DataFrame), f"Expected DataFrame, got {type(result_df).__name__}"
        dfs[k] = result_df
    return core.Bundle(dfs=dfs)

