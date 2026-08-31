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

@op("Import from database with SQL", color="green", icon="database", slow=True)
def import_from_database_with_SQL(
    *, database_type: DatabaseType = DatabaseType.postgresql, database_name:str="default", query="<enter query>"
) -> core.Bundle:
    """Import data from a database using an SQL query"""


    user = os.getenv("PGUSER", "postgres")
    host = os.getenv("PGHOST", "localhost")
    port = os.getenv("PGPORT", "5432")


    conn = ibis.postgres.connect(
        database=database_name,
        user=user,
        host=host,
        port=port
    )

    executable = conn.sql(query)
    df = executable.execute()
    return core.Bundle(df)


@op("Import all tables from a database", color="green", icon = "database", slow=True)
def import_all_tables_from_a_database(*, database_type: DatabaseType = DatabaseType.postgresql, database_name: str = "default") -> core.Bundle:
    """Import all tables from a database"""

    user = os.getenv("PGUSER", "postgres")
    host = os.getenv("PGHOST", "localhost")
    port = os.getenv("PGPORT", "5432")


    conn = ibis.postgres.connect(
        database=database_name,
        user=user,
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


@op("Explode table", icon = "bomb")
def explode_table(b: core.Bundle, *,column_divide_by: core.TableColumn):
    """Divide a table into multiple tables, by the unique values of a column from the table"""
    if b is None:
        return core.Bundle(dfs={})
    dfs = {}

    if not column_divide_by or not column_divide_by[0]:
        raise ValueError("No table selected")
    if len(column_divide_by) < 2 or not column_divide_by[1]:
        raise ValueError("No column selected")

    table = column_divide_by[0]
    column = column_divide_by[1]

    df = b.dfs[table]
    unique_values = df[column].unique()
    for value in unique_values:
        dfs[f"{table}_{column}_{value}"] = df[df[column] == value]
    return core.Bundle(dfs=dfs)


@op("Parsing JSON column", icon = "bomb") # what is an explosion? Change it to parsing
def parse_json_column(b: core.Bundle, *, select_column: core.TableColumn):
    """
    Parse a JSON column in a table and add its keys as new columns to the table.
    The original JSON column is removed after parsing.
    Returns a new bundle with the updated table.
    """

    if not select_column or not select_column[0]:
        raise ValueError("No table selected")
    if len(select_column) < 2 or not select_column[1]:
        raise ValueError("No column selected")

    table = select_column[0]
    column = select_column[1]

    if b is None:
        return core.Bundle(dfs={})

    df = b.dfs[table]
    parsed = df[column].apply(json.loads).tolist()
    json_df = pd.json_normalize(parsed)
    json_df.index = df.index
    dfs = {table: pd.concat([df.drop(columns=[column]), json_df], axis=1)}
    return core.Bundle(dfs=dfs)
