import os

from lynxkite_core.ops import op
from lynxkite_graph_analytics import Bundle
from lynxkite_graph_analytics.core import TableColumn
from lynxkite_graph_analytics.operations.db_ops import DatabaseType
from matplotlib import table
import pandas as pd
import ibis
import json

@op("LynxKite Graph Analytics", "Funny data")
def funny(*, n="Apple"):
    """Give in a number, and it gives back a fruit"""
    if n.isnumeric():
        n = "Banana"
    df = pd.DataFrame(
        {
            "number": n
        }

    , index=[0])
    return df

@op("LynxKite Graph Analytics", "Funny Bundle data")
def funny_bundle(*, n="Apple"):
    """Give in a number, and it gives back a fruit 5 times"""
    if n.isnumeric():
        n = "Banana"
    df = pd.DataFrame(
        {
            "number": [n, n, n,n, n, "funny"]
        }
    , index=range(6))
    return Bundle(dfs={"funny_data": df})



@op("LynxKite Graph Analytics", "Import from Database with credentials", icon = "database")
def import_from_database_with_cred(*, database="<enter database name>", user="<enter username>", password="<enter password>", host="<enter host>", port="<enter port>", query="<enter query>"):
    """Import data from PostgreSQL using a SQL query"""

    #TODO: save credentials to env variables, and read from there, so that the user does not have to enter them every time
    # maybe can add to just pick the table from the database, sql inj.
    database = "L"
    user = "postgres"
    password = "database"
    # host = "127.0.0.1"
    host = "localhost"
    port = 5432
    conn = ibis.postgres.connect(
        database=database,
        user=user,
        password=password,
        host=host,
        port=port
    )

    executable = conn.sql(query)
    df = executable.execute()
    return Bundle(dfs={"postgresql_data": df})

@op("LynxKite Graph Analytics", "Import from Database DEMO", icon = "database")
def import_from_database_demo(*, query="select * from hetionet_nodes;"):
    """Import data from PostgreSQL using a SQL query"""


    # maybe can add to just pick the table from the database, sql inj.
    database = "L"
    user = "postgres"
    password = "database"
    # host = "127.0.0.1"
    host = "localhost"
    port = 5432
    conn = ibis.postgres.connect(
        database=database,
        user=user,
        password=password,
        host=host,
        port=port
    )

    executable = conn.sql(query)
    df = executable.execute()
    return Bundle(dfs={"postgresql_data": df})


@op("LynxKite Graph Analytics", "Get Database Tables ex", icon = "database")
def get_database_tables_ex(*, database_type: DatabaseType = DatabaseType.postgresql, database_name: str = "default"):
    """Get the list of tables from a PostgreSQL database"""

    user = os.getenv("PGUSER", "postgres")
    host = os.getenv("PGHOST", "localhost")
    port = os.getenv("PGPORT", "5432")

    # maybe can add to just pick the table from the database, sql inj.
    database = database_name
    get_tablesql = "SELECT table_name FROM information_schema.tables WHERE table_schema='public';"
    conn = ibis.postgres.connect(
        database=database,
        user=user,
        host=host,
        port=port
    )
    tablenames = conn.sql(get_tablesql).execute()

    return Bundle(dfs={"postgresql_data": pd.DataFrame(tablenames, columns=["table_name"])})

@op("LynxKite Graph Analytics", "Get datatables ex", icon = "database")
def get_datatables_ex(*, database="<enter database name>", user="<enter username>", password="<enter password>", host="<enter host>", port="<enter port>"):
    """Import data from PostgreSQL using a SQL query"""


    # maybe can add to just pick the table from the database, sql inj.
    database = "L"
    user = "postgres"
    password = "database"
    # host = "127.0.0.1"
    host = "localhost"
    port = 5432
    get_tablesql = "SELECT table_name FROM information_schema.tables WHERE table_schema='public';"
    conn = ibis.postgres.connect(
        database=database,
        user=user,
        password=password,
        host=host,
        port=port
    )
    tablenames = conn.sql(get_tablesql).execute()

    return Bundle(dfs={"postgresql_data": pd.DataFrame(tablenames, columns=["table_name"])})

@op("LynxKite Graph Analytics", "Crumble Database Table ex", icon = "database")
def crumble_table_ex(b: Bundle, *,column_divide_by = "<column name>"):
    """Divide a database table into multiple tables, by the values of a column"""
    if b is None:
        return Bundle(dfs={})
    # TODO do ti for other tables in the Bundle
    df = b.dfs["postgresql_data"]
    if column_divide_by not in df.columns:
        raise ValueError(f"Column {column_divide_by} not found in dataframe")
    unique_values = df[column_divide_by].unique()
    dfs = {}
    for value in unique_values:
        dfs[f"{column_divide_by}_{value}"] = df[df[column_divide_by] == value]
    return Bundle(dfs=dfs)

@op("LynxKite Graph Analytics", "Explode Column ex", icon = "database")
def explode_column_ex(b: Bundle, *, added_columns_from="<column name>"):
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
        return Bundle(dfs={})
    dfs = {}

    def _parse(x):
        if isinstance(x, dict):
            return x
        if pd.isna(x):
            return {}
        return json.loads(x)

    for k in b.dfs:
        df = b.dfs[k]
        if added_columns_from not in df.columns:
            # skip this table if the column is not found
            dfs[k] = df
            continue
        json_df = pd.json_normalize(df[added_columns_from].apply(_parse).tolist())
        df = df.drop(columns=[added_columns_from]).reset_index(drop=True)
        dfs[k] = pd.concat([df, json_df], axis=1)

    return Bundle(dfs=dfs)

@op("LynxKite Graph Analytics", "Fill in Credentials for DEMO ex", icon = "file-filled")
def fill_in_credentials():
    database = "L"
    user = "postgres"
    password = "database"
    host = "localhost"
    port = 5432

    """Fill in the credentials for the demo database"""
    return Bundle(dfs={"credentials": pd.DataFrame({"database": [database], "user": [user], "password": [password], "host": [host], "port": [port]})})


@op("LynxKite Graph Analytics", "print env variables", icon = "database")
def print_env_variables():
    """Print the environment variables"""
    env_df = pd.DataFrame(list(os.environ.items()), columns=["name", "value"])
    return Bundle(dfs={"env_variables": env_df})


@op("LynxKite Graph Analytics", "get all database tables", icon = "database")
def import_all_tables_from_database_boxpy(*, database_type: DatabaseType = DatabaseType.postgresql) -> Bundle:
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
    try:
        port = int(port_raw)
    except ValueError as e:
        raise ValueError(
            f"Invalid PostgreSQL port '{port_raw}'. Set PGPORT or POSTGRES_PORT to an integer."
        ) from e

    connection_kwargs = {
        "database": database,
        "user": user,
        "host": host,
        "port": port,
    }
    if password:
        connection_kwargs["password"] = password
    conn = ibis.postgres.connect(**connection_kwargs)
    query = "SELECT table_name FROM information_schema.tables WHERE table_schema='public';"
    executable = conn.sql(query)
    tables = executable.execute()
    tables_df = pd.DataFrame(tables, columns=["table_name"])
    dfs = {}

    for table in tables_df["table_name"]:
        table_query = f"SELECT * FROM {table};"
        table_executable = conn.sql(table_query)
        dfs[table] = pd.DataFrame(table_executable.execute())

    return Bundle(dfs=dfs)

@op("LynxKite Graph Analytics", "parse JSON column try", icon = "bomb") # what is an explosion? Change it to parsing
def parse_json_column_try(b: Bundle, *, select_column: TableColumn):
    """Add columns from a table's JSON column to the main table, by the values of the column"""


    if not select_column or not select_column[0]:
        raise ValueError("0: no table selected")
    if len(select_column) < 2 or not select_column[1]:
        raise ValueError("1: no column selected")


    table = select_column[0]
    column = select_column[1]

    if b is None:
        return Bundle(dfs={})

    def _parse(value):
        return json.loads(value)
        if value is None or (isinstance(value, float) and pd.isna(value)):
            return {}
        if isinstance(value, str):
            value = json.loads(value)
        if isinstance(value, dict):
            return value
        if isinstance(value, list):
            parsed = {}
            for item in value:
                if "key" in item:
                    key = item["key"]
                elif "property" in item:
                    key = item["property"]
                else:
                    key = list(item.values())[1]
                parsed[key] = item["value"] if "value" in item else list(item.values())[0]
            return parsed
            # return {
            #     item.get("key", item.get("property")): item.get("value")
            #     for item in value
            # }
        raise ValueError(
            f"Expected JSON object or list in column {column!r}, got {type(value).__name__}"
        )

    df = b.dfs[table]
    # parsed = json.loads(df[column]).tolist()
    parsed = df[column].apply(json.loads)#.tolist()
    json_df = pd.json_normalize(parsed)
    json_df.index = df.index
    dfs = {table: pd.concat([df.drop(columns=[column]), json_df], axis=1)}
    return Bundle(dfs=dfs)


@op("LynxKite Graph Analytics", "get data with sql", icon="database")
def import_from_database_with_SQL_ex_py(
    *, database_type: DatabaseType = DatabaseType.postgresql, database_name:str="default", query="<enter query>"
) ->Bundle:
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
    return df

@op("LynxKite Graph Analytics", "get demo datas", icon = "database")
def import_demo_from_all_tables_from_a_database_ex_py(*, database_type: DatabaseType = DatabaseType.postgresql, database_name: str = "default"
    ) -> Bundle:
    """Import demo from all tables from a database"""

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
        table_query = f"""SELECT * FROM "{table}" limit 50;"""
        table_executable = conn.sql(table_query)
        dfs[table] = pd.DataFrame(table_executable.execute())

    return Bundle(dfs=dfs)

# TODO: the import all tables is working, have to save it in dbops, then push it to github
