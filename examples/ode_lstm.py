from lynxkite_core.ops import op_registration, LongStr
from lynxkite_graph_analytics.core import Bundle
from matplotlib import pyplot as plt
import numpy as np
import pandas as pd
import json

op = op_registration("LynxKite Graph Analytics")


@op("Drop NA")
def drop_na(df: pd.DataFrame):
    return df.replace("", np.nan).dropna()


@op("Sort by")
def sort_by(df: pd.DataFrame, *, key_columns: str):
    df = df.copy()
    df.sort_values(
        by=[k.strip() for k in key_columns.split(",")],
        inplace=True,
        ignore_index=True,
    )
    return df


@op("Group by")
def group_by(df: pd.DataFrame, *, key_columns: str, aggregation: LongStr):
    key_columns: list[str] = [k.strip() for k in key_columns.split(",")]
    j = json.loads(aggregation)
    for k, vs in j.items():
        j[k] = [list if v == "list" else v for v in vs]
    res = df.groupby(key_columns).agg(j).reset_index()
    res.columns = ["_".join(col) for col in res.columns]
    return res


@op("Take first element of list")
def take_first_element(df: pd.DataFrame, *, column: str):
    df = df.copy()
    df[f"{column}_first_element"] = df[column].apply(lambda x: x[0])
    return df


@op("Plot time series", view="matplotlib")
def plot_time_series(bundle: Bundle, *, table_name: str, index: int, x_column: str, y_columns: str):
    df = bundle.dfs[table_name]
    y_columns: list[str] = [y.strip() for y in y_columns.split(",")]
    x = df[x_column].iloc[index]
    for y_column in y_columns:
        y = df[y_column].iloc[index]
        plt.plot(x, y, "o-", label=y_column)
    plt.xlabel(x_column)
    plt.legend()
