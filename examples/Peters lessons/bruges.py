"""Custom operations for the In Bruges demo workspace."""

import numpy as np
from lynxkite_core.ops import op_registration
from lynxkite_graph_analytics import core


op = op_registration(core.ENV, "In Bruges")


@op("Derive segment_length")
def derive_segment_length(b: core.Bundle, *, table_name: core.TableName) -> core.Bundle:
    b = b.copy()
    df = b.dfs[table_name].copy()

    r = 6371000  # radius of Earth in meters

    lat1 = np.radians(df["lat_src"])
    lat2 = np.radians(df["lat_dst"])
    lon1 = np.radians(df["lon_src"])
    lon2 = np.radians(df["lon_dst"])

    d_lat = lat2 - lat1
    d_lon = lon2 - lon1
    a = np.sin(d_lat / 2) ** 2 + np.cos(lat1) * np.cos(lat2) * (np.sin(d_lon / 2) ** 2)
    c = 2 * np.arctan2(np.sqrt(a), np.sqrt(1 - a))

    df["segment_length"] = r * c
    b.dfs[table_name] = df
    return b
