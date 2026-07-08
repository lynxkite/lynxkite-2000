## Here you can see some of the defined types in LynxKite

### Bundle
Here is a skeleton of type python code of `Bundle`:
```
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
    def from_nx(cls, graph: nx.Graph) -> Bundle:

    @classmethod
    def from_df(cls, df: pd.DataFrame) -> Bundle:

    def to_nx(self) -> networkx.Graph:

    def copy(self):
        """
        Returns a shallow copy of the bundle. The Bundle and its containers are new, but
        the DataFrames and RelationDefinitions are shared. (The contents of `other` are also shared.)
        """

    def to_table_view(self, limit: int = 100) -> BundleTableView:
        """Converts the bundle to a format suitable for display as tables in the frontend."""


### BundleTableView
```python
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
```

## Parameter types
Here you can find some types you can use as parameters in your custom boxes.

Annotated types with format "dropdown" specify the available options
as a query on the input_metadata. These query expressions are JMESPath expressions.

Parameter names in angle brackets, like <table_name>, will be replaced with the parameter values. (This is not part of JMESPath.)
eg. ColumnNameByTableName will list the columns of the DataFrame with the name
specified by the `table_name` parameter.

### TableName
```
TableName = typing.Annotated[
    str, {"format": "dropdown", "metadata_query": "[].dataframes[].keys(@)[]"}
]
```

A type annotation to be used for parameters of an operation. TableName is
rendered as a dropdown in the frontend, listing all DataFrames in the Bundle.
The table name is passed to the operation as a string.

### NodePropertyName
```
NodePropertyName = typing.Annotated[
    str, {"format": "dropdown", "metadata_query": "[].dataframes[].nodes[].columns[]"}
]
```

A type annotation to be used for parameters of an operation. NodePropertyName is rendered as a dropdown in the frontend, listing the columns of the "nodes" DataFrame. The column name is passed to the operation as a string.

### EdgePropertyName
```
EdgePropertyName = typing.Annotated[
    str, {"format": "dropdown", "metadata_query": "[].dataframes[].edges[].columns[]"}
]
```

A type annotation to be used for parameters of an operation. EdgePropertyName is rendered as a dropdown in the frontend, listing the columns of the "edges" DataFrame. The column name is passed to the operation as a string.

### OtherName
```
OtherName = typing.Annotated[str, {"format": "dropdown", "metadata_query": "[].other.keys(@)[]"}]
```

A type annotation to be used for parameters of an operation. OtherName is
rendered as a dropdown in the frontend, listing the keys on the "other" part of the Bundle.
The key is passed to the operation as a string.

### ColumnNameByTableName
```
ColumnNameByTableName = typing.Annotated[
    str, {"format": "dropdown", "metadata_query": "[].dataframes[].<table_name>.columns[]"}
]
```

A type annotation to be used for parameters of an operation. ColumnNameByTableName is
rendered as a dropdown in the frontend, listing the columns of the DataFrame
named by the "table_name" parameter. The column name is passed to the operation as a string.

### TableColumn
```
TableColumn = typing.Annotated[
    tuple[str, str],
    {
        "format": "double-dropdown",
        "metadata_query1": "[].dataframes[].keys(@)[]",
        "metadata_query2": "[].dataframes[].<first>.columns[]",
    },
]
```

A type annotation to be used for parameters of an operation. TableColumn is
rendered as a pair of dropdowns for selecting a table in the Bundle and a column inside of
that table. Effectively "TableName" and "ColumnNameByTableName" combined.
The selected table and column name is passed to the operation as a 2-tuple of strings.

### RecordsColumn
```
RecordsColumn = typing.Annotated[
    str, {"format": "dropdown", "metadata_query": "[].dataframes[].records.columns[]"}
]
```

A type annotation to be used for parameters of an operation. RecordsColumn is
rendered as a dropdown in the frontend, listing the columns of the "records" DataFrame.
The column name is passed to the operation as a string.
