import dataclasses


@dataclasses.dataclass
class RelationDefinition:
    """
    Defines a set of edges.

    Attributes:
        df: The name of the DataFrame that contains the edges.
        source_column: The column in the edge DataFrame that contains the source node ID.
        target_column: The column in the edge DataFrame that contains the target node ID.
        source_table: The name of the DataFrame that contains the source nodes.
        target_table: The name of the DataFrame that contains the target nodes.
        source_key: The column in the source table that contains the node ID.
        target_key: The column in the target table that contains the node ID.
        name: Descriptive name for the relation.
    """

    df: str
    source_column: str
    target_column: str
    source_table: str
    target_table: str
    source_key: str
    target_key: str
    name: str
