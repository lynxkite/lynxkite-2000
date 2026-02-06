from lynxkite_core.ops import op
from lynxkite_graph_analytics.core import Bundle, TableColumn
import pandas as pd
import base64


@op("LynxKite Graph Analytics", "Example image table", color="green", icon="photo")
def make_image_table():
    svg = '<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 64 64" enable-background="new 0 0 64 64"><path d="M56 2 18.8 42.909 8 34.729 2 34.729 18.8 62 62 2z"/></svg>'
    data = "data:image/svg+xml;base64," + base64.b64encode(svg.encode("utf-8")).decode("utf-8")
    http = "https://upload.wikimedia.org/wikipedia/commons/2/2e/Emojione_BW_2714.svg"
    return pd.DataFrame({"names": ["svg", "data", "http"], "images": [svg, data, http]})


@op("LynxKite Graph Analytics", "Fetch molecule images", icon="microscope-filled")
def fetch_molecule_images(b: Bundle, *, table_column: TableColumn, save_as: str = "image"):
    """Adds molecule images in a table."""
    b = b.copy()
    table_name, smiles_column = table_column
    df = b.dfs[table_name]
    df = df.copy()
    df[save_as] = df[smiles_column].apply(
        lambda x: f"https://cactus.nci.nih.gov/chemical/structure/{x}/image"
    )
    b.dfs[table_name] = df
    return b
