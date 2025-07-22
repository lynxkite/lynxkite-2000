from lynxkite_core.ops import op
import pandas as pd
import base64


@op("LynxKite Graph Analytics", "Example image table")
def make_image_table():
    svg = '<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 64 64" enable-background="new 0 0 64 64"><path d="M56 2 18.8 42.909 8 34.729 2 34.729 18.8 62 62 2z"/></svg>'
    data = "data:image/svg+xml;base64," + base64.b64encode(svg.encode("utf-8")).decode("utf-8")
    http = "https://upload.wikimedia.org/wikipedia/commons/2/2e/Emojione_BW_2714.svg"
    return pd.DataFrame({"names": ["svg", "data", "http"], "images": [svg, data, http]})
