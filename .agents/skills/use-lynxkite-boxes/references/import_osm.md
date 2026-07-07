**Graph from OSM:**

```python
@op("Graph from OSM", slow=True)
def import_osm(*, location: str):
    import osmnx as ox

    return ox.graph.graph_from_place(location, network_type="drive")

```
