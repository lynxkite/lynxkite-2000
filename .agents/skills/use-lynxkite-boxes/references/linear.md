**Linear:**

```python
@op("Linear", weights=True)
def linear(x, *, output_dim=1024):
    import torch_geometric.nn as pyg_nn

    return pyg_nn.Linear(-1, output_dim)

```
