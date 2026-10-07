**Mean pool:**

```python
@op("Mean pool")
def mean_pool(x):
    import torch_geometric.nn as pyg_nn

    return pyg_nn.global_mean_pool

```
