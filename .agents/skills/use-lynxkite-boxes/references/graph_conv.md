**Graph conv:**
A graph convolution layer.
```python
@op("Graph conv")
def graph_conv(x, edges, *, convolution_type: ConvolutionTypes, output_dim: int):
    """A graph convolution layer.
    :param x: The feature tensor.
    :param edges: The edge tensor.
    :param convolution_type: The type of graph convolution to apply.
    :param output_dim: The number of outputs of this layer.
    """
    import torch_geometric.nn as pyg_nn

    conv = getattr(pyg_nn, convolution_type.value)
    return conv(-1, output_dim)

```
