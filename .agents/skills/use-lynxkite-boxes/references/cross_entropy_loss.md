**Cross-entropy loss:**
Returns the cross-entropy loss for the given input tensors.
```python
@op("Cross-entropy loss", outputs=["loss"])
def cross_entropy_loss(x, y):
    """
    Returns the cross-entropy loss for the given input tensors.
    :param x: the input tensor.
    :param y: the target tensor.
    """
    return torch.nn.functional.cross_entropy

```
