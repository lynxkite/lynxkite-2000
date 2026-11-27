**Drop first n:**
Returns the input tensor with the first n elements dropped.
```python
@op("Drop first n")
def drop_first_n(x, *, n: int = 1):
    """
    Returns the input tensor with the first n elements dropped.
    :param x: The input tensor.
    :param n: The number of elements to drop from the beginning of the tensor.
    """
    return x[n:]

```
