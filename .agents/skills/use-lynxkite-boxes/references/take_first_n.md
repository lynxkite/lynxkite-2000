**Take first n:**
Returns the first n elements from the input tensor.
```python
@op("Take first n")
def take_first_n(x, *, n: int = 1):
    """
    Returns the first n elements from the input tensor.
    :param x: The input tensor.
    :param n: The number of elements to take from the beginning of the tensor.
    """
    return x[:n]

```
