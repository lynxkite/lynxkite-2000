**Pick element by constant:**
Picks an element from the input tensor by the specified index parameter.
```python
@op("Pick element by constant")
def pick_element_by_constant(x, *, index: int = 0):
    """
    Picks an element from the input tensor by the specified index parameter.
    :param x: The input tensor.
    :param index: The index of the element to pick.
    """
    return x[index]

```
