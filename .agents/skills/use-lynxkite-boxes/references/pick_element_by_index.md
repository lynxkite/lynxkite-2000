**Pick element by index:**
Picks an element from the input tensor by the specified input index.
```python
@op("Pick element by index")
def pick_element_by_index(x, index):
    """
    Picks an element from the input tensor by the specified input index.
    :param x: The input tensor.
    :param index: The index of the element to pick.
    """
    return x[index]

```
