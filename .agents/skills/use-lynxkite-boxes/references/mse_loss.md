**MSE loss:**

```python
@op("MSE loss")
def mse_loss(x, y):
    return torch.nn.functional.mse_loss

```
