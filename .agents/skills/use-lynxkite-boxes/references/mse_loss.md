**MSE loss:**

```python
@op("MSE loss")
def mse_loss(x, y):
    def _loss(x, y):
        # Common regression case: predictions [N, 1] vs labels [N].
        if x.shape != y.shape:
            if x.ndim == y.ndim + 1 and x.shape[-1] == 1 and x.shape[:-1] == y.shape:
                y = y.unsqueeze(-1)
            elif y.ndim == x.ndim + 1 and y.shape[-1] == 1 and y.shape[:-1] == x.shape:
                x = x.unsqueeze(-1)
        return torch.nn.functional.mse_loss(x, y)

    return _loss

```
