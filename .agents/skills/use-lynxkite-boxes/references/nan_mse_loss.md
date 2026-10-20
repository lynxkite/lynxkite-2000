**NaN-aware MSE loss:**
Mean squared error over the entries of the label tensor that are not NaN.
```python
@op("NaN-aware MSE loss")
def nan_mse_loss(pred, label):
    """Mean squared error over the entries of the label tensor that are not NaN.
    :param pred: The tensor of predictions.
    :param label: The label tensor, with NaN for entries that should be ignored.
    """

    def _nan_mse_loss(pred, label):
        if pred.shape != label.shape:
            if (
                pred.ndim == label.ndim + 1
                and pred.shape[-1] == 1
                and pred.shape[:-1] == label.shape
            ):
                label = label.unsqueeze(-1)
            elif (
                label.ndim == pred.ndim + 1
                and label.shape[-1] == 1
                and label.shape[:-1] == pred.shape
            ):
                pred = pred.unsqueeze(-1)

        valid = torch.isfinite(label)
        if not torch.any(valid):
            return pred.sum() * 0.0
        diff = pred[valid] - label[valid]
        return torch.mean(diff**2)

    return _nan_mse_loss

```
