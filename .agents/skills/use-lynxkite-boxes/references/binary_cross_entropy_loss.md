**Binary cross-entropy with logits loss:**

```python
@op("Binary cross-entropy with logits loss", outputs=["loss"])
def binary_cross_entropy_loss(x, y):
    return torch.nn.functional.binary_cross_entropy_with_logits

```
