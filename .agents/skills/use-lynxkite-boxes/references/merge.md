**Merge:**
Merge multiple inputs
```python
@op("Merge", icon="arrows-join")
def merge(
    bundles: list[core.Bundle],
    *,
    merge_mode: bundle.BundleMergeMode = bundle.BundleMergeMode.must_be_unique,
):
    """Merge multiple inputs"""
    b = bundle.merge_bundles(bundles, merge_mode=merge_mode)
    return b

```
