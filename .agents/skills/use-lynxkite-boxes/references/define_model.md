**Define model:**
Trains the selected model on the selected dataset. Most training parameters are set in the model definition.
```python
@op("Define model", color="purple")
def define_model(
    bundle: core.Bundle,
    *,
    model_workspace: str,
    save_as: str = "model",
):
    """Trains the selected model on the selected dataset. Most training parameters are set in the model definition."""
    assert model_workspace, "Model workspace is unset."
    ws = load_ws(model_workspace + ".lynxkite.json")
    m = pytorch_core.build_model(ws)
    m.source_workspace = model_workspace
    bundle = bundle.copy()
    bundle.other[save_as] = m
    return bundle

```
