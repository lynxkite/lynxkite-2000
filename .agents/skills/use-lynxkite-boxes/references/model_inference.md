**Model inference:**
Executes a trained model.
```python
@op("Model inference", slow=True)
def model_inference(
    bundle: core.Bundle,
    *,
    model_name: pytorch_core.PyTorchModelName = "model",
    input_mapping: ModelInferenceInputMapping | None,
    output_mapping: ModelOutputMapping | None,
    batch_size: int = 1,
):
    """Executes a trained model."""
    if input_mapping is None or output_mapping is None:
        return ops.Result(bundle, error="Mapping is unset.")
    m: pytorch_core.ModelConfig = bundle.other[model_name]
    assert m.trained, "The model is not trained."
    input_ctx = pytorch_core.InputContext(batch_size=batch_size, batch_index=0)
    outputs = {}
    tbatch = tqdm(total=100)  # Initial guess. Will update after the first iteration.
    while (
        input_ctx.total_samples is None
        or input_ctx.batch_index * batch_size < input_ctx.total_samples
    ):
        inputs = m.inputs_from_bundle(bundle, m.model_inputs, input_mapping, input_ctx)
        assert input_ctx.total_samples is not None
        tbatch.total = input_ctx.total_samples // batch_size
        batch_outputs = m.inference(inputs)
        for k, v in batch_outputs.items():
            v = v.detach().numpy()
            outputs.setdefault(k, []).extend(v.tolist())
        input_ctx.batch_index += 1
    bundle = bundle.copy()
    copied = set()
    for k, v in output_mapping.map.items():
        df = v.get("table_name")
        col = v.get("column")
        if not df or not col:
            continue
        if df not in copied:
            bundle.dfs[df] = bundle.dfs[df].copy()
            copied.add(df)
        bundle.dfs[df][col] = outputs[k]
    return bundle

```
Custom types:
  - model_name: typing.Annotated[str, {'format': 'dropdown', 'metadata_query': "[].other.*[] | [?type == 'pytorch-model'].key"}]
