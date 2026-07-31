**Train model:**
Trains the selected model on the selected dataset.
Training parameters specific to the model are set in the model definition,
while parameters specific to the hardware environment and dataset are set here.
```python
@op("Train model", slow=True)
def train_model(
    bundle: core.Bundle,
    *,
    model_name: pytorch_core.PyTorchModelName = "model",
    input_mapping: ModelTrainingInputMapping | None,
    epochs: int = 1,
    batch_size: int = 1,
):
    """
    Trains the selected model on the selected dataset.
    Training parameters specific to the model are set in the model definition,
    while parameters specific to the hardware environment and dataset are set here.
    """
    if input_mapping is None:
        return ops.Result(bundle, error="No inputs are selected.")
    m: pytorch_core.ModelConfig = bundle.other[model_name].copy()
    tepochs = tqdm(range(epochs), desc="Training model")
    losses = []
    input_ctx = pytorch_core.InputContext(batch_size=batch_size, batch_index=0)
    for _ in tepochs:
        total_loss = 0
        tbatch = tqdm(total=100)  # Initial guess. Will update after the first iteration.
        input_ctx.batch_index = 0
        while (
            input_ctx.total_samples is None
            or input_ctx.batch_index * batch_size < input_ctx.total_samples
        ):
            inputs = m.inputs_from_bundle(
                bundle,
                list(set(m.model_inputs) | set(m.loss_inputs) - set(m.model_outputs)),
                input_mapping,
                input_ctx,
            )
            assert input_ctx.total_samples is not None
            tbatch.total = input_ctx.total_samples // batch_size
            loss = m.train(inputs)
            total_loss += loss
            input_ctx.batch_index += 1
        mean_loss = total_loss / input_ctx.total_samples
        tepochs.set_postfix({"loss": mean_loss})
        losses.append(mean_loss)
    m.trained = True
    bundle = bundle.copy()
    bundle.dfs["training"] = pd.DataFrame({"training_loss": losses})
    bundle.other[model_name] = m
    return bundle

```
Custom types:
  - model_name: typing.Annotated[str, {'format': 'dropdown', 'metadata_query': "[].other.*[] | [?type == 'pytorch-model'].key"}]
