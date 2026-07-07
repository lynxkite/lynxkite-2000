**Extract embeddings from PyKEEN model:**

```python
@op("Extract embeddings from PyKEEN model", color="orange", icon="database-export")
def extract_from_pykeen(
    bundle: core.Bundle,
    *,
    model_name: PyKEENModelName = "PyKEENmodel",
):
    bundle = bundle.copy()
    model_wrapper = bundle.other[model_name]
    model = model_wrapper.model
    state_dict = model.state_dict()

    entity_embeddings = []
    for key in state_dict.keys():
        if "entity" in key.lower() and "embedding" in key.lower():
            entity_embeddings.append(state_dict[key].cpu().detach().numpy())

    id_to_entity = {v: k for k, v in model_wrapper.entity_to_id.items()}
    for i, embedding in enumerate(entity_embeddings):
        entity_embedding_df = pd.DataFrame({"embedding": list(embedding)})
        entity_embedding_df["node_label"] = entity_embedding_df.index.map(id_to_entity)
        bundle.dfs[f"node_embedding_{i}"] = entity_embedding_df

    relation_embeddings = []
    for key in state_dict.keys():
        if "relation" in key.lower() and "embedding" in key.lower():
            relation_embeddings.append(state_dict[key].cpu().detach().numpy())

    id_to_relation = {v: k for k, v in model_wrapper.relation_to_id.items()}
    for i, embedding in enumerate(relation_embeddings):
        relation_embedding_df = pd.DataFrame({"embedding": list(embedding)})
        relation_embedding_df["relation_label"] = relation_embedding_df.index.map(id_to_relation)
        bundle.dfs[f"relation_embedding_{i}"] = relation_embedding_df

    return bundle

```
Custom types:
  - model_name: typing.Annotated[str, {'format': 'dropdown', 'metadata_query': "[].other.*[] | [?type == 'pykeen-model'].key"}]
