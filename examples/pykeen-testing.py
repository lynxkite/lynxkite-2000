from lynxkite.core import ops
from lynxkite_graph_analytics import core

from lynxkite_graph_analytics.pykeen_ops import PyKEENModelWrapper
from lynxkite_graph_analytics.pykeen_ops import PyKEENModel
from lynxkite_graph_analytics.pykeen_ops import req_inverse_triples
from pykeen.triples import TriplesFactory
from ast import literal_eval
import numpy as np
import torch


op = ops.op_registration("LynxKite Graph Analytics")


# TODO: Needs more work
@op("Load embedding into PyKEEN model")
def load_pykeen_embeddings(
    dataset: core.Bundle,
    embeddings: core.Bundle,
    *,
    model: PyKEENModel = PyKEENModel.TransE,
    node_embeddings_table: core.TableName,
    edge_embeddings_table: core.TableName,
    save_as: str = "PreparedModel",
):
    """
    Assuming tables has the following columns: [label, id, embedding]
    Where embedding is a list of floats, or a python list.
    """
    bundle_dataset = dataset.copy()
    bundle_embeddings = embeddings.copy()

    node_embeddings = (
        bundle_embeddings.dfs[node_embeddings_table][["id", "embedding"]]
        .set_index("id")["embedding"]
        .apply(lambda x: x if isinstance(x, np.ndarray) else literal_eval(x))
        .to_numpy()
    )
    edge_embeddings = (
        bundle_embeddings.dfs[edge_embeddings_table][["id", "embedding"]]
        .set_index("id")["embedding"]
        .apply(lambda x: x if isinstance(x, np.ndarray) else literal_eval(x))
        .to_numpy()
    )
    node_embeddings = np.stack(node_embeddings)
    edge_embeddings = np.stack(edge_embeddings)

    print(node_embeddings.shape)
    print(edge_embeddings.shape)
    actual_model = model.to_class(
        triples_factory=TriplesFactory.from_labeled_triples(
            bundle_dataset.dfs["edges"][["source", "relation", "target"]].values,
            entity_to_id=bundle_dataset.dfs["nodes"].set_index("label")["id"].to_dict(),
            relation_to_id=bundle_dataset.dfs["relations"].set_index("label")["id"].to_dict(),
            create_inverse_triples=req_inverse_triples(model),
        ),
        embedding_dim=node_embeddings.shape[1],
    )

    state_dict = actual_model.state_dict()

    # Find the correct parameter names for entity and relation embeddings
    entity_embedding_key = None
    relation_embedding_key = None

    for key in state_dict.keys():
        if "entity" in key.lower() and "embedding" in key.lower() and entity_embedding_key is None:
            entity_embedding_key = key
        elif (
            "relation" in key.lower()
            and "embedding" in key.lower()
            and relation_embedding_key is None
        ):
            relation_embedding_key = key

    print(f"Entity embedding key: {entity_embedding_key}")
    print(f"Relation embedding key: {relation_embedding_key}")

    if entity_embedding_key:
        state_dict[entity_embedding_key] = torch.from_numpy(node_embeddings).float()
    if relation_embedding_key:
        state_dict[relation_embedding_key] = torch.from_numpy(edge_embeddings).float()

    actual_model.load_state_dict(state_dict)

    # Freeze the embedding weights
    for name, param in actual_model.named_parameters():
        if ("entity" in name.lower() and "embedding" in name.lower()) or (
            "relation" in name.lower() and "embedding" in name.lower()
        ):
            param.requires_grad = False
            print(f"Frozen embedding parameter: {name}")

    model_wrapper = PyKEENModelWrapper(actual_model)

    bundle_dataset.other[save_as] = model_wrapper
    return bundle_dataset
