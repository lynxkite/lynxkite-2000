from lynxkite.core import ops
from lynxkite_graph_analytics import core

from pykeen import models
from pykeen.triples import TriplesFactory


op = ops.op_registration("LynxKite Graph Analytics")


# TODO: Fix, and update to newest conventions
@op("Load embedding into PyKEEN model")
def load_pykeen_embeddings(bundle: core.Bundle, *, _model_name: str):
    """
    Assuming embeddings are torch tensors
    Entity names has to be a file with comma separated entity names
    Very much work-in-progress...
    """
    node_embeddings = bundle.other["node_embedding"]
    edge_embeddings = bundle.other["edge_embedding"]

    model = models.TransE(
        triples_factory=TriplesFactory.from_labeled_triples(
            bundle.dfs["edges_train"][["head", "relation", "tail"]].values
        ),
        embedding_dim=node_embeddings.shape[1],
    )
    model.entity_representations[0].weight.data = node_embeddings
    model.relation_representations[0].weight.data = edge_embeddings

    bundle.other["model"] = model

    return bundle
