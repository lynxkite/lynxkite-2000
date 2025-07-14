from lynxkite.core import ops
from lynxkite_graph_analytics import core

import pandas as pd
from pykeen import models
from pykeen import evaluation
from pykeen.triples import TriplesFactory
import enum

op = ops.op_registration("LynxKite Graph Analytics")


class EvaluatorTypes(str, enum.Enum):
    ClassificationEvaluator = "Classification Evaluator"
    MacroRankBasedEvaluator = "Macro Rank Based Evaluator"
    RankBasedEvaluator = "Rank Based Evaluator"

    def to_class(self):
        return getattr(evaluation, self.name.replace(" ", ""))()


@op("Evaluate model")
def evaluate(
    bundle: core.Bundle,
    *,
    model_name: core.PyKEENModelName = "PyKEENmodel",
    evaluator: EvaluatorTypes = EvaluatorTypes.RankBasedEvaluator,
    metrics_str: str = "ALL",
):
    """Metrics are a comma separated list, "ALL" if all metrics are needed"""
    bundle = bundle.copy()
    model: models.Model = bundle.other.get(model_name)
    evaluator = evaluator.to_class()
    testing_triples = TriplesFactory.from_labeled_triples(
        bundle.dfs["triples_test"][["head", "relation", "tail"]].values
    )
    training_triples = TriplesFactory.from_labeled_triples(
        bundle.dfs["triples_train"][["head", "relation", "tail"]].values
    )
    validation_triples = TriplesFactory.from_labeled_triples(
        bundle.dfs["triples_val"][["head", "relation", "tail"]].values
    )

    evaluated = evaluator.evaluate(
        model,
        testing_triples.mapped_triples,
        additional_filter_triples=[
            training_triples.mapped_triples,
            validation_triples.mapped_triples,
        ],
    )
    if metrics_str == "ALL":
        bundle.dfs["metrics"] = evaluated.to_df()
        return bundle

    metrics = metrics_str.split(",")
    metrics_df = pd.DataFrame(columns=["metric", "score"])

    for metric in metrics:
        metric = metric.strip()
        try:
            score = evaluated.get_metric(metric)
        except Exception as e:
            raise Exception(f"Possibly unknown metric: {metric}\nError: {e}")
        metrics_df = pd.concat(
            [metrics_df, pd.DataFrame([[metric, score]], columns=metrics_df.columns)]
        )

    bundle.dfs["metrics"] = metrics_df

    return bundle


@op("Load embedding into PyKEEN model")
def load_pykeen_embeddings(bundle: core.Bundle, *, _model_name: str):
    """
    Assuming embeddings are torch tensors
    Entity names has to be a file with comma separated entity names
    """
    node_embeddings = bundle.other["node_embedding"]
    edge_embeddings = bundle.other["edge_embedding"]

    model = models.TransE(
        triples_factory=TriplesFactory.from_labeled_triples(
            bundle.dfs["triples_train"][["head", "relation", "tail"]].values
        ),
        embedding_dim=node_embeddings.shape[1],
    )
    model.entity_representations[0].weight.data = node_embeddings
    model.relation_representations[0].weight.data = edge_embeddings

    bundle.other["model"] = model

    return bundle
