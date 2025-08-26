from lynxkite_core import ops
from lynxkite_graph_analytics import core
from lynxkite_graph_analytics.pykeen_ops import (
    PyKEENModel,
    PyKEENModelWrapper,
    PyKEENModelName,
    mapped_triples_to_df,
    PyKEENSupportedOptimizers,
    TrainingType,
)

import enum
import pandas as pd
from pykeen import models
from pykeen.triples import TriplesFactory
from pykeen.datasets import inductive
from pykeen.evaluation.rank_based_evaluator import SampledRankBasedEvaluator
from pykeen.stoppers import EarlyStopper
from pykeen.training import SLCWATrainingLoop, LCWATrainingLoop


op = ops.op_registration("LynxKite Graph Analytics")


class InductiveDataset(str, enum.Enum):
    ILPC2022Large = "ILPC2022Large"
    ILPC2022Small = "ILPC2022Small"
    InductiveFB15k237 = "InductiveFB15k237"
    InductiveNELL = "InductiveNELL"
    InductiveWN18RR = "InductiveWN18RR"

    def to_dataset(self) -> inductive.LazyInductiveDataset:
        return getattr(inductive, self.value)()


@op("Import inductive dataset")
def import_inductive_dataset(*, dataset: InductiveDataset = InductiveDataset.ILPC2022Small):
    ds = dataset.to_dataset()
    transductive = mapped_triples_to_df(
        ds.transductive_training.mapped_triples,
        entity_to_id=ds.transductive_training.entity_to_id,
        relation_to_id=ds.transductive_training.relation_to_id,
    )
    inductive_inference = mapped_triples_to_df(
        ds.inductive_inference.mapped_triples,
        entity_to_id=ds.inductive_inference.entity_to_id,
        relation_to_id=ds.inductive_inference.relation_to_id,
    )
    inductive_testing = mapped_triples_to_df(
        ds.inductive_testing.mapped_triples,
        entity_to_id=ds.inductive_testing.entity_to_id,
        relation_to_id=ds.inductive_testing.relation_to_id,
    )
    assert ds.inductive_validation is not None, "Inductive validation set is missing!"
    inductive_validation = mapped_triples_to_df(
        ds.inductive_validation.mapped_triples,
        entity_to_id=ds.inductive_validation.entity_to_id,
        relation_to_id=ds.inductive_validation.relation_to_id,
    )

    bundle = core.Bundle()
    bundle.dfs["transductive_training"] = transductive
    bundle.dfs["inductive_inference"] = inductive_inference
    bundle.dfs["inductive_testing"] = inductive_testing
    bundle.dfs["inductive_validation"] = inductive_validation

    return bundle


@op("Split inductive dataset")
def inductively_split_dataset(
    bundle: core.Bundle,
    *,
    dataset_table: core.TableName,
    entity_ratio: float = 0.5,
    training_ratio: float = 0.8,
    testing_ratio: float = 0.1,
    validation_ratio: float = 0.1,
    seed: int = 42,
):
    """
    test.

    Args:
        entity_ratio: How many percent of the entities in the dataset should be in the transductive training graph. If `0` semi-inductive split is applied, else fully-inductive split is applied
        training_ratio: When semi-inductive this is *entity* ratio, when fully-inductive this is the inference training split
        testing_ratio: When semi-inductive this is *entity* ratio, when fully-inductive this is the inference testing split
        validation_ratio: When semi-inductive this is *entity* ratio, when fully-inductive this is the inference validation split
    """
    bundle = bundle.copy()

    bundle.dfs[dataset_table]["head"] = bundle.dfs[dataset_table]["head"].astype(str)
    bundle.dfs[dataset_table]["tail"] = bundle.dfs[dataset_table]["tail"].astype(str)
    tf_all = TriplesFactory.from_labeled_triples(
        bundle.dfs[dataset_table][["head", "relation", "tail"]].to_numpy(),
    )
    ratios = (training_ratio, testing_ratio, validation_ratio)
    if entity_ratio == 0:
        tf_training, tf_validation, tf_testing = tf_all.split_semi_inductive(
            ratios=ratios, random_state=seed
        )
        inference_set = mapped_triples_to_df(
            tf_training.mapped_triples,
            tf_training.entity_to_id,
            tf_training.relation_to_id,
        )
        testing_set = mapped_triples_to_df(
            tf_testing.mapped_triples,
            tf_testing.entity_to_id,
            tf_testing.relation_to_id,
        )
        validation_set = mapped_triples_to_df(
            tf_validation.mapped_triples,
            tf_validation.entity_to_id,
            tf_validation.relation_to_id,
        )
        training_set = inference_set.sample(frac=0.7, random_state=seed)
        bundle.dfs["transductive_training"] = training_set
        bundle.dfs["inductive_inference"] = inference_set
        bundle.dfs["inductive_testing"] = testing_set
        bundle.dfs["inductive_validation"] = validation_set
        return bundle

    tf_training, tf_inference, tf_validation, tf_testing = tf_all.split_fully_inductive(
        entity_split_train_ratio=entity_ratio, evaluation_triples_ratios=ratios, random_state=seed
    )
    transductive = mapped_triples_to_df(
        tf_training.mapped_triples,
        entity_to_id=tf_training.entity_to_id,
        relation_to_id=tf_training.relation_to_id,
    )
    inductive_inference = mapped_triples_to_df(
        tf_inference.mapped_triples,
        entity_to_id=tf_inference.entity_to_id,
        relation_to_id=tf_inference.relation_to_id,
    )
    inductive_testing = mapped_triples_to_df(
        tf_testing.mapped_triples,
        entity_to_id=tf_testing.entity_to_id,
        relation_to_id=tf_testing.relation_to_id,
    )
    inductive_validation = mapped_triples_to_df(
        tf_validation.mapped_triples,
        entity_to_id=tf_validation.entity_to_id,
        relation_to_id=tf_validation.relation_to_id,
    )

    bundle.dfs["transductive_training"] = transductive
    bundle.dfs["inductive_inference"] = inductive_inference
    bundle.dfs["inductive_testing"] = inductive_testing
    bundle.dfs["inductive_validation"] = inductive_validation

    return bundle


# TODO: Move Loss HERE
@op("Define inductive PyKEEN model")
def get_inductive_model(
    bundle: core.Bundle,
    *,
    triples_table: core.TableName,
    inference_table: core.TableName,
    interaction: PyKEENModel = PyKEENModel.DistMult,
    embedding_dim: int = 200,
    num_tokens: int = 2,
    aggregation: str = "mlp",
    use_GNN: bool = False,
    seed: int = 42,
    save_as: str = "InductiveModel",
):
    """
    Defines an InductiveNodePiece(GNN) model for inductive link prediction tasks.

    Args:
        triples_table: The transductive edges of the graph.
        inference_table: The inductive edges of the graph.
        interaction: Type of interaction the model will use for link prediction scoring.
        num_tokens: Number of hash tokens for each node representation, usually 66th percentiles of the number of unique incident relations per node.
        aggregation: Aggregation of multiple token representations to a single entity representation. The module assumes that this refers to a top-level torch function, or use 'mlp' for a two-layer built-in mlp aggregator.
    """
    bundle = bundle.copy()
    transductive_training = TriplesFactory.from_labeled_triples(
        bundle.dfs[triples_table][["head", "relation", "tail"]].to_numpy(),
        create_inverse_triples=True,
    )
    inductive_inference = TriplesFactory.from_labeled_triples(
        bundle.dfs[inference_table][["head", "relation", "tail"]].to_numpy(),
        relation_to_id=transductive_training.relation_to_id,
        create_inverse_triples=True,
    )
    model_cls = models.InductiveNodePieceGNN if use_GNN else models.InductiveNodePiece
    base_model_cls = interaction.to_class(transductive_training, embedding_dim, 42)
    assert isinstance(base_model_cls, models.ERModel), "Base model class is not an ERModel"
    interaction_cls = base_model_cls.interaction
    model = model_cls(
        triples_factory=transductive_training,  # training factory, used to tokenize training nodes
        inference_factory=inductive_inference,  # inference factory, used to tokenize inference nodes
        loss="Pairwise Hinge Loss",
        interaction=interaction_cls,
        embedding_dim=embedding_dim,
        num_tokens=num_tokens,  # length of a node hash - how many unique relations per node will be used
        aggregation=aggregation,  # aggregation function, defaults to an MLP, can be any PyTorch function
        random_seed=seed,
    )

    model_wrapper = PyKEENModelWrapper(
        model=model,
        embedding_dim=embedding_dim,
        entity_to_id=inductive_inference.entity_to_id,
        relation_to_id=transductive_training.relation_to_id,
        edges_data=bundle.dfs[triples_table][["head", "relation", "tail"]],
        seed=seed,
        interaction=model.interaction,
        inductive_inference=bundle.dfs[inference_table][["head", "relation", "tail"]],
        inductive_kwargs=dict(
            num_tokens=num_tokens,
            aggregation=aggregation,
            use_GNN="True" if use_GNN else "False",
        ),
    )

    bundle.other[save_as] = model_wrapper
    return bundle


# TODO: Move loss_function away from here to `define inductive model`
@op("Train inductive model", slow=True)
def train_inductive_pykeen_model(
    bundle: core.Bundle,
    *,
    model_name: PyKEENModelName,
    transductive_table_name: core.TableName,
    inductive_inference_table: core.TableName,
    inductive_validation_table: core.TableName,
    optimizer_type: PyKEENSupportedOptimizers = PyKEENSupportedOptimizers.Adam,
    loss_function: str = "BCEWithLogitsLoss",
    epochs: int = 5,
    training_approach: TrainingType = TrainingType.sLCWA,
):
    bundle = bundle.copy()
    model_wrapper: PyKEENModelWrapper = bundle.other.get(model_name)
    model = model_wrapper.model
    transductive_training = TriplesFactory.from_labeled_triples(
        bundle.dfs[transductive_table_name][["head", "relation", "tail"]].to_numpy(),
        relation_to_id=model_wrapper.relation_to_id,
        create_inverse_triples=True,
    )
    inductive_inference = TriplesFactory.from_labeled_triples(
        bundle.dfs[inductive_inference_table][["head", "relation", "tail"]].to_numpy(),
        entity_to_id=model_wrapper.entity_to_id,
        relation_to_id=model_wrapper.relation_to_id,
        create_inverse_triples=True,
    )
    inductive_validation = TriplesFactory.from_labeled_triples(
        bundle.dfs[inductive_validation_table][["head", "relation", "tail"]].to_numpy(),
        entity_to_id=model_wrapper.entity_to_id,
        relation_to_id=model_wrapper.relation_to_id,
        create_inverse_triples=True,
    )
    training_loop_cls = (
        SLCWATrainingLoop if training_approach == TrainingType.sLCWA else LCWATrainingLoop
    )
    loop_kwargs = (
        dict(
            negative_sampler_kwargs=dict(num_negs_per_pos=32),
        )
        if training_approach == TrainingType.sLCWA
        else dict()
    )
    training_loop = training_loop_cls(
        triples_factory=transductive_training,  # training triples
        model=model,
        optimizer=optimizer_type,
        mode="training",  # necessary to specify for the inductive mode - training has its own set of nodes
        **loop_kwargs,
    )

    # Validation and Test evaluators use a restricted protocol ranking against 50 random negatives
    valid_evaluator = SampledRankBasedEvaluator(
        mode="validation",  # necessary to specify for the inductive mode - this will use inference nodes
        evaluation_factory=inductive_validation,  # validation triples to predict
        additional_filter_triples=inductive_inference.mapped_triples,  # filter out true inference triples
    )

    early_stopper = EarlyStopper(
        model=model,
        training_triples_factory=inductive_inference,
        evaluation_triples_factory=inductive_validation,
        frequency=5,
        patience=40,
        metric="ah@k",
        result_tracker=None,
        evaluation_batch_size=256,
        evaluator=valid_evaluator,
    )

    # Training starts here
    losses = training_loop.train(
        triples_factory=transductive_training,
        stopper=early_stopper,
        num_epochs=epochs,
    )

    model_wrapper.trained = True

    bundle.dfs["training"] = pd.DataFrame({"training_loss": losses})
    bundle.dfs["early_stopper_metric"] = pd.DataFrame(
        {"early_stopper_metric": early_stopper.results}
    )
    bundle.other[model_name] = model_wrapper

    return bundle


@op("Evaluate inductive model")
def eval_inductive_model(
    bundle: core.Bundle,
    *,
    model_name: PyKEENModelName,
    inductive_testing_table: core.TableName,
    inductive_inference_table: core.TableName,
    inductive_validation_table: core.TableName,
    metrics_str: str = "ALL",
):
    bundle = bundle.copy()
    model = bundle.other.get(model_name)
    inductive_testing = TriplesFactory.from_labeled_triples(
        bundle.dfs[inductive_testing_table][["head", "relation", "tail"]].to_numpy(),
        entity_to_id=model.entity_to_id,
        relation_to_id=model.relation_to_id,
    )
    inductive_inference = TriplesFactory.from_labeled_triples(
        bundle.dfs[inductive_inference_table][["head", "relation", "tail"]].to_numpy(),
        entity_to_id=model.entity_to_id,
        relation_to_id=model.relation_to_id,
    )
    inductive_validation = TriplesFactory.from_labeled_triples(
        bundle.dfs[inductive_validation_table][["head", "relation", "tail"]].to_numpy(),
        entity_to_id=model.entity_to_id,
        relation_to_id=model.relation_to_id,
    )

    test_evaluator = SampledRankBasedEvaluator(
        mode="testing",  # necessary to specify for the inductive mode - this will use inference nodes
        evaluation_factory=inductive_testing,  # test triples to predict
        additional_filter_triples=[
            inductive_inference.mapped_triples,
            inductive_validation.mapped_triples,
        ],
    )

    # Test evaluation
    result = test_evaluator.evaluate(
        model=model,
        mapped_triples=inductive_testing.mapped_triples,
        additional_filter_triples=[
            inductive_inference.mapped_triples,
            inductive_validation.mapped_triples,
        ],
        batch_size=256,
    )
    if metrics_str == "ALL":
        bundle.dfs["metrics"] = result.to_df()
        return bundle

    metrics = metrics_str.split(",")
    metrics_df = pd.DataFrame(columns=["metric", "score"])

    for metric in metrics:
        metric = metric.strip()
        try:
            score = result.get_metric(metric)
        except Exception as e:
            raise Exception(f"Possibly unknown metric: {metric}") from e
        metrics_df = pd.concat(
            [metrics_df, pd.DataFrame([[metric, score]], columns=metrics_df.columns)]
        )

    bundle.dfs["metrics"] = metrics_df

    return bundle
