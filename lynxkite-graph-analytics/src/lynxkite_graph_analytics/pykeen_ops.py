"""PyKEEN graph embedding operations."""

from lynxkite_core import ops
import numpy as np
from . import core

import typing
import pandas as pd
import io
import enum
import torch
from torch import nn
from pykeen import models, evaluation, stoppers
from pykeen.nn import Interaction, combination
from pykeen.pipeline import pipeline, PipelineResult
from pykeen.datasets import get_dataset, inductive
from pykeen.predict import predict_triples, predict_target, predict_all
from pykeen.triples import (
    CoreTriplesFactory,
    TriplesFactory,
    TriplesNumericLiteralsFactory,
    leakage,
)
from pykeen.models import LiteralModel
from pykeen.training import SLCWATrainingLoop, LCWATrainingLoop


op = ops.op_registration(core.ENV, "Graph embedding and link prediction")

# Used for initializing TriplesFactory when only mappings are needed.
DUMMY_TRIPLET = np.array([[0, 0, 0]])

PyKEENModelName = typing.Annotated[
    str,
    {
        "format": "dropdown",
        "metadata_query": "[].other.*[] | [?type == 'pykeen-model'].key",
    },
]
"""A type annotation to be used for parameters of an operation. PyKEENModelName is
rendered as a dropdown in the frontend, listing the PyKEEN models in the Bundle.
The model name is passed to the operation as a string."""


def factory_to_df(factory: CoreTriplesFactory) -> pd.DataFrame:
    """Convert a TriplesFactory to a DataFrame with labeled columns."""
    df = factory.tensor_to_df(factory.mapped_triples)
    return df[["head_label", "relation_label", "tail_label"]].rename(
        columns={"head_label": "head", "relation_label": "relation", "tail_label": "tail"}
    )


class PyKEENDataset(str, enum.Enum):
    AristoV4 = "AristoV4"
    BioKG = "BioKG"
    CKG = "CKG"
    CN3l = "CN3l"
    CoDExLarge = "CoDExLarge"
    CoDExMedium = "CoDExMedium"
    CoDExSmall = "CoDExSmall"
    ConceptNet = "ConceptNet"
    Countries = "Countries"
    CSKG = "CSKG"
    DB100K = "DB100K"
    DBpedia50 = "DBpedia50"
    DRKG = "DRKG"
    FB15k = "FB15k"
    FB15k237 = "FB15k237"
    Globi = "Globi"
    Hetionet = "Hetionet"
    Kinships = "Kinships"
    Nations = "Nations"
    NationsLiteral = "NationsLiteral"
    OGBBioKG = "OGBBioKG"
    OGBWikiKG2 = "OGBWikiKG2"
    OpenBioLink = "OpenBioLink"
    OpenBioLinkLQ = "OpenBioLinkLQ"
    OpenEA = "OpenEA"
    PharMeBINet = "PharMeBINet"
    PharmKG = "PharmKG"
    PharmKG8k = "PharmKG8k"
    PrimeKG = "PrimeKG"
    UMLS = "UMLS"
    WD50KT = "WD50KT"
    Wikidata5M = "Wikidata5M"
    WK3l120k = "WK3l120k"
    WK3l15k = "WK3l15k"
    WN18 = "WN18"
    WN18RR = "WN18RR"
    YAGO310 = "YAGO310"

    def to_dataset(self):
        return get_dataset(dataset=self.value)


class InductiveDataset(str, enum.Enum):
    ILPC2022Large = "ILPC2022Large"
    ILPC2022Small = "ILPC2022Small"
    InductiveFB15k237 = "InductiveFB15k237"
    InductiveNELL = "InductiveNELL"
    InductiveWN18RR = "InductiveWN18RR"

    def to_dataset(self) -> inductive.LazyInductiveDataset:
        return getattr(inductive, self.value)()


@op("Import PyKEEN dataset", color="green", icon="3d-scale")
def import_pykeen_dataset_path(*, dataset: PyKEENDataset = PyKEENDataset.Nations) -> core.Bundle:
    """Imports a dataset from the PyKEEN library."""
    ds = dataset.to_dataset()
    bundle = core.Bundle()

    bundle.dfs["edges_train"] = factory_to_df(factory=ds.training)
    bundle.dfs["edges_test"] = factory_to_df(factory=ds.testing)
    if ds.validation:
        bundle.dfs["edges_val"] = factory_to_df(factory=ds.validation)

    bundle.dfs["nodes"] = pd.DataFrame(
        {
            "id": list(ds.entity_to_id.values()),
            "label": list(ds.entity_to_id.keys()),
        }
    )
    bundle.dfs["relations"] = pd.DataFrame(
        {
            "id": list(ds.relation_to_id.values()),
            "label": list(ds.relation_to_id.keys()),
        }
    )

    df_all = pd.concat(
        [bundle.dfs["edges_train"], bundle.dfs["edges_test"], bundle.dfs["edges_val"]],
        ignore_index=True,
    )
    bundle.dfs["edges"] = pd.DataFrame(
        {
            "head": df_all["head"].tolist(),
            "tail": df_all["tail"].tolist(),
            "relation": df_all["relation"].tolist(),
        }
    )
    return bundle


@op("Inductive setting", "Import inductive dataset", color="green", icon="affiliate-filled")
def import_inductive_dataset(*, dataset: InductiveDataset = InductiveDataset.ILPC2022Small):
    """Imports an inductive dataset from the PyKEEN library."""
    ds = dataset.to_dataset()
    bundle = core.Bundle()
    bundle.dfs["transductive_training"] = pd.DataFrame(
        ds.transductive_training.triples, columns=["head", "relation", "tail"]
    )
    bundle.dfs["inductive_inference"] = pd.DataFrame(
        ds.inductive_inference.triples, columns=["head", "relation", "tail"]
    )
    bundle.dfs["inductive_testing"] = pd.DataFrame(
        ds.inductive_testing.triples, columns=["head", "relation", "tail"]
    )
    assert ds.inductive_validation is not None
    bundle.dfs["inductive_validation"] = pd.DataFrame(
        ds.inductive_validation.triples, columns=["head", "relation", "tail"]
    )
    return bundle


@op("Inductive setting", "Split inductive dataset", color="orange", icon="circle-half-2")
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
    Splits incoming data into 4 subsets. Transductive training on which training should be run, inductive inference on which during training inference is done.
    Inference testing and validation sets that can be used to evaluate model performance.

    Args:
        entity_ratio: How many percent of the entities in the dataset should be in the transductive training graph. If `0` semi-inductive split is applied, else fully-inductive split is applied
        training_ratio: When semi-inductive this is *entity* ratio, when fully-inductive this is the inference training split
        testing_ratio: When semi-inductive this is *entity* ratio, when fully-inductive this is the inference testing split
        validation_ratio: When semi-inductive this is *entity* ratio, when fully-inductive this is the inference validation split
    """
    bundle = bundle.copy()

    bundle.dfs[dataset_table] = bundle.dfs[dataset_table].astype(str)
    tf_all = TriplesFactory.from_labeled_triples(
        bundle.dfs[dataset_table][["head", "relation", "tail"]].to_numpy(),
    )
    ratios = (training_ratio, testing_ratio, validation_ratio)
    if entity_ratio == 0:
        tf_training, tf_validation, tf_testing = tf_all.split_semi_inductive(
            ratios=ratios, random_state=seed
        )
    else:
        tf_training, tf_inference, tf_validation, tf_testing = tf_all.split_fully_inductive(
            entity_split_train_ratio=entity_ratio,
            evaluation_triples_ratios=ratios,
            random_state=seed,
        )
    transductive = pd.DataFrame(tf_training.triples, columns=["head", "relation", "tail"])
    inductive_testing = pd.DataFrame(tf_testing.triples, columns=["head", "relation", "tail"])
    inductive_val = pd.DataFrame(tf_validation.triples, columns=["head", "relation", "tail"])

    inductive_inference = (
        pd.concat([inductive_val, inductive_testing], ignore_index=True)
        if entity_ratio == 0
        else pd.DataFrame(tf_inference.triples, columns=["head", "relation", "tail"])
    )

    bundle.dfs["transductive_training"] = transductive
    bundle.dfs["inductive_inference"] = inductive_inference
    bundle.dfs["inductive_testing"] = inductive_testing
    bundle.dfs["inductive_validation"] = inductive_val

    return bundle


class PyKEENModelEnum(enum.StrEnum):
    def to_class(
        self, triples_factory: TriplesFactory, loss_func: str, embedding_dim: int, seed: int = 42
    ) -> models.Model:
        model = getattr(models, self.value)(
            triples_factory=triples_factory,
            loss=loss_func,
            embedding_dim=embedding_dim,
            random_seed=seed,
        )
        if torch.cuda.is_available():
            model = model.to(torch.device("cuda"))
        return model


class PyKEENModel1D(PyKEENModelEnum):
    CompGCN = "CompGCN"
    ComplEx = "ComplEx"
    ConvKB = "ConvKB"
    Cooccurrence_Filtered = "CooccurrenceFilteredModel"
    DistMA = "DistMA"
    DistMult = "DistMult"
    ER_MLP = "ERMLP"
    ER_MLPE = "ERMLPE"
    Fixed_Model = "FixedModel"
    HolE = "HolE"
    NodePiece = "NodePiece"
    ProjE = "ProjE"
    QuatE = "QuatE"
    RGCN = "RGCN"
    RESCAL = "RESCAL"
    RotatE = "RotatE"
    TorusE = "TorusE"
    TransE = "TransE"
    TransF = "TransF"
    TuckER = "TuckER"


class PyKEENModelMoreD(PyKEENModelEnum):
    locals().update({m.name: m.value for m in PyKEENModel1D})
    AutoSF = "AutoSF"
    BoxE = "BoxE"
    ConvE = "ConvE"
    Canonical_Tensor_Decomposition = "CP"
    CrossE = "CrossE"
    KG2E = "KG2E"
    MuRE = "MuRE"
    NTN = "NTN"
    PairRE = "PairRE"
    SimplE = "SimplE"
    Structured_Embedding = "SE"
    TransD = "TransD"
    TransH = "TransH"
    TransR = "TransR"


class PyKEENCombinations(str, enum.Enum):
    ComplexSeparated = "ComplexSeparated"
    ConcatProjection = "ConcatProjection"
    Gated = "Gated"

    def to_class(
        self, embedding_dim: int, literal_shape: int, **kwargs
    ) -> combination.Combination:  # ty: ignore[invalid-return-type]
        match self:
            case "ComplexSeparated":
                return combination.ComplexSeparatedCombination(
                    combination=combination.ConcatProjectionCombination,
                    combination_kwargs=dict(
                        input_dims=[embedding_dim, literal_shape],
                        output_dim=embedding_dim,
                        bias=True,
                        activation=nn.Tanh,
                    ),
                )
            case "ConcatProjection":
                return combination.ConcatProjectionCombination(
                    input_dims=[embedding_dim, literal_shape],
                    output_dim=embedding_dim,
                    bias=kwargs.get("bias", False),
                    dropout=float(kwargs.get("dropout", 0.0)),
                    activation=kwargs.get("activation", "ReLU"),
                )
            case "Gated":
                return combination.GatedCombination(
                    entity_dim=embedding_dim,
                    literal_dim=literal_shape,
                    input_dropout=float(kwargs.get("dropout", 0.0)),
                    gate_activation=kwargs.get("gate_activation", "Sigmoid"),
                    hidden_activation=kwargs.get("hidden_activation", "Tanh"),
                )


class PyKEENModelWrapper:
    """Wrapper to add metadata method to PyKEEN models for dropdown queries, and to enable caching of model"""

    def __init__(
        self,
        model: models.Model,
        loss: str,
        embedding_dim: int,
        entity_to_id: typing.Mapping[str, int],
        relation_to_id: typing.Mapping[str, int],
        edges_data: pd.DataFrame,
        seed: int,
        model_type: typing.Optional[PyKEENModelMoreD] = None,
        interaction: typing.Optional[Interaction] = None,
        inductive_inference: typing.Optional[pd.DataFrame] = None,
        inductive_kwargs: typing.Optional[dict] = None,
        combination: typing.Optional[PyKEENCombinations] = None,
        combination_kwargs: typing.Optional[dict] = None,
        literals_data: typing.Optional[pd.DataFrame] = None,
        trained: bool = False,
    ):
        if model_type is None:
            # either inductive or literals model
            if inductive_inference is not None:
                assert (
                    interaction is not None
                    and inductive_inference is not None
                    and combination is None
                    and literals_data is None
                ), (
                    "For inductive models, interaction and inductive inference table must be provided"
                )
            else:
                assert (
                    interaction is not None
                    and combination is not None
                    and literals_data is not None
                ), (
                    "For a model with literals, interaction, combination and literals data must be provided"
                )
        else:
            assert (
                interaction is None
                and combination is None
                and literals_data is None
                and inductive_inference is None
            ), "For transdutive models, only model_type must be provided"
        if torch.cuda.is_available():
            model = model.to(torch.device("cuda"))
        self.model = model
        self.loss = loss
        self.embedding_dim = embedding_dim
        self.entity_to_id = entity_to_id
        self.relation_to_id = relation_to_id
        self.edges_data = edges_data
        self.seed = seed
        self.model_type = model_type
        self.interaction = interaction
        self.combination = combination
        self.combination_kwargs = combination_kwargs
        self.literals_data = literals_data
        self.inductive_inference = inductive_inference
        self.inductive_kwargs = inductive_kwargs or {}
        self.trained = trained

    def metadata(self) -> dict:
        return {
            "type": "pykeen-model",
            "model_class": self.model.__class__.__name__,
            "module": self.model.__module__,
            "trained": self.trained,
            "embedding_dim": self.embedding_dim,
        }

    def __getattr__(self, name):
        # Delegate all other attributes to the wrapped model
        # Use object.__getattribute__ to avoid recursion when accessing self.model
        model = object.__getattribute__(self, "model")
        return getattr(model, name)

    def __str__(self):
        return str(self.model)

    def __getstate__(self):
        state = dict(self.__dict__)
        del state["model"]
        if self.trained:
            buffer = io.BytesIO()
            self.model.save_state(buffer)  # ty: ignore[invalid-argument-type]
            state["model_state"] = buffer.getvalue()

        return state

    def __setstate__(self, state: dict) -> None:
        model_state = state.pop("model_state", None)
        self.__dict__.update(state)
        if self.model_type is not None:
            self.model = self.model_type.to_class(
                triples_factory=prepare_triples(
                    self.edges_data,
                    entity_to_id=self.entity_to_id,
                    relation_to_id=self.relation_to_id,
                    inv_triples=req_inverse_triples(self.model_type),
                ),
                loss_func=self.loss,
                embedding_dim=self.embedding_dim,
                seed=self.seed,
            )
        elif self.inductive_inference is not None:
            model_cls = (
                models.InductiveNodePieceGNN
                if self.inductive_kwargs.get("use_GNN", "False") == "True"
                else models.InductiveNodePiece
            )
            kwargs = self.inductive_kwargs
            kwargs.pop("use_GNN", None)
            self.model = model_cls(
                triples_factory=prepare_triples(
                    self.edges_data,
                    relation_to_id=self.relation_to_id,
                    inv_triples=True,
                ),
                inference_factory=prepare_triples(
                    self.inductive_inference,
                    entity_to_id=self.entity_to_id,
                    relation_to_id=self.relation_to_id,
                    inv_triples=True,
                ),
                interaction=self.interaction,
                embedding_dim=self.embedding_dim,
                **kwargs,
            )
        else:
            assert self.literals_data is not None
            assert self.interaction is not None
            assert self.combination is not None
            combination_cls = self.combination.to_class(
                embedding_dim=self.embedding_dim,
                literal_shape=self.literals_data.shape[1],
                **(self.combination_kwargs or {}),
            )
            self.model = models.LiteralModel(
                triples_factory=prepare_triples(  # type: ignore[invalid-argument-type]
                    self.edges_data,
                    entity_to_id=self.entity_to_id,
                    relation_to_id=self.relation_to_id,
                    numeric_literals=self.literals_data,
                ),
                entity_representations_kwargs=dict(
                    shape=self.embedding_dim,
                ),
                relation_representations_kwargs=dict(
                    shape=self.embedding_dim,
                ),
                interaction=self.interaction,
                combination=combination_cls,
                loss=self.loss,
                random_seed=self.seed,
            )

        if self.trained and model_state is not None:
            buffer = io.BytesIO(model_state)
            self.model.load_state(buffer)  # ty: ignore[invalid-argument-type]
            if torch.cuda.is_available():
                self.model = self.model.to(torch.device("cuda"))

    def __repr__(self):
        return f"PyKEENModelWrapper({self.model.__class__.__name__})"

    def copy(self, deep: bool = True):
        import copy as _copy

        new_wrapper = PyKEENModelWrapper.__new__(PyKEENModelWrapper)
        new_wrapper.__dict__ = _copy.deepcopy(self.__dict__) if deep else self.__dict__.copy()
        return new_wrapper


def req_inverse_triples(model: models.Model | PyKEENModel1D | PyKEENModelMoreD) -> bool:
    """
    Check if the model requires inverse triples.
    """
    return isinstance(
        model,
        (models.CompGCN, models.NodePiece, models.InductiveNodePiece, models.InductiveNodePieceGNN),
    ) or model in {PyKEENModel1D.CompGCN, PyKEENModel1D.NodePiece}


class TrainingType(str, enum.Enum):
    sLCWA = "sLCWA"
    LCWA = "LCWA"

    def __str__(self):
        return self.value


class PyKEENSupportedLosses(str, enum.Enum):
    PointwiseLoss = "PointwiseLoss"
    DeltaPointwiseLoss = "DeltaPointwiseLoss"
    MarginPairwiseLoss = "MarginPairwiseLoss"
    PairwiseLoss = "PairwiseLoss"
    SetwiseLoss = "SetwiseLoss"
    AdversarialLoss = "AdversarialLoss"
    AdversarialBCEWithLogitsLoss = "AdversarialBCEWithLogitsLoss"
    BCEAfterSigmoidLoss = "BCEAfterSigmoidLoss"
    BCEWithLogitsLoss = "BCEWithLogitsLoss"
    CrossEntropyLoss = "CrossEntropyLoss"
    FocalLoss = "FocalLoss"
    InfoNCELoss = "InfoNCELoss"
    MarginRankingLoss = "MarginRankingLoss"
    MSELoss = "MSELoss"
    NSSALoss = "NSSALoss"
    SoftplusLoss = "SoftplusLoss"
    SoftPointwiseHingeLoss = "SoftPointwiseHingeLoss"
    PointwiseHingeLoss = "PointwiseHingeLoss"
    DoubleMarginLoss = "DoubleMarginLoss"
    SoftMarginRankingLoss = "SoftMarginRankingLoss"
    PairwiseLogisticLoss = "PairwiseLogisticLoss"

    def __str__(self):
        return self.value


@op("Define PyKEEN model", color="green", icon="file-3d")
def define_pykeen_model(
    bundle: core.Bundle,
    *,
    model: PyKEENModelMoreD = PyKEENModelMoreD.MuRE,
    edge_data_table: core.TableName = "edges",
    embedding_dim: int = 50,
    loss_function: PyKEENSupportedLosses = PyKEENSupportedLosses.NSSALoss,
    seed: int = 42,
    save_as: str = "PyKEENmodel",
):
    """Defines a PyKEEN model based on the selected model type."""
    bundle = bundle.copy()
    edges_data = bundle.dfs[edge_data_table][["head", "relation", "tail"]]
    triples_factory = prepare_triples(
        edges_data,
        inv_triples=req_inverse_triples(model),
    )

    model_class = model.to_class(
        triples_factory=triples_factory,
        loss_func=loss_function,
        embedding_dim=embedding_dim,
        seed=seed,
    )
    model_wrapper = PyKEENModelWrapper(
        model_class,
        loss=loss_function,
        model_type=model,
        embedding_dim=embedding_dim,
        entity_to_id=triples_factory.entity_to_id,
        relation_to_id=triples_factory.relation_to_id,
        edges_data=edges_data,
        seed=seed,
    )
    bundle.other[save_as] = model_wrapper
    return bundle


@op(
    "Define PyKEEN model with node attributes",
    color="green",
    icon="file-3d",
    params=[
        ops.ParameterGroup(
            name="combination_group",
            selector=ops.Parameter(
                name="combination_name",
                type=PyKEENCombinations,
                default=PyKEENCombinations.ConcatProjection,
            ),
            groups={
                "ComplexSeparated": [],
                # "Concat": [
                #     ops.Parameter.basic(name="dim", type=int, default=-1),
                # ],
                # "ConcatAggregation": [],
                "ConcatProjection": [
                    ops.Parameter.basic(name="bias", type=bool, default=False),
                    ops.Parameter.basic(name="dropout", type=float, default=0.0),
                    ops.Parameter.basic(name="activation", type=str, default="ReLU"),
                ],
                "Gated": [
                    ops.Parameter.basic(name="input_dropout", type=float, default=0.0),
                    ops.Parameter.basic(name="gate_activation", type=str, default="Sigmoid"),
                    ops.Parameter.basic(name="hidden_activation", type=str, default="Tanh"),
                ],
            },
            default=PyKEENCombinations.ConcatProjection,
        )
    ],
)
def def_pykeen_with_attributes(
    dataset: core.Bundle,
    *,
    interaction_name: PyKEENModel1D = PyKEENModel1D.TransE,
    combination_name: PyKEENCombinations = PyKEENCombinations.ConcatProjection,
    embedding_dim: int,
    loss_function: str,
    random_seed: int,
    save_as: str,
    **kwargs,
) -> core.Bundle:
    """Defines a PyKEEN model capable of using numeric literals as node attributes."""
    dataset = dataset.copy()

    edges_data = dataset.dfs["edges"][["head", "relation", "tail"]].astype(str)
    triples_no_literals = prepare_triples(
        edges_data,
    )
    temp_model = interaction_name.to_class(
        triples_factory=triples_no_literals,
        loss_func=loss_function,
        embedding_dim=embedding_dim,
        seed=random_seed,
    )

    num_literals = dataset.dfs["literals"]
    if "node_id" not in num_literals.columns:
        raise ValueError("Expected a 'node_id' column in literals DataFrame.")
    num_literals["node_id"] = num_literals["node_id"].astype(str)
    order = [
        label for label, _ in sorted(triples_no_literals.entity_to_id.items(), key=lambda kv: kv[1])
    ]
    num_literals = num_literals.set_index("node_id").reindex(order)

    if num_literals.isna().any().any():
        raise ValueError("Some entities are missing literals after reindexing.")

    features = num_literals.reset_index(drop=True)

    dataset.dfs["literals"] = num_literals
    literals_to_id = {label: i for i, label in enumerate(features.columns)}

    combination_cls = combination_name.to_class(embedding_dim, len(features.columns), **kwargs)

    assert isinstance(temp_model, models.ERModel), "Only models derived from ERModel are supported."
    try:
        interaction: Interaction = temp_model.interaction
    except AttributeError as e:
        raise Exception(
            "Interaction not supported for this model type. Please use a different interaction."
        ) from e

    model = LiteralModel(
        triples_factory=TriplesNumericLiteralsFactory(
            mapped_triples=triples_no_literals.mapped_triples,
            entity_to_id=triples_no_literals.entity_to_id,
            relation_to_id=triples_no_literals.relation_to_id,
            numeric_literals=torch.from_numpy(features.to_numpy())
            .contiguous()
            .detach()
            .cpu()
            .numpy(),
            literals_to_id=literals_to_id,
        ),
        entity_representations_kwargs=dict(
            shape=embedding_dim,
        ),
        relation_representations_kwargs=dict(
            shape=embedding_dim,
        ),
        interaction=interaction,  # ty: ignore[invalid-argument-type]
        combination=combination_cls,
        loss=loss_function,
        random_seed=random_seed,
    )

    model_wrapper = PyKEENModelWrapper(
        model=model,
        loss=loss_function,
        interaction=model.interaction,
        combination=combination_name,
        combination_kwargs=kwargs,
        literals_data=features,
        embedding_dim=embedding_dim,
        entity_to_id=triples_no_literals.entity_to_id,
        relation_to_id=triples_no_literals.relation_to_id,
        edges_data=edges_data,
        seed=random_seed,
    )

    dataset.other[save_as] = model_wrapper
    return dataset


class PyTorchAggregationFunctions(str, enum.Enum):
    MLP = "mlp"
    torch.sum
    torch.mean
    torch.amin
    torch.amax
    torch.prod
    torch.var
    torch.std


@op("Inductive setting", "Define inductive PyKEEN model", color="green", icon="file-3d")
def get_inductive_model(
    bundle: core.Bundle,
    *,
    triples_table: core.TableName,
    inference_table: core.TableName,
    interaction: PyKEENModel1D = PyKEENModel1D.DistMult,
    embedding_dim: int = 200,
    loss_function: str,
    num_tokens: int = 2,
    aggregation: PyTorchAggregationFunctions = PyTorchAggregationFunctions.MLP,
    use_GNN: bool = False,
    seed: int = 42,
    save_as: str = "InductiveModel",
):
    """
    Defines an InductiveNodePiece model (with an optional GNN message passing layer) for inductive link prediction tasks.

    Args:
        triples_table: The transductive edges of the graph.
        inference_table: The inductive edges of the graph.
        interaction: Type of interaction the model will use for link prediction scoring.
        num_tokens: Number of hash tokens for each node representation, usually 66th percentiles of the number of unique incident relations per node.
        aggregation: Aggregation of multiple token representations to a single entity representation. Pick a top-level torch function, or use 'mlp' for a two-layer built-in mlp aggregator.
    """
    bundle = bundle.copy()
    transductive_training = prepare_triples(
        bundle.dfs[triples_table][["head", "relation", "tail"]],
        inv_triples=True,
    )
    inductive_inference = prepare_triples(
        bundle.dfs[inference_table][["head", "relation", "tail"]],
        relation_to_id=transductive_training.relation_to_id,
        inv_triples=True,
    )
    model_cls = models.InductiveNodePieceGNN if use_GNN else models.InductiveNodePiece
    base_model_cls = interaction.to_class(transductive_training, loss_function, embedding_dim, 42)
    assert isinstance(base_model_cls, models.ERModel), "Base model class is not an ERModel"
    interaction_cls = base_model_cls.interaction

    model = model_cls(
        triples_factory=transductive_training,
        inference_factory=inductive_inference,
        loss=loss_function,
        interaction=interaction_cls,
        embedding_dim=embedding_dim,
        num_tokens=num_tokens,
        aggregation=aggregation,
        random_seed=seed,
    )

    model_wrapper = PyKEENModelWrapper(
        model=model,
        loss=loss_function,
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


class PyKEENSupportedOptimizers(str, enum.Enum):
    Adam = "Adam"
    AdamW = "AdamW"
    Adamax = "Adamax"
    Adagrad = "Adagrad"
    SGD = "SGD"


def prepare_triples(
    triples_df: pd.DataFrame,
    entity_to_id: typing.Optional[typing.Mapping[str, int]] = None,
    relation_to_id: typing.Optional[typing.Mapping[str, int]] = None,
    inv_triples: bool = False,
    numeric_literals: typing.Optional[pd.DataFrame] = None,
) -> TriplesFactory | TriplesNumericLiteralsFactory:
    """Prepare triples for PyKEEN from a DataFrame."""
    triples = TriplesFactory.from_labeled_triples(
        triples_df.astype(str).to_numpy(dtype=str),
        entity_to_id=entity_to_id,
        relation_to_id=relation_to_id,
        create_inverse_triples=inv_triples,
    )
    if numeric_literals is not None:
        return TriplesNumericLiteralsFactory(
            mapped_triples=triples.mapped_triples,
            entity_to_id=entity_to_id,
            relation_to_id=relation_to_id,
            numeric_literals=torch.from_numpy(numeric_literals.to_numpy())
            .contiguous()
            .detach()
            .cpu()
            .numpy(),
            literals_to_id={label: i for i, label in enumerate(numeric_literals.columns)},
        )
    return triples


@op("Train embedding model", slow=True, color="purple", icon="barbell-filled")
def train_embedding_model(
    bundle: core.Bundle,
    *,
    model: PyKEENModelName = "PyKEENmodel",
    training_table: core.TableName = "edges_train",
    testing_table: core.TableName = "edges_test",
    validation_table: core.TableName = "edges_val",
    optimizer_type: PyKEENSupportedOptimizers = PyKEENSupportedOptimizers.Adam,
    learning_rate: float = 0.0001,
    epochs: int = 5,
    training_approach: TrainingType = TrainingType.sLCWA,
    number_of_negative_samples_per_positive: int = 512,
):
    bundle_copy = bundle.copy()
    for key, value in bundle.dfs.items():
        bundle_copy.dfs[key] = value.copy(deep=True)

    model_wrapper: PyKEENModelWrapper = bundle_copy.other.get(model)
    bundle_copy.other[model] = model_wrapper.copy(deep=True)
    model_wrapper = bundle_copy.other[model]
    actual_model = model_wrapper.model
    sampler = None
    if isinstance(actual_model, models.RGCN) and training_approach == TrainingType.sLCWA:
        # Currently RGCN is the only model that requires a sampler and only when using sLCWA
        sampler = "schlichtkrull"

    entity_to_id = model_wrapper.entity_to_id
    relation_to_id = model_wrapper.relation_to_id

    training_set = prepare_triples(
        bundle_copy.dfs[training_table][["head", "relation", "tail"]],
        inv_triples=req_inverse_triples(actual_model),
        entity_to_id=entity_to_id,
        relation_to_id=relation_to_id,
        numeric_literals=model_wrapper.literals_data,
    )
    testing_set = prepare_triples(
        bundle_copy.dfs[testing_table][["head", "relation", "tail"]],
        inv_triples=req_inverse_triples(actual_model),
        entity_to_id=entity_to_id,
        relation_to_id=relation_to_id,
        numeric_literals=model_wrapper.literals_data,
    )
    validation_set = prepare_triples(
        bundle_copy.dfs[validation_table][["head", "relation", "tail"]],
        inv_triples=req_inverse_triples(actual_model),
        entity_to_id=entity_to_id,
        relation_to_id=relation_to_id,
        numeric_literals=model_wrapper.literals_data,
    )
    training_set, testing_set, validation_set = leakage.unleak(
        training_set, testing_set, validation_set
    )
    result: PipelineResult = pipeline(
        training=training_set,
        testing=testing_set,
        validation=validation_set,
        model=actual_model,
        loss=model_wrapper.loss,
        optimizer=optimizer_type,
        optimizer_kwargs=dict(
            lr=learning_rate,
        ),
        lr_scheduler="PolynomialLR",
        lr_scheduler_kwargs=dict(
            total_iters=epochs,
            power=0.95,
        ),
        training_loop=training_approach,
        negative_sampler="bernoulli" if training_approach == TrainingType.sLCWA else None,
        negative_sampler_kwargs=dict(
            num_negs_per_pos=number_of_negative_samples_per_positive,
        ),
        epochs=epochs,
        training_kwargs=dict(
            sampler=sampler,
            continue_training=model_wrapper.trained,
        ),
        stopper="early",
        stopper_kwargs=dict(
            frequency=5,
            patience=40,
            relative_delta=0.0005,
            metric="ah@k",
        ),
        random_seed=model_wrapper.seed,
    )

    model_wrapper.model = result.model
    model_wrapper.trained = True

    bundle_copy.dfs["training"] = pd.DataFrame({"training_loss": result.losses})
    if isinstance(result.stopper, stoppers.EarlyStopper):
        bundle_copy.dfs["early_stopper_metric"] = pd.DataFrame(
            {"early_stopper_metric": result.stopper.results}
        )
    bundle_copy.other[model] = model_wrapper

    return bundle_copy


@op("Inductive setting", "Train inductive model", slow=True, color="purple", icon="barbell-filled")
def train_inductive_pykeen_model(
    bundle: core.Bundle,
    *,
    model_name: PyKEENModelName,
    transductive_table_name: core.TableName,
    inductive_inference_table: core.TableName,
    inductive_validation_table: core.TableName,
    optimizer_type: PyKEENSupportedOptimizers = PyKEENSupportedOptimizers.Adam,
    epochs: int = 5,
    training_approach: TrainingType = TrainingType.sLCWA,
):
    bundle_copy = bundle.copy()
    for key, value in bundle.dfs.items():
        bundle_copy.dfs[key] = value.copy(deep=True)

    model_wrapper: PyKEENModelWrapper = bundle_copy.other.get(model_name)
    bundle_copy.other[model_name] = model_wrapper.copy(deep=True)
    model_wrapper = bundle_copy.other[model_name]

    model = model_wrapper.model
    transductive_training = prepare_triples(
        bundle_copy.dfs[transductive_table_name][["head", "relation", "tail"]],
        relation_to_id=model_wrapper.relation_to_id,
        inv_triples=True,
    )
    inductive_inference = prepare_triples(
        bundle_copy.dfs[inductive_inference_table][["head", "relation", "tail"]],
        entity_to_id=model_wrapper.entity_to_id,
        relation_to_id=model_wrapper.relation_to_id,
        inv_triples=True,
    )
    inductive_validation = prepare_triples(
        bundle_copy.dfs[inductive_validation_table][["head", "relation", "tail"]],
        entity_to_id=model_wrapper.entity_to_id,
        relation_to_id=model_wrapper.relation_to_id,
        inv_triples=True,
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
        triples_factory=transductive_training,
        model=model,
        optimizer=optimizer_type,
        mode="training",
        **loop_kwargs,
    )

    valid_evaluator = evaluation.SampledRankBasedEvaluator(
        mode="validation",
        evaluation_factory=inductive_validation,
        additional_filter_triples=inductive_inference.mapped_triples,
    )

    early_stopper = stoppers.EarlyStopper(
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

    losses = training_loop.train(
        triples_factory=transductive_training,
        stopper=early_stopper,
        num_epochs=epochs,
    )

    model_wrapper.trained = True

    bundle_copy.dfs["training"] = pd.DataFrame({"training_loss": losses})
    bundle_copy.dfs["early_stopper_metric"] = pd.DataFrame(
        {"early_stopper_metric": early_stopper.results}
    )
    bundle_copy.other[model_name] = model_wrapper

    return bundle_copy


@op("View early stopping metric", view="visualization", color="blue", icon="chart-line")
def view_early_stopping(bundle: core.Bundle):
    metric = bundle.dfs["early_stopper_metric"].early_stopper_metric.tolist()
    v = {
        "title": {"text": "Early Stopping Metric"},
        "xAxis": {"type": "category"},
        "yAxis": {"type": "value"},
        "series": [{"data": metric, "type": "line"}],
    }
    return v


@op("Triples prediction", color="yellow", icon="sparkles")
def triple_predict(
    bundle: core.Bundle,
    *,
    model_name: PyKEENModelName = "PyKEENmodel",
    table_name: core.TableName = "edges_val",
    inductive_setting: bool = False,
):
    bundle = bundle.copy()
    model: PyKEENModelWrapper = bundle.other.get(model_name)
    actual_model = model.model
    entity_to_id = model.entity_to_id
    relation_to_id = model.relation_to_id
    triples_to_predict_tf = prepare_triples(
        bundle.dfs[table_name][["head", "relation", "tail"]],
        entity_to_id=entity_to_id,
        relation_to_id=relation_to_id,
        inv_triples=req_inverse_triples(actual_model) or inductive_setting,
    )

    if inductive_setting and isinstance(actual_model, models.InductiveERModel):
        original_repr = actual_model._get_entity_representations_from_inductive_mode(
            mode="validation"
        )
        actual_model = actual_model.to(torch.device("cpu"))
        actual_model.replace_entity_representations_(
            mode="validation",
            representation=actual_model.create_entity_representation_for_new_triples(
                triples_to_predict_tf
            ),
        )
        if torch.cuda.is_available():
            actual_model = actual_model.to(torch.device("cuda"))

    pred_df = (
        predict_triples(
            model=actual_model,
            triples_factory=triples_to_predict_tf,
            mode="validation" if inductive_setting else None,
        )
        .process(
            factory=TriplesFactory(
                DUMMY_TRIPLET,
                entity_to_id=entity_to_id,
                relation_to_id=relation_to_id,
            )
        )
        .df[["head_label", "relation_label", "tail_label", "score"]]
    )
    bundle.dfs["pred"] = pred_df
    if inductive_setting and isinstance(actual_model, models.InductiveERModel):
        # Restore the original entity representations after prediction
        actual_model.replace_entity_representations_(
            mode="validation", representation=original_repr
        )
    return bundle


@op("Target prediction", color="yellow", icon="sparkles")
def target_predict(
    bundle: core.Bundle,
    *,
    model_name: PyKEENModelName = "PyKEENmodel",
    head: str,
    relation: str,
    tail: str,
    inductive_setting: bool = False,
):
    """
    Leave the target prediction field empty
    """
    bundle = bundle.copy()
    model_wrapper: PyKEENModelWrapper = bundle.other.get(model_name)
    entity_to_id = model_wrapper.entity_to_id
    relation_to_id = model_wrapper.relation_to_id
    pred = predict_target(
        model=model_wrapper,
        head=head if head != "" else None,
        relation=relation if relation != "" else None,
        tail=tail if tail != "" else None,
        triples_factory=TriplesFactory(
            DUMMY_TRIPLET,
            entity_to_id=entity_to_id,
            relation_to_id=relation_to_id,
        ),
        mode="validation" if inductive_setting else None,
    )

    col = "head_label" if head == "" else "tail_label" if tail == "" else "relation_label"
    df = pred.df[[col, "score"]]

    bundle.dfs["pred"] = df
    bundle.dfs["pred"].sort_values(by="score", ascending=False, inplace=True)
    return bundle


@op("Full prediction", slow=True, color="yellow", icon="sparkles")
def full_predict(
    bundle: core.Bundle,
    *,
    model_name: PyKEENModelName = "PyKEENmodel",
    k: int | None = None,
    inductive_setting: bool = False,
):
    """
    Warning: This prediction can be a very expensive operation!

    Args:
        k: Pass "" to keep all scores
    """
    bundle = bundle.copy()
    model_wrapper: PyKEENModelWrapper = bundle.other.get(model_name)
    entity_to_id = model_wrapper.entity_to_id
    relation_to_id = model_wrapper.relation_to_id
    pred = predict_all(
        model=model_wrapper, batch_size=None, k=k, mode="validation" if inductive_setting else None
    )
    pack = pred.process(
        factory=TriplesFactory(
            DUMMY_TRIPLET,
            entity_to_id=entity_to_id,
            relation_to_id=relation_to_id,
        ),
    )
    bundle.dfs["pred"] = pack.df[
        ["head_label", "relation_label", "tail_label", "score"]
    ].sort_values(by="score", ascending=False)

    return bundle


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


class EvaluatorTypes(str, enum.Enum):
    ClassificationEvaluator = "Classification Evaluator"
    MacroRankBasedEvaluator = "Macro Rank Based Evaluator"
    RankBasedEvaluator = "Rank Based Evaluator"
    SampledRankBasedEvaluator = "Sampled Rank Based Evaluator"

    def to_class(self) -> evaluation.Evaluator:
        return getattr(evaluation, self.name.replace(" ", ""))()


@op("Evaluate model", slow=True, color="orange", icon="microscope-filled")
def evaluate(
    bundle: core.Bundle,
    *,
    model_name: PyKEENModelName = "PyKEENmodel",
    evaluator_type: EvaluatorTypes = EvaluatorTypes.RankBasedEvaluator,
    eval_table: core.TableName = "edges_test",
    additional_true_triples_table: core.TableName = "edges_train",
    metrics_str: str = "ALL",
    batch_size: int = 32,
):
    """
    Evaluates the given model on the test set using the specified evaluator type.
    Args:
        evaluator_type: The type of evaluator to use. Note: When using classification based methods, evaluation may be extremely slow.
        metrics_str: Comma separated list, "ALL" if all metrics are needed.
    """

    bundle = bundle.copy()
    model_wrapper: PyKEENModelWrapper = bundle.other.get(model_name)
    entity_to_id = model_wrapper.entity_to_id
    relation_to_id = model_wrapper.relation_to_id
    evaluator = evaluator_type.to_class()
    if isinstance(evaluator, evaluation.ClassificationEvaluator):
        from pykeen.metrics.classification import classification_metric_resolver

        evaluator.metrics = tuple(
            classification_metric_resolver.make(metric_cls) for metric_cls in metrics_str.split(",")
        )
    testing_triples = prepare_triples(
        bundle.dfs[eval_table][["head", "relation", "tail"]],
        entity_to_id=entity_to_id,
        relation_to_id=relation_to_id,
    )
    additional_filters = prepare_triples(
        bundle.dfs[additional_true_triples_table][["head", "relation", "tail"]],
        entity_to_id=entity_to_id,
        relation_to_id=relation_to_id,
    )

    evaluated = evaluator.evaluate(
        model=model_wrapper.model,
        mapped_triples=testing_triples.mapped_triples,
        additional_filter_triples=additional_filters.mapped_triples,
        batch_size=batch_size,
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
            raise Exception(f"Possibly unknown metric: {metric}") from e
        metrics_df = pd.concat(
            [metrics_df, pd.DataFrame([[metric, score]], columns=metrics_df.columns)]
        )

    bundle.dfs["metrics"] = metrics_df

    return bundle


@op("Inductive setting", "Evaluate inductive model", color="orange", icon="microscope-filled")
def eval_inductive_model(
    bundle: core.Bundle,
    *,
    model_name: PyKEENModelName,
    inductive_testing_table: core.TableName,
    inductive_inference_table: core.TableName,
    inductive_validation_table: core.TableName,
    metrics_str: str = "ALL",
    batch_size: int = 32,
):
    bundle = bundle.copy()
    model_wrapper: PyKEENModelWrapper = bundle.other.get(model_name)
    inductive_testing = prepare_triples(
        bundle.dfs[inductive_testing_table][["head", "relation", "tail"]],
        entity_to_id=model_wrapper.entity_to_id,
        relation_to_id=model_wrapper.relation_to_id,
    )
    inductive_inference = prepare_triples(
        bundle.dfs[inductive_inference_table][["head", "relation", "tail"]],
        entity_to_id=model_wrapper.entity_to_id,
        relation_to_id=model_wrapper.relation_to_id,
    )
    inductive_validation = prepare_triples(
        bundle.dfs[inductive_validation_table][["head", "relation", "tail"]],
        entity_to_id=model_wrapper.entity_to_id,
        relation_to_id=model_wrapper.relation_to_id,
    )

    test_evaluator = evaluation.SampledRankBasedEvaluator(
        mode="testing",
        evaluation_factory=inductive_testing,
        additional_filter_triples=[
            inductive_inference.mapped_triples,
            inductive_validation.mapped_triples,
        ],
    )

    result = test_evaluator.evaluate(
        model=model_wrapper,
        mapped_triples=inductive_testing.mapped_triples,
        additional_filter_triples=[
            inductive_inference.mapped_triples,
            inductive_validation.mapped_triples,
        ],
        batch_size=batch_size,
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
