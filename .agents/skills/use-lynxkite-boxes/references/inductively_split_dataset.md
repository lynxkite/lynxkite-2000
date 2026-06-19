**Split inductive dataset:**
Splits incoming data into 4 subsets. Transductive training on which training should be run, inductive inference on which during training inference is done.
Inference testing and validation sets that can be used to evaluate model performance.
parameters:
  - dataset_table: typing.Annotated[str, {'format': 'dropdown', 'metadata_query': '[].dataframes[].keys(@)[]'}] = ? --?
  - entity_ratio: <class 'float'> = 0.5 --How many percent of the entities in the dataset should be in the transductive training graph. If `0` semi-inductive split is applied, else fully-inductive split is applied
  - training_ratio: <class 'float'> = 0.8 --When semi-inductive this is *entity* ratio, when fully-inductive this is the inference training split
  - testing_ratio: <class 'float'> = 0.1 --When semi-inductive this is *entity* ratio, when fully-inductive this is the inference testing split
  - validation_ratio: <class 'float'> = 0.1 --When semi-inductive this is *entity* ratio, when fully-inductive this is the inference validation split
  - seed: <class 'int'> = 42 --?
  - bundle: <class 'lynxkite_graph_analytics.bundle.Bundle'> = ? --?

returns:
  - output: ? - ?.

usage:
output_variable = lynxkite_graph_analytics.operations.pykeen_ops.inductively_split_dataset(dataset_table=<dataset_table_value>, entity_ratio=<entity_ratio_value>, training_ratio=<training_ratio_value>, testing_ratio=<testing_ratio_value>, validation_ratio=<validation_ratio_value>, seed=<seed_value>, bundle=<bundle_variable>)
