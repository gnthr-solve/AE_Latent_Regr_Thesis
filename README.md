# AE Latent Regr Thesis

This repository contains the source code related to the thesis "Machine Learning for Process Control in Semiconductor -Manufacturing". Designed to conduct Chemical-Mechanical Planarisation (CMP) outcome prediction based on *partially* labelled input data.
Principal components include:
1. Data management tools and preprocessing routine.
2. Models and training routines for
    - Direct regression on labelled data (Linear model, DNN model),
    - Autoencoder models (deterministic, VAE, NVAE) for latent space regression.
3. Composed loss architecture for End-to-End training of composed models.
4. Hyperparameter optimisation routines for finding optimal models
5. Tools for evaluating model performance.

While the intended use is for CMP data, the majority of the code can run on any tabular dataset with associated metadata.
I.e. 
- `X_data` (input) of shape `(N, d)`, $N$ samples of dimension $d$.
- `y_data` (labels) of `(N, l)`, $N$ samples of dimension $l$, where any number can contain NaN values.
- A metadata column in both, linking the input samples to the labels.

Preprocessing routine transforms the data into `torch.Tensors`, saves a metadata csv and a mapping index json file.
In the tensor form samples are of shape `(N, d+1)`, linked by the first 'column' dimension, for example
```python
X_data_tensor[:, 0] == mapping_indices
```

## Requirements

- **Python:** Version 3.11.9 or later
- **Essential Python Libraries:**
    - torch
    - numpy
    - pandas
    - ray
    - optuna
    - scikit-learn
    - hydra-core
    - matplotlib
    - Additional libraries as specified in the `requirements.txt` file


## Project Structure

Below is an overview of the current repository structure:

| Directory/File | Description                                                              |
|  :--           |  :--  |
| `data/`        | Raw and processed datasets used for experiments and analysis.            |
| `data_utils/`  | Contains the dataset implementation and tools to instantiate, normalise and filter it.|
| `evaluation/`| Contains `Evaluation` class following the Visitor pattern, containers and visitors.     |
| `helper_tools/`| Various small helper tools and functions used in other subpackages.      |
| `loss/`     | Contains Composite pattern `LossTerm` ABCs and concrete implementations used as training losses.  |
| `hyperoptim/`   | Trainables, config classes, experiment configs and routine for hyperparameter opt. in `ray.tune`.|
| `models/`   | Source code for regression and autoencoder models.           |
| `observers/`   | Observer classes for supervising and evaluating training training routines. |
| `preprocessing/`   | Primary post-extraction preprocessing routine and normaliser classes. |
| `visualisation/`   | `PlotMatrix` design for creating composed plots during evaluation (Note: Implemented in research project) |
| `preprocess.py`    | File to execute preprocessing routine. |
| `main_train_tests.py`    | Primary file for individual model training and testing routines. |
| `hyperp_opt_run.py`    | Allows configuring and executing hyperparameter opt. routine using experiment configs. |
| `eval.py`    | File used to evaluate the best performing models found by hyperparameter opt. run and export/plot results. |
| `implementation_testing.py`    | File for executing small unit tests of components. |


## Excluded Files
Due to the confidentiality of both the data and the details of feature names and metadata, 
the following files are excluded from the repository:

- `data_utils/info.py` containing:
    - `process_metadata_cols`: List of columns of raw csv data containing wafer production metadata.
    - `file_metadata_cols`: List of columns that contain pathing and file metadata created and used in the extraction pipeline designed for Infineon.
    - `identifier_col`: Column name of the identifier, that links rows to processed wafer and its metadata.
    - `exclude_columns`: List of input features to be excluded by dataset builder.
    - `time_col`: Metadata column for the timestamp a wafer was processed at.
    - `y_col_rename_map`: Dictionary mapping to rename the y-data features.
    - `ts_rename_dict`: Dictionary mapping to rename the x-data features. 
    - `process_step_mapping`: Dictionary mapping the original process-step names to obfuscated versions.
- `data_utils/data_filters.py`: 
    - Contains callables mapping pandas DataFrames to boolean Series.
    - Used by `DatasetBuilder` as masks to create dataset instances filtered by metadata.
    - Currently only contains a closure for a by-machine filter.


## Usage
All primary execution files contain or import functions for experiment runs or tests and an execution section.
In files with various functions, the function to be executed is commented in and the others commented out, e.g.:
```python
if __name__=="__main__":
    #function_A()
    function_B() #active function
    #function_C()
```

*Note: The provided functionality is primarily utilized through interacting with and executing modules directly, and does not feature command-line customisation interfaces.*

The main files to be used for training, optimisation and evaluation are:
- `hyperp_opt_run.py`: Most important file, runs hyperparameter optimisation procedures, creates a log file, saves the trials results as a dataframe and the best performing model's state dict.
- `main_train_tests.py`: File for testing training routines for specific model combinations with manually selected parameters and loss compositions.
- `experiment_eval.py`: Loads hyperparameter optimisation results from a directory and uses `Evaluation`s to create plots or extract result statistics.



## Additional Subpackage Descriptions

### `evaluation` Package
Contains evaluation tools, organised using the Visitor pattern [3].
`Evaluation` stores all relevant components; accepts visitors operating on its attributes.

- `Evaluation`: Primary container class for model evaluations, accepts visitors that operate on it.
- `EvaluationResults`: Container storing losses, metrics and produced figures.
- `ModelOutput`: Container class for the outputs produced by trained a model, applied to the test data.
- `EvaluationVisitor`: Visitor base class and concrete implementations, for applying and evaluating models.
- `experiment_eval_funcs`: Functions to load and evaluate best models from hyperparameter optimisations.


### `models` Package
Contains architecture elements, regressors, both variational and deterministic encoder and decoder modules
and the composed models for deterministic autoencoders, VAE and NVAE.
*Note: contains unfinished transformer_ae elements, that could not be finished in the scope of the thesis.*


### `loss` Package
Contains `LossTerm` class hierarchy following the Composite pattern [3].
Single `LossTerm`s can be composed into a `CompositeLossTerm`, used to combine complex losses.
Members include the $L_p$ losses, log-likelihood and KL-Divergence terms.
Special members:
- In `determ_terms.py` contains `KMeansLoss` inspired by [1].
- In `topology_term.py` classes adapted from [2]


### `hyperoptim` Package
Hyperparameter optimisation implementations employing Ray Tune framework.
- `trainables`: Contains the trainable functions optimised by the tuner.
- `config.py`: Config dataclasses for hyperparameter optimisation experiments.
- `experiment_cfgs.py` Concrete configs - create or adapt for specific experiments.
- `callbacks.py`: Callbacks used in the optimisation routine. 
- `optim_routine.py`: Contains function that configures and employs the tuner for experiments


### `visualisation` Package
Initially developed as part of my former research project on Recovery Likelihood Estimation, this package
contains the `PlotMatrix` design and `AxesComponent` ABC and subclasses.
Every concrete `AxesComponent` is responsible for a specific type of plot, 
incorporated into the `PlotMatrix` and `PlotMosaic` classes.
The latter
- create and manage a figure and provide a set of maplotlib `Axes` placeholders
- incorporate concrete `AxesComponent` instances in a grid or mosaic layout respectively
- draw (and save) a composed figure consisting of the various components

The idea is to arrange multiple semantically connected plots into one figure.
It also contains some standalone plot functions in `general_plot_funcs.py`, that are used e.g. in `main_train_tests.py`.


## Additional Comments
The `training`, `observers`, `hydra_configs` and `hydra_sweeps.py` members contain tools that were developed along the way,
but either superceded or not used in the final outputs. 
They are hence not comprehensively documented and commented, contain experimental features and can be ignored.
The `visualisation` package was designed as part of my former research project and is utilised here, 
but also lacks documentation and is not crucial for the principle functionality.
It contains also experimental and untested `AxisComponent`s that might not function properly.


## References

1. Kart-Leong Lim and Rahul Dutta (2021). *Prognostics and health management of wafer chemical-mechanical polishing system using autoencoder*. In 2021 IEEE International Conference on Prognostics and Health Management (ICPHM), pages 1–8.
2. Michael Moor, Max Horn, Bastian Rieck, and Karsten M. Borgwardt (2019). *Topological autoencoders*. CoRR, abs/1906.00722.
3. Gamma, Erich and Helm, Richard and Johnson, Ralph and Vlissides, John M. (1994). *Design Patterns: Elements of Reusable Object-Oriented Software*. Addison-Wesley Professional