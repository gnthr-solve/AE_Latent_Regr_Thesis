# AE Latent Regr Thesis

This repository contains the code and documentation related to the thesis project focused on autoencoder latent regression. The project explores methodologies for analyzing latent representations using autoencoder frameworks.


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


## Detailed Subpackage Descriptions

### `loss` Package
- In `determ_terms.py` contains `KMeansLoss` inspired by [1].
- In `topology_term.py` classes adapted from [2]


## References

1. Kart-Leong Lim and Rahul Dutta (2021). *Prognostics and health management of wafer chemical-mechanical polishing system using autoencoder*. In 2021 IEEE International Conference on Prognostics and Health Management (ICPHM), pages 1–8.
2. Michael Moor, Max Horn, Bastian Rieck, and Karsten M. Borgwardt (2019). *Topological autoencoders*. CoRR, abs/1906.00722.
