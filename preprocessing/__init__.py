
"""
Preprocessing
-------------------------------------------------------------------------------------------------------------------------------------------
Two preprocessing categories.

I. Preprocessing the raw dataframes into organised pytorch tensors
    0. Load Raw Data
    1. Eliminate File Metadata and Rename
    2. Extract Column/Parameter Names for X, y
        2.1. Create Column Maps
        2.2. Export Column Maps
    3. Merge X and y DataFrames on Identifier & Export
        3.1. Eliminate NaN-Containing Rows
        3.2. Export Joint DataFrame
    4. Create Index Map and Export
    5. Separate and Export DataFrames
        5.1. Split Joint DataFrame
        5.2. Export DataFrames
    6. Convert to Tensors and Export


II. Preprocessing the data for use in the neural networks architecture
Normalisation, scaling, mapping to e.g. [0, 1]
"""

from .preprocess_raw import preprocess_raw