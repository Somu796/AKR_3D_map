# %% Setting up Environment
# 1. Importing Libraries
import sys
from pathlib import Path

# 2. Setting project paths
project_root = Path.cwd().parents[0]
sys.path.append(str(project_root))

# Define standard data subdirectories for easy access later
RAW_DATA_DIR = project_root / "data" / "raw"
PROCESSED_DATA_DIR = project_root / "data" / "processed"
ASSETS_DIR = project_root / "assets" / "3D_Objects"

# %% Reading Data
import pandas as pd

wind_data = pd.read_parquet(
    f"{project_root}/data/processed/01_processed_wind_data_fogg_akr_burst_list_1995_2004.parquet",
)

residence_data = pd.read_parquet(f"{RAW_DATA_DIR}/residence_data.parquet")

# %% Preprocessing
cols_to_round = ["LT_gse", "radius", "lat_gse", "lon_gse", "x_gse", "y_gse", "z_gse"]

merged_df = (
    wind_data.assign(**{col: lambda df, c=col: df[c].round(2) for col in cols_to_round})
    .drop_duplicates(subset=cols_to_round, keep="first")
    .merge(
        # Round residence_data on-the-fly to ensure keys match
        residence_data.assign(
            **{col: lambda df, c=col: df[c].round(2) for col in cols_to_round},
        ),
        on=cols_to_round,
        how="outer",
    )
)
