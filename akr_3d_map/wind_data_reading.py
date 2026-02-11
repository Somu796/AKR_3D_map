from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


# %% This is for AKR Burst data reading
# Loading and applying schema to wind data from CSV
def load_apply_schema_wind_csv(path: str) -> pd.DataFrame:
    """
    Reading from csv file of Alexandra. Converting to proper datatypes.

    Args:
        path (str): Path to the CSV file.

    Returns:
        pd.DataFrame: DataFrame with applied schema.

    Example:
        >>> df = load_apply_schema_wind_csv('path/to/file.csv')
        >>> print(df.dtypes)

    """
    df = pd.read_csv(path)

    df[df.columns[0]] = pd.to_datetime(df[df.columns[0]])
    df[df.columns[1]] = pd.to_datetime(df[df.columns[1]])
    df[df.columns[2]] = (
        df.loc[:, df.columns[2]]
        .str.split(", ")
        .apply(pd.to_datetime, format="ISO8601", errors="coerce")
    )
    for i in range(3, 12, 1):
        df[df.columns[i]] = (
            df.loc[:, df.columns[i]]
            .str.split(", ")
            .apply(pd.to_numeric, errors="coerce")
        )

    return df


# Exploding and removing NaNs from wind data DataFrame
def exploding_saving_wind_data(
    df: pd.DataFrame,
    *,
    drop_na: bool = True,
) -> pd.DataFrame:
    """
    Explode the wind data DataFrame and remove rows with NaNs.

    Args:
        df (pd.DataFrame): Input DataFrame with list columns.
        drop_na (bool) : True or False. Whether to drop NA values.

    Returns:
        pd.DataFrame: Exploded DataFrame with NaNs removed.

    Example:
        >>> exploded_df = exploding_saving_wind_data(df)
        >>> print(exploded_df.head())

    """
    # 1. Identify columns with list data
    list_cols = df.columns[2 : len(df.columns)].tolist()

    # 2. Explode them simultaneously
    # This ensures that the 1st item of x_gse stays with the 1st timestamp, etc.
    df_exploded = df.explode(list_cols)
    # IMPORTANT: Reset the index so you can identify which points belonged together
    df_exploded = df_exploded.reset_index().rename(
        columns={"index": "original_burst_id"},
    )

    # 3. Remove rows where ANY of the data columns are NaN
    # We use the same list of columns to check for missing values
    if drop_na:
        df_exploded = df_exploded.dropna(subset=list_cols, how="any")
        print(f"Data cleaned. Remaining rows: {len(df_exploded)}")

    return df_exploded


# %% Residence Data Processing


def convert_to_timestamp(df: pd.DataFrame) -> pd.DataFrame:
    """Take residence data frame and convert its 3 colmns yyyy doy and hh:mm:ss to Timestamp."""
    # Use .assign to create columns without modifying the original slice immediately
    time_col = (
        pd.to_datetime(df["yyyy"], format="%Y")
        + pd.to_timedelta(df["doy"].astype(int) - 1, unit="D")
        + pd.to_timedelta(df["hh:mm:ss"])
    )

    to_remove = {"yyyy", "doy", "hh:mm:ss"}
    # Keep original order of remaining columns
    other_cols = [c for c in df.columns if c not in to_remove]

    # Return a fresh DataFrame with the new column at index 0
    return df.assign(time_stamp=time_col)[["time_stamp", *other_cols]]


def to_decimal_hours(time_val: pd.Series) -> float:
    """Takes in LT_gse in hh:mm:ss ad convert to float time."""
    if pd.isna(time_val) or str(time_val).lower() == "nan":
        return np.nan
    try:
        parts = str(time_val).strip().split(":")
        h, m, s = map(float, parts)
        return h + (m / 60.0) + (s / 3600.0)
    except (ValueError, AttributeError, IndexError):
        return np.nan


def vstack_residence_data(target_path_obj: "str", output_path: str) -> None:
    files = list(
        Path(target_path_obj).glob("wind*"),
    )  # Get specific files using pathlib

    df_list = []  # List to collect DataFrames

    for file_path in files:
        print(f"processing {file_path}")
        read_options: dict[str, Any] = {
            "filepath_or_buffer": file_path,
            "sep": r"\s+",
            "engine": "python",
        }

        # 1. Read and process
        df: pd.DataFrame = pd.read_csv(**read_options)
        df = convert_to_timestamp(df)

        # 2. Process LT (Local Time)
        # Using .copy() here is good practice to ensure we own the data
        df = df.copy()
        df["gseLT"] = df["gseLT"].map(to_decimal_hours)

        df_list.append(df)

    # 3. Concatenate all at once (Much faster!)
    if df_list:
        main_df = pd.concat(df_list, axis=0, ignore_index=True)

        # 4. Save to Parquet
        # Note: Parquet usually takes a string path, use .as_posix() for pathlib compatibility
        main_df.to_parquet(
            Path(output_path).as_posix(),
            engine="pyarrow",
            index=False,
            compression="snappy",
        )
    else:
        print("No files found to process.")
