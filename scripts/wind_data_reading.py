import pandas as pd


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
def exploding_saving_wind_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Explode the wind data DataFrame and remove rows with NaNs.

    Args:
        df (pd.DataFrame): Input DataFrame with list columns.

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
    df_exploded = df_exploded.dropna(subset=list_cols, how="any")
    print(f"Data cleaned. Remaining rows: {len(df_exploded)}")

    return df_exploded
