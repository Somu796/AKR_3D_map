import pandas as pd

# from scripts.grid_3d import Cartesian, LTRMLat


# %% observation_time calculations
def add_time_intervals(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add a column 'time_interval' showing how long spacecraft was at each position.

    Args:
        df: DataFrame with 'original_burst_id' and 'burst_timestamp' columns

    Returns:
        df: DataFrame with new 'time_interval' column (in seconds)

    Strategy:
        - Within each burst, calculate time to next position
        - For last position in each burst, use same interval as previous
          (or drop it, or use half interval - your choice!)

    """
    df = df.sort_values(by=["original_burst_id", "burst_timestamp"])

    # Get the NEXT timestamp
    next_time = (
        df.groupby("original_burst_id")[
            "burst_timestamp"
        ].shift(  # group burst timestamp by id
            -1,
        )  # get next timestamp within each group
    )

    # Time interval = next_time - current_time (naturally positive!)
    df["time_interval"] = (
        (
            next_time - df["burst_timestamp"]
        )  # Difference between next and current timestamp
        .dt.total_seconds()  # Convert to total seconds
    )

    # For last position in each burst, use previous interval
    df["time_interval"] = df.groupby("original_burst_id")[
        "time_interval"
    ].ffill()  # Fill NaNs with previous value # ? Check (the last point can have NaN interval?)

    return df
