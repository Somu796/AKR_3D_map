# File: grid/mixins/residence_time.py
# Description: Mixin for calculating Features of AKR in grid cells
# %% Imports
from typing import Self, cast

import numpy as np
import pandas as pd

# from scripts.grid_3d import Cartesian, LTRMLat
from scripts.variables import (
    burst_id_colname,
    burst_timestamp_colname,
    time_interval_colname,
)

# %% Observation time mixin class


class ObservationTimeCalculator:
    """
    Mixin for calculating observation time in grid cells.

    Requires the class to have:
    - _validate_and_get_grid() method
    - _validate_coord_colnames() method
    - _assign_bin_indices() method
    - get_dimension_names() method
    """

    def _add_time_intervals(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Add a column 'time_interval' showing how long spacecraft was at each position.

        Args:
            df: DataFrame with 'original_burst_id' and 'burst_timestamp' columns

        Returns:
            DataFrame with new 'time_interval' column (in seconds)

        Strategy:
            - Within each burst, calculate time to next position
            - For last position in each burst, use same interval as previous

        """
        df = df.sort_values(by=[burst_id_colname, burst_timestamp_colname])

        # Get the NEXT timestamp
        next_time = df.groupby(burst_id_colname)[burst_timestamp_colname].shift(-1)

        # Time interval = next_time - current_time
        df[time_interval_colname] = (
            next_time - df[burst_timestamp_colname]
        ).dt.total_seconds()

        # For last position in each burst, use previous interval
        df[time_interval_colname] = df.groupby(burst_id_colname)[
            time_interval_colname
        ].ffill()

        return df

    def add_observation_time(
        self,
        df: pd.DataFrame,
        coord_colnames: tuple[str, str, str],
    ) -> Self:
        """
        Calculate time intervals and populate the grid with observation time.

        Args:
            df: DataFrame with position and timestamp data
            coord_colnames: Column names for coordinates (coord1, coord2, coord3)

        Returns:
            self (for method chaining)

        Example:
            >> cart.add_observation_time(
                df=spacecraft_data,
                coord_colnames=("x_gse", "y_gse", "z_gse"),
            )
            >> observation_time = cart.grid.observation_time  # Access the populated grid

        """
        # 1. Validations
        # validate and return grid, type check safe
        grid = self._validate_and_get_grid()  # type: ignore[attr-defined]
        # validate coord colnames exists in given dataframe
        self._validate_coord_colnames(df, coord_colnames)  # type: ignore[attr-defined]

        # 2. Importing dimension for the specific child class
        dim_names = self.get_dimension_names()  # type: ignore[attr-defined]

        # 3. Add time intervals
        df = self._add_time_intervals(df)

        # 4. Assign bins
        df = self._assign_bin_indices(df, coord_colnames)  # type: ignore[attr-defined]

        # 5. Filter only data within grid boundaries
        in_grid = (
            (df[f"bin_{dim_names[0]}"] >= 0)
            & (df[f"bin_{dim_names[1]}"] >= 0)
            & (df[f"bin_{dim_names[2]}"] >= 0)
        )
        df_in_grid = df[in_grid]

        # 6. Group by bin indices and sum intervals
        grouped = df_in_grid.groupby(
            [f"bin_{dim_names[0]}", f"bin_{dim_names[1]}", f"bin_{dim_names[2]}"],
        )["time_interval"].sum()

        # 7. Update the internal xarray data directly
        obs_array: np.ndarray = grid.observation_time.data

        for iteration, (idx, total_time) in enumerate(grouped.items()):
            i, j, k = cast("tuple[int, int, int]", idx)
            obs_array[int(i), int(j), int(k)] += total_time

            if iteration % 500 == 0:
                print(f"Update in progress... processed {iteration} bins.")

        print(f"Grid populated: {np.count_nonzero(obs_array)} bins updated.")

        return self  # type: ignore[return-value]
