# File: grid/mixins/residence_time.py
# Description: Mixin for calculating Features of AKR in grid cells
# %% Imports
from typing import Self, cast

import numpy as np
import pandas as pd

# from scripts.grid_3d import Cartesian, LTRMLat
from akr_3d_map.variables import (
    burst_id_colname,
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

    def _add_time_intervals(
        self,
        df: pd.DataFrame,
        timestamp_colname: str,
        variable: str | None = None,
    ) -> pd.DataFrame:
        """
        Add a column 'time_interval' showing how long spacecraft was at each position.

        Args:
            df: DataFrame with 'original_burst_id' and 'burst_timestamp' columns
            variable: depending on for which variable the calculation differs
            timestamp_colname: date.time column to calculate the time interval
        Returns:
            DataFrame with new 'time_interval' column (in seconds)

        Strategy:
            - Within each burst, calculate time to next position
            - For last position in each burst, use same interval as previous

        """
        if variable == "residence_time":
            df = df.sort_values(by=[timestamp_colname])
            next_time = df[timestamp_colname].shift(-1)

        if variable == "observation_time":
            df = df.sort_values(by=[burst_id_colname, timestamp_colname])
            # Get the NEXT timestamp
            next_time = df.groupby(burst_id_colname)[timestamp_colname].shift(-1)

        # Time interval = next_time - current_time
        df[time_interval_colname] = (
            next_time - df[timestamp_colname]
        ).dt.total_seconds()

        # For last position in each burst, use previous interval
        if variable == "residence_time":
            df[time_interval_colname] = df[time_interval_colname].ffill()
        if variable == "observation_time":
            df[time_interval_colname] = df.groupby(burst_id_colname)[
                time_interval_colname
            ].ffill()

        return df

    def add_burst_count(
        self,
        df: pd.DataFrame,
        coord_colnames: tuple[str, str, str],
        burst_id_colname: str = burst_id_colname,
    ) -> Self:
        """
        Populate the grid with burst counts, considering only the first occurrence (onset) of each burst ID.

        Args:
            df: DataFrame with position and timestamp data
            coord_colnames: Column names for coordinates (coord1, coord2, coord3)
            burst_id_colname: Unique burst id identifier

        Returns:
            Self: for method chaining

        Example:
            >>> cart.add_burst_count(
            df=wind_data,
            coord_colnames=("x_gse", "y_gse", "z_gse"),
            )
            >>> burst_count_data = cart.grid.burst_count  # Access the populated grid

        """
        # 1. Validations
        # validate and return grid, type check safe
        grid = self._validate_and_get_grid()  # type: ignore[attr-defined]
        # validate coord colnames exists in given dataframe
        self._validate_coord_colnames(df, coord_colnames)  # type: ignore[attr-defined]

        # 2. Filter for first entry per burst
        df_first_entry = (
            df.sort_values([burst_id_colname, "burst_timestamp"])
            .groupby(burst_id_colname)
            .first()
            .reset_index()
        )

        # 2. Importing dimension for the specific child class
        dim_names = self.get_dimension_names()  # type: ignore[attr-defined]

        # 4. Assign bins using the filtered 'onset' data
        df_first_entry = self._assign_bin_indices(df_first_entry, coord_colnames)  # type: ignore[attr-defined]

        # 5. Filter only data within grid boundaries
        in_grid = (
            (df_first_entry[f"bin_{dim_names[0]}"] >= 0)
            & (df_first_entry[f"bin_{dim_names[1]}"] >= 0)
            & (df_first_entry[f"bin_{dim_names[2]}"] >= 0)
        )
        df_in_grid = df_first_entry[in_grid]

        # 6. Group by bin indices and count unique bursts
        # Since we only have one row per ID now, .size() or .nunique() works
        grouped = df_in_grid.groupby(
            [f"bin_{dim_names[0]}", f"bin_{dim_names[1]}", f"bin_{dim_names[2]}"],
        )[burst_id_colname].nunique()

        # 7. Update the internal xarray data directly
        obs_array: np.ndarray = grid.burst_count.data

        for iteration, (idx, n_bursts) in enumerate(grouped.items()):
            i, j, k = cast("tuple[int, int, int]", idx)
            obs_array[int(i), int(j), int(k)] += n_bursts

            if iteration % 500 == 0:
                print(f"Update in progress... processed {iteration} bins.")

        print(f"Grid populated: {np.count_nonzero(obs_array)} bins updated.")
        return self  # type: ignore[return-value]

    def add_observation_count(
        self,
        df: pd.DataFrame,
        coord_colnames: tuple[str, str, str],
    ) -> Self:
        """Populate the grid by adding 1 for every row in the dataframe to its corresponding 3D bin."""
        # 1. Validations
        grid = self._validate_and_get_grid()  # type: ignore[attr-defined]
        self._validate_coord_colnames(df, coord_colnames)  # type: ignore[attr-defined]
        dim_names = self.get_dimension_names()  # type: ignore[attr-defined]

        # 2. Assign bins to EVERY row (No grouping or filtering here)
        df_binned = self._assign_bin_indices(df, coord_colnames)  # type: ignore[attr-defined]

        # 3. Filter only data within grid boundaries
        bin_cols = [f"bin_{dim_names[0]}", f"bin_{dim_names[1]}", f"bin_{dim_names[2]}"]
        in_grid = (df_binned[bin_cols] >= 0).all(axis=1)
        df_in_grid = df_binned[in_grid]

        # 4. Group by bin indices and count the number of rows (.size())
        # Each row in the group represents +1 for that bin
        grouped = df_in_grid.groupby(bin_cols).size()

        # 5. Update the internal xarray data directly
        # Ensure 'observation_count' exists in your grid data_vars
        obs_array: np.ndarray = grid.observation_count.data

        for iteration, (idx, count) in enumerate(grouped.items()):
            # idx is (i, j, k)
            i, j, k = map(int, cast(tuple, idx))

            # Add the count (number of rows) to the existing value in the bin
            obs_array[i, j, k] += int(count)

            if iteration % 1000 == 0:
                print(f"Update in progress... processed {iteration} active bins.")

        print(f"Grid populated: {np.count_nonzero(obs_array)} bins updated.")
        return self  # type: ignore[return-value]

    def add_residence_count(
        self,
        df: pd.DataFrame,
        coord_colnames: tuple[str, str, str],
        residence_timestamp_colname: str = "time_stamp",
        gap_hours: int = 2,
    ) -> Self:
        """
        Populates 'residence_count' in the grid by identifying unique spacecraft passes based on time gaps in the residence data.

        Args:
            df: residence time dataframe
            coord_colnames: Column names for coordinates (coord1, coord2, coord3)
            residence_timestamp_colname: Timestamp column for residence data
            gap_hours: Hours threshold to distinguish separate spacecraft passes when auto-populating residence time
        Returns:
            Self: for method chaining

        """
        # 1. Validations
        grid = self._validate_and_get_grid()  # type: ignore[attr-defined]
        self._validate_coord_colnames(df, coord_colnames)  # type: ignore[attr-defined]
        dim_names = self.get_dimension_names()  # type: ignore[attr-defined]

        # 2. Setup Pass ID (The "Once or Twice" Logic)
        # We must sort to detect gaps between consecutive chronological points
        df = df.sort_values(residence_timestamp_colname).copy()
        df[residence_timestamp_colname] = pd.to_datetime(
            df[residence_timestamp_colname],
        )

        # Mark a new pass if the gap between points is larger than threshold
        df["new_pass"] = df[residence_timestamp_colname].diff() > pd.Timedelta(
            hours=gap_hours,
        )
        df["pass_id"] = df["new_pass"].cumsum()

        # 3. Assign Bins
        df = self._assign_bin_indices(df, coord_colnames)  # type: ignore[attr-defined]

        # 4. Filter for In-Grid Data
        bin_cols = [f"bin_{name}" for name in dim_names]
        in_grid = (df[bin_cols] >= 0).all(axis=1)
        df_in_grid = df[in_grid]

        # 5. Group by bins and count UNIQUE pass_ids
        grouped = df_in_grid.groupby(bin_cols)["pass_id"].nunique()

        # 6. Update the Xarray Data Array
        res_count_array: np.ndarray = grid.residence_count.data

        for iteration, (idx, n_passes) in enumerate(grouped.items()):
            # idx is (bin_0, bin_1, bin_2)
            i, j, k = map(int, cast(tuple, idx))
            res_count_array[i, j, k] += int(n_passes)

            if iteration % 1000 == 0:
                print(f"Residence Update: Processed {iteration} bins...")

        print(
            f"Success: {np.count_nonzero(res_count_array)} bins updated with pass counts."
        )
        return self  # type: ignore[return-value]  # type: ignore[return-value]

    def add_observation_time(
        self,
        df: pd.DataFrame,
        coord_colnames: tuple[str, str, str],
        timestamp_colname: str = "burst_timestamp",
    ) -> Self:
        """
        Calculate time intervals and populate the grid with observation time.

        Args:
            df: DataFrame with position and timestamp data
            coord_colnames: Column names for coordinates (coord1, coord2, coord3)
            timestamp_colname: which datetime column to use to calculate the time interval

        Returns:
            Self: for method chaining

        Example:
            >>> cart.add_observation_time(
            df=spacecraft_data,
            coord_colnames=("x_gse", "y_gse", "z_gse"),
            )
            >>> observation_time = cart.grid.observation_time  # Access the populated grid

        """
        # 1. Validations
        # validate and return grid, type check safe
        grid = self._validate_and_get_grid()  # type: ignore[attr-defined]
        # validate coord colnames exists in given dataframe
        self._validate_coord_colnames(df, coord_colnames)  # type: ignore[attr-defined]

        # 2. Importing dimension for the specific child class
        dim_names = self.get_dimension_names()  # type: ignore[attr-defined]

        # 3. Add time intervals
        df = self._add_time_intervals(
            df,
            timestamp_colname=timestamp_colname,
            variable="observation_time",
        )

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
        )[time_interval_colname].sum()

        # 7. Update the internal xarray data directly
        obs_array: np.ndarray = grid.observation_time.data

        for iteration, (idx, total_time) in enumerate(grouped.items()):
            i, j, k = cast("tuple[int, int, int]", idx)
            obs_array[int(i), int(j), int(k)] += total_time

            if iteration % 500 == 0:
                print(f"Update in progress... processed {iteration} bins.")

        print(f"Grid populated: {np.count_nonzero(obs_array)} bins updated.")

        return self  # type: ignore[return-value]

    def add_residence_time(
        self,
        df: pd.DataFrame,
        coord_colnames: tuple[str, str, str],
        timestamp_colname: str = "time_stamp",
    ) -> Self:
        """
        Populate the grid with residence time (total seconds spent per bin).

        Args:
            df: DataFrame with position and interval data
            coord_colnames: Column names for coordinates (coord1, coord2, coord3)
            timestamp_colname: which datetime column to use to calculate the time interval

        Returns:
            Self: for method chaining

        """
        # 1. Validations
        grid = self._validate_and_get_grid()  # type: ignore[attr-defined]
        self._validate_coord_colnames(df, coord_colnames)  # type: ignore[attr-defined]

        # 2. Identify dimension names
        dim_names = self.get_dimension_names()  # type: ignore[attr-defined]

        # 3. Ensure time intervals exist (Calculation logic)
        if time_interval_colname not in df.columns:
            df = self._add_time_intervals(
                df,
                timestamp_colname=timestamp_colname,
                variable="residence_time",
            )

        # 4. Assign bins based on coordinates
        df = self._assign_bin_indices(df, coord_colnames)  # type: ignore[attr-defined]

        # 5. Filter for data within grid boundaries
        in_grid = (
            (df[f"bin_{dim_names[0]}"] >= 0)
            & (df[f"bin_{dim_names[1]}"] >= 0)
            & (df[f"bin_{dim_names[2]}"] >= 0)
        )
        df_in_grid = df[in_grid]

        # 6. Group by bin indices and SUM the time intervals
        # This gives total residence time per 3D cell
        grouped = df_in_grid.groupby(
            [f"bin_{dim_names[0]}", f"bin_{dim_names[1]}", f"bin_{dim_names[2]}"],
        )[time_interval_colname].sum()

        # 7. Update the internal xarray data (residence_time)
        # Assuming your grid object has a .residence_time data variable
        res_array: np.ndarray = grid.residence_time.data

        for iteration, (idx, total_residence) in enumerate(grouped.items()):
            i, j, k = cast("tuple[int, int, int]", idx)
            res_array[int(i), int(j), int(k)] += total_residence

            if iteration % 500 == 0:
                print(f"Residence Update: processed {iteration} bins.")

        print(
            f"Grid populated: {np.count_nonzero(res_array)} bins updated with residence time.",
        )

        return self  # type: ignore[return-value]

    def add_normalised_observation_time(
        self,
        akr_df: pd.DataFrame,
        satellite_residence_df: pd.DataFrame,
        coord_colnames: tuple[str, str, str],
        akr_timestamp_colname: str = "burst_timestamp",
        residence_timestamp_colname: str = "time_stamp",
    ) -> Self:
        """
        Populate the grid with normalised observation time (Observation time / Residence time).

        This accounts for sampling bias, a cell might have high AKR observation
        time simply because the satellite spent a lot of time there. Normalisation
        reveals the true activity rate.

        Args:
            akr_df: DataFrame with AKR burst events.
            satellite_residence_df: DataFrame with full spacecraft trajectory.
            coord_colnames: Column names for coordinates (coord1, coord2, coord3).
            akr_timestamp_colname: Timestamp column for AKR data.
            residence_timestamp_colname: Timestamp column for residence data.

        Returns:
            Self: for method chaining

        """
        # 1. Validations
        grid = self._validate_and_get_grid()  # type: ignore[attr-defined]

        # 2. Ensure Denominator (Residence Time) is populated
        # We check if the residence_time array is effectively empty (all zeros)
        if np.count_nonzero(grid.residence_time.data) == 0:
            print(
                "Residence time grid empty. Populating from satellite_residence_df...",
            )
            self.add_residence_time(
                df=satellite_residence_df,
                coord_colnames=coord_colnames,
                timestamp_colname=residence_timestamp_colname,
            )

        # 3. Ensure Numerator (Observation Time) is populated
        if np.count_nonzero(grid.observation_time.data) == 0:
            print("Observation time grid empty. Populating from akr_df...")
            self.add_observation_time(
                df=akr_df,
                coord_colnames=coord_colnames,
                timestamp_colname=akr_timestamp_colname,
            )

        # 4. Perform Normalisation
        # Strategy: Use np.divide with 'where' to avoid DivisionByZero errors
        # in cells where the satellite never visited.

        num = grid.observation_time.data
        den = grid.residence_time.data

        # normalized_obs_time = (Total AKR Time in bin) / (Total Spacecraft Time in bin)
        norm_array = np.divide(
            num,
            den,
            out=np.zeros_like(num, dtype=float),
            where=den != 0,
        )

        # 5. Update the internal xarray data directly
        # Assuming your grid object has a .normalised_observation_time data variable
        grid.normalised_observation_time.data = norm_array

        # Apply a physical constraint (normalised time cannot exceed 1.0)
        grid.normalised_observation_time.data = np.clip(
            grid.normalised_observation_time.data,
            0,
            1,
        )

        print(
            f"Normalistion complete. Activity rates calculated for "
            f"{np.count_nonzero(norm_array)} active grid cells.",
        )

        return self  # type: ignore[return-value]


# %% Previous logic
# def add_burst_count(
#     self,
#     df: pd.DataFrame,
#     coord_colnames: tuple[str, str, str],
# ) -> Self:
#     """
#     Same as burst count, only logic different is this it adds up 1 for all the grid not only first grid.

#     Args:
#         df: DataFrame with position and timestamp data
#         coord_colnames: Column names for coordinates (coord1, coord2, coord3)

#     Returns:
#         Self: for method chaining

#     Example:
#         >>> cart.add_burst_count(
#         df=wind_data,
#         coord_colnames=("x_gse", "y_gse", "z_gse"),
#         )
#         >>> burst_count_data = cart.grid.burst_count  # Access the populated grid

#     """
#     # 1. Validations
#     # validate and return grid, type check safe
#     grid = self._validate_and_get_grid()  # type: ignore[attr-defined]
#     # validate coord colnames exists in given dataframe
#     self._validate_coord_colnames(df, coord_colnames)  # type: ignore[attr-defined]

#     # 2. Importing dimension for the specific child class
#     dim_names = self.get_dimension_names()  # type: ignore[attr-defined]

#     # 4. Assign bins
#     df = self._assign_bin_indices(df, coord_colnames)  # type: ignore[attr-defined]

#     # 5. Filter only data within grid boundaries
#     in_grid = (
#         (df[f"bin_{dim_names[0]}"] >= 0)
#         & (df[f"bin_{dim_names[1]}"] >= 0)
#         & (df[f"bin_{dim_names[2]}"] >= 0)
#     )
#     df_in_grid = df[in_grid]

#     # 6. Group by bin indices and sum intervals
#     grouped = df_in_grid.groupby(
#         [f"bin_{dim_names[0]}", f"bin_{dim_names[1]}", f"bin_{dim_names[2]}"],
#     )[burst_id_colname].nunique()

#     # 7. Update the internal xarray data directly
#     obs_array: np.ndarray = grid.observation_count.data

#     for iteration, (idx, n_bursts) in enumerate(grouped.items()):
#         i, j, k = cast("tuple[int, int, int]", idx)
#         obs_array[int(i), int(j), int(k)] += n_bursts

#         if iteration % 500 == 0:
#             print(f"Update in progress... processed {iteration} bins.")

#     print(f"Grid populated: {np.count_nonzero(obs_array)} bins updated.")

#     return self  # type: ignore[return-value]
