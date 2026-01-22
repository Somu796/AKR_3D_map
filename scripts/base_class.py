from abc import ABC, abstractmethod
from dataclasses import dataclass, field

import numpy as np
import pandas as pd
import xarray as xr

from scripts.variables import n_coord_colnames


@dataclass
class AKRGrid(ABC):
    """Abstract Base Class for all AKR coordinate systems."""

    # Placeholders for any coordinate system
    # coord_names: tuple[str, str, str] = ("c1", "c2", "c3")
    # ranges: dict[str, tuple[float, float]] = field(default_factory=dict)
    # grid: xr.Dataset | None = None
    @abstractmethod
    def _get_attribute_names_for_ranges(self) -> list[str]:
        """Each child MUST provide its own list of names."""
        pass

    def _validate_coord_colnames(
        self, df: pd.DataFrame, coord_colnames: tuple[str, str, str]
    ) -> None:
        """Rule checking: Tuple of 3 and columns must exist."""
        if (
            not isinstance(coord_colnames, tuple)
            or len(coord_colnames) != n_coord_colnames
        ):
            error_tuple_size = f"data_cols must be a tuple of 3, got {coord_colnames}"
            raise ValueError(error_tuple_size)

        for col in coord_colnames:
            if col not in df.columns:
                error_col_not_exist = f"Column {col} not found in input data."
                raise ValueError(error_col_not_exist)

    def decide_boundaries(
        self,
        df: pd.DataFrame,
        coord_colnames: tuple[str, str, str],  # e.g., ("x_gse", "y_gse", "z_gse")
        padding: float = 0.01,
    ) -> "AKRGrid":
        """
        Automatically determine grid boundaries from data.

        Args:
            df: DataFrame with (x_gse, y_gse, z_gse) or, (local time, radius, magnetic latitude) columns
            coord_colnames: Names of the columns to use for coordinates in tuple, keep (x_gse, y_gse, z_gse) or, (local time, radius, magnetic latitude) order
            padding: Fraction to pad around data (default 0.1 = 10%)

        Returns:
            self (for method chaining)

        Example:
            >>> cart = Cartesian(bin_size=2.0).decide_boundaries(df, coord_colnames=(x_gse, y_gse, z_gse), padding=0.01)
            >>> print(cart.x_range)
            (-15.2, 15.8)  # Auto-determined from data

        """
        # Validate column names
        self._validate_coord_colnames(df, coord_colnames)

        # 2. Get the target variable names from the Child (e.g. ["x_range", ...])
        attribute_names_to_update = self._get_attribute_names_for_ranges()

        # 3. Calculating minimum and maximum boundary in the coordinate data
        for i, col in enumerate(coord_colnames):
            c_min, c_max = df[col].min(), df[col].max()
            width = c_max - c_min

            new_range = (
                c_min - padding * width,
                c_max + padding * width,
            )

            var_name: str = attribute_names_to_update[i]

            # 4. Target Mapping: This updates self.x_range, self.y_range, etc.
            setattr(self, var_name, new_range)

        return self

    def assign_bin_indices(
        self,
        df: pd.DataFrame,
    ) -> pd.DataFrame:
        """

        Assign each position to a Cartesian grid bin.

        Args:
            df: DataFrame with x_gse, y_gse, z_gse columns
            grid: Cartesian grid object

        Returns:
            df: DataFrame with new columns 'bin_x', 'bin_y', 'bin_z'

        """
        if self.grid is None:
            raise ValueError(
                "Grid not initialized. Run create_grid() before assigning bins.",
            )

        # Get bin edges
        x_edges = self.grid.x_edges.to_numpy()
        y_edges = self.grid.y_edges.to_numpy()
        z_edges = self.grid.z_edges.to_numpy()

        # Digitize
        bin_x = np.digitize(df["x_gse"].to_numpy(), x_edges) - 1
        bin_y = np.digitize(df["y_gse"].to_numpy(), y_edges) - 1
        bin_z = np.digitize(df["z_gse"].to_numpy(), z_edges) - 1

        # Mark out-of-bounds
        bin_x = np.where((bin_x >= 0) & (bin_x < len(x_edges) - 1), bin_x, -1)
        bin_y = np.where((bin_y >= 0) & (bin_y < len(y_edges) - 1), bin_y, -1)
        bin_z = np.where((bin_z >= 0) & (bin_z < len(z_edges) - 1), bin_z, -1)

        df["bin_x"] = bin_x
        df["bin_y"] = bin_y
        df["bin_z"] = bin_z

        return df

    # def _prepare_empty_dataset(self, coords, dims, attrs) -> None:

    #     """Standardized AKR data structure creator."""
    #     shape = tuple(len(coords[d]) for d in dims)

    #     # Define shared AKR variables
    #     data_vars = {
    #         "residence_time": (dims, np.zeros(shape), {"units": "seconds"}),
    #         "burst_count": (dims, np.zeros(shape, dtype=int), {"units": "count"}),
    #         "probability": (dims, np.zeros(shape), {"units": "percent"}),
    #     }

    #     self.grid = xr.Dataset(data_vars=data_vars, coords=coords, attrs=attrs)
    #     return self

    # @abstractmethod
    # def create_grid(self) -> None:
    #     """Must be implemented by child classes."""
    #     pass
