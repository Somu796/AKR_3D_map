from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import ClassVar

import plotly.graph_objects as go  # type: ignore[import-untyped]
import numpy as np
import pandas as pd
import xarray as xr

from scripts.mixins.observation_time import ObservationTimeCalculator
from scripts.utils import (
    creates_bin1d, 
    add_celestial_bodies, 
    get_3d_layout_config, 
    save_plot)
from scripts.variables import (
    NumericType,
    PositiveNumber,
    n_coord_colnames,
    padding_grid,
)


@dataclass
class AKRGrid(ABC, ObservationTimeCalculator):
    """Abstract Base Class for all AKR coordinate systems."""

    # init=False means you don't do Cartesian(coord_colnames=...)
    coord_colnames: tuple[str, str, str] | None = field(init=False, default=None)
    grid: xr.Dataset | None = None
    N_DIMENSIONS: ClassVar[int] = n_coord_colnames

    # _____________________ ABSTRACT METHODS (TO CALL CHILD ATTRIBUTES) _____________________
    # Private methods to map coord_1/2/3 to x/y/z
    # Collects range attributes from child classes
    @abstractmethod
    def _get_range_attrs(self) -> tuple[str, str, str]:
        """
        Return attribute names for coordinate ranges.

        Example: ("x_range", "y_range", "z_range") or ("local_time_range", "radius_range", "mlat_range")
        """

    # Collects bin size attributes from child classes
    @abstractmethod
    def _get_bin_attrs(self) -> tuple[str, str, str]:
        """
        Return attribute names for coordinate bin sizes.

        Example: ("x_bin", "y_bin", "z_bin") or ("local_time_bin", "radius_bin", "mlat_bin")
        """

    # Public methods to get dimension names, units, and attributes
    # Collects dimension names from child classes useful for xarray Dataset
    @abstractmethod
    def get_dimension_names(self) -> tuple[str, str, str]:
        """
        Return dimension names for xarray Dataset.

        Example: ("x", "y", "z") or ("local_time", "radius", "mlat")
        """

    # Collects coordinate units from child classes useful for xarray Dataset
    @abstractmethod
    def get_coord_units(self) -> tuple[str, str, str]:
        """
        Return units for each coordinate.

        Example: ("R_E", "R_E", "R_E") or ("hours", "R_E", "degrees")
        """

    # Collects grid metadata attributes from child classes useful for xarray Dataset
    @abstractmethod
    def get_grid_attrs(self) -> dict:
        """Return grid-specific metadata attributes."""

    # Private method: in child classes transform coordinate system values (LT, R, MLat) to Cartesian (x, y, z)
    @abstractmethod
    def _transform_to_cartesian(
        self,
        coord_1_val: float,
        coord_2_val: float, 
        coord_3_val: float,
    ) -> tuple[float, float, float]:
        """
        Transform coordinate system values to Cartesian (X, Y, Z).
        
        Args:
            coord_1_val: First coordinate value
            coord_2_val: Second coordinate value
            coord_3_val: Third coordinate value
        
        Returns:
            (x, y, z) in Cartesian coordinates
        """

    # _____________________ATTRIBUTE MAPPING_____________________
    # Maps to child attributes
    # Ranges: X, Y, Z coordinate  or Local Time, Radius, MLat ranges etc.
    @property
    def coord_1_range(self) -> tuple[NumericType, NumericType]:
        """Get coordinate 1 range."""
        range_attr = self._get_range_attrs()[0]
        return getattr(self, range_attr)

    @coord_1_range.setter
    def coord_1_range(self, value: tuple[NumericType, NumericType]) -> None:
        """Set coordinate 1 range."""
        range_attr = self._get_range_attrs()[0]
        setattr(self, range_attr, value)

    @property
    def coord_2_range(self) -> tuple[NumericType, NumericType]:
        """Get coordinate 2 range."""
        range_attr = self._get_range_attrs()[1]
        return getattr(self, range_attr)

    @coord_2_range.setter
    def coord_2_range(self, value: tuple[NumericType, NumericType]) -> None:
        """Set coordinate 2 range."""
        range_attr = self._get_range_attrs()[1]
        setattr(self, range_attr, value)

    @property
    def coord_3_range(self) -> tuple[NumericType, NumericType]:
        """Get coordinate 3 range."""
        range_attr = self._get_range_attrs()[2]
        return getattr(self, range_attr)

    @coord_3_range.setter
    def coord_3_range(self, value: tuple[NumericType, NumericType]) -> None:
        """Set coordinate 3 range."""
        range_attr = self._get_range_attrs()[2]
        setattr(self, range_attr, value)

    # Bins: X, Y, Z coordinate  or Local Time, Radius, MLat ranges etc.
    @property
    def coord_1_bin(self) -> PositiveNumber:
        """Get coordinate 1 bin size."""
        bin_attr = self._get_bin_attrs()[0]
        return getattr(self, bin_attr)

    @coord_1_bin.setter
    def coord_1_bin(self, value: PositiveNumber) -> None:
        """Set coordinate 1 bin size."""
        bin_attr = self._get_bin_attrs()[0]
        setattr(self, bin_attr, value)

    @property
    def coord_2_bin(self) -> PositiveNumber:
        """Get coordinate 2 bin size."""
        bin_attr = self._get_bin_attrs()[1]
        return getattr(self, bin_attr)

    @coord_2_bin.setter
    def coord_2_bin(self, value: PositiveNumber) -> None:
        """Set coordinate 2 bin size."""
        bin_attr = self._get_bin_attrs()[1]
        setattr(self, bin_attr, value)

    @property
    def coord_3_bin(self) -> PositiveNumber:
        """Get coordinate 3 bin size."""
        bin_attr = self._get_bin_attrs()[2]
        return getattr(self, bin_attr)

    @coord_3_bin.setter
    def coord_3_bin(self, value: PositiveNumber) -> None:
        """Set coordinate 3 bin size."""
        bin_attr = self._get_bin_attrs()[2]
        setattr(self, bin_attr, value)

    # _____________________METHODS_____________________

    # Private method to validate coordinate column names
    def _validate_coord_colnames(  # Validates user given column names exists in the wind data dataframe
        self,
        df: pd.DataFrame,
        coord_colnames: tuple[str, str, str],
    ) -> None:
        """Validate coordinate column names."""
        # Check if it is a tuple and of size 3
        if (
            not isinstance(coord_colnames, tuple)
            or len(coord_colnames) != self.N_DIMENSIONS
        ):
            error_col_tuple_size = f"coord_colnames must be a tuple of {self.N_DIMENSIONS}, got {coord_colnames}"
            raise ValueError(
                error_col_tuple_size,
            )
        # Check if all columns maps to DataFrame
        missing = [col for col in coord_colnames if col not in df.columns]
        if missing:
            error_col_missing = (
                f"DataFrame is missing required coordinate columns: {missing}"
            )
            raise ValueError(error_col_missing)

    # Private method to validate that the xarray grid has been created
    def _validate_and_get_grid(
        self,
    ) -> xr.Dataset:
        """Validate that the grid has been created."""
        if self.grid is None:
            error_grid_not_created = (
                "Grid has not been created. Call create_grid() first."
            )
            raise ValueError(error_grid_not_created)
        return self.grid

    # Public method to decide grid boundaries from data
    def decide_boundaries(
        self,
        df: pd.DataFrame,
        coord_colnames: tuple[str, str, str],
        padding: float = padding_grid,  # Variable imported from scripts/variables.py
        *,
        verbose: bool = True,
    ) -> "AKRGrid":
        """
        Automatically determine grid boundaries from data.

        Args:
            df: DataFrame with coordinate columns
            coord_colnames: Column names in order (coord1, coord2, coord3)
            padding: Fraction to pad around data (default 0.01 = 1%)
            verbose: Whether to print the determined ranges

        Returns:
            self (for method chaining)

        """
        # Validate column names
        self._validate_coord_colnames(df, coord_colnames)

        # remember colnames
        self.coord_colnames = coord_colnames

        # Get dimension names (Dim names are always X, Y, Z even if user put x,Y,z) and units
        dim_names = self.get_dimension_names()
        units = self.get_coord_units()

        # Loop through each coordinate (X, Y, Z) to determine min/max and set ranges
        for i, col in enumerate(coord_colnames):
            # Calculate boundaries
            c_min, c_max = df[col].min(), df[col].max()
            width = c_max - c_min

            new_range = (
                c_min - padding * width,
                c_max + padding * width,
            )

            # Update generic coordinate range
            setattr(self, f"coord_{i + 1}_range", new_range)

            # Print info
            if verbose:
                print("Data range:\n")
                print(f"  {dim_names[i]} Data: {c_min:.2f} to {c_max:.2f} {units[i]}")
                print(f"Grid range (data range with {padding * 100:.0f}% padding):\n")
                print(
                    f"{dim_names[i]} Grid: {new_range[0]:.2f} to {new_range[1]:.2f} {units[i]}",
                )
        return self

    # Public method to create the grid based on either default, user given or decide_boundaries method call
    def create_grid(self) -> "AKRGrid":
        """
        Generic grid creation logic.

        Returns:
            self with xarray Dataset with coordinates (x, y, z) containing
            placeholder data variables for observation_time, burst_count, burst_time,
            and probability.

        """
        dim_names = self.get_dimension_names()
        units = self.get_coord_units()

        range_attrs = self._get_range_attrs()
        bin_attrs = self._get_bin_attrs()

        # Create bins for each dimension x, y, z or local_time, radius, mlat etc.
        all_edges = []
        all_centers = []

        for i in range(self.N_DIMENSIONS):
            coord_range = getattr(self, range_attrs[i])
            bin_size = getattr(self, bin_attrs[i])

            # Create bins
            edges, centers = creates_bin1d(
                float(coord_range[0]),
                float(coord_range[1]),
                float(bin_size),
            )
            all_edges.append(edges)
            all_centers.append(centers)

        # Create shape
        shape = tuple(len(centers) for centers in all_centers)

        # Initialize data variables (same for all coordinate systems)
        data_vars = {
            "observation_time": (
                list(dim_names),
                np.zeros(shape, dtype=np.float64),
                {"units": "seconds", "dtype": "float64"},
            ),
            "burst_count": (
                list(dim_names),
                np.zeros(shape, dtype=np.int32),
                {"units": "count", "dtype": "int32"},
            ),
            "residence_time": (
                list(dim_names),
                np.zeros(shape, dtype=np.float64),
                {"units": "seconds", "dtype": "float64"},
            ),
            "normalised_observation_time": (
                list(dim_names),
                np.zeros(shape, dtype=np.float64),
                {"units": "seconds", "dtype": "float64"},
            ),
        }

        # Create coordinates
        coords = {}
        for i, dim_name in enumerate(dim_names):
            coords[dim_name] = (
                [dim_name],
                all_centers[i],
                {"units": units[i], "dtype": "float64"},
            )
            coords[f"{dim_name}_edges"] = (
                [f"{dim_name}_edges"],
                all_edges[i],
                {"units": units[i], "dtype": "float64"},
            )

        # Create dataset
        self.grid = xr.Dataset(
            data_vars=data_vars,
            coords=coords,
            attrs=self.get_grid_attrs(),
        )

        return self

    def _assign_bin_indices(
        self,
        df: pd.DataFrame,
        coord_colnames: tuple[str, str, str],
    ) -> pd.DataFrame:
        """
        Assign each position to a grid bin.

        Args:
            df: DataFrame with coordinate columns
            coord_colnames: Column names for (coord1, coord2, coord3)

        Returns:
            DataFrame with new columns like 'bin_x', 'bin_y', 'bin_z' or
            'bin_local_time', 'bin_radius', 'bin_mlat'

        """
        # Validations
        grid = (
            self._validate_and_get_grid()
        )  # validate and return grid, type check safe
        self._validate_coord_colnames(df, coord_colnames)

        dim_names = self.get_dimension_names()

        # Process each dimension
        for _, (col, dim_name) in enumerate(
            zip(coord_colnames, dim_names, strict=True),
        ):
            # Get bin edges for this dimension
            edges = grid[f"{dim_name}_edges"].to_numpy()

            # Digitize
            bins = np.digitize(df[col].to_numpy(), edges) - 1

            # Mark out-of-bounds
            bins = np.where((bins >= 0) & (bins < len(edges) - 1), bins, -1)

            # Assign with dimension-specific name
            df[f"bin_{dim_name}"] = bins

        return df

    # Private method to plot the grid in Cartesian coordinates
    def _add_wireframe(self, fig: go.Figure, grid: xr.Dataset) -> None:
        """Add wireframe grid lines to 3D plot (works for any coordinate system)."""
        # Get grid and dimension info
        dim_names = self.get_dimension_names()
        bin_attrs = self._get_bin_attrs()
        
        # Get edges for all dimensions
        edges = [
            grid.coords[f"{dim_name}_edges"].to_numpy()
            for dim_name in dim_names
        ]
        
        # Determine step sizes for sampling (to avoid too many lines)
        steps = []
        for i, bin_attr in enumerate(bin_attrs):
            # bin_size = getattr(self, bin_attr)
            
            num_edges = len(edges[i])
            step = max(1, num_edges // 15)  # Aiming for around 15 lines per dimension
            steps.append(step)
        
        # Draw grid lines for each pair of constant coordinates
        # Lines along dimension 0 (vary dim 0, fix dim 1 & 2)
        for val_1 in edges[1][::steps[1]]:
            for val_2 in edges[2][::steps[2]]:
                x_line, y_line, z_line = [], [], []
                for val_0 in edges[0]:
                    x, y, z = self._transform_to_cartesian(val_0, val_1, val_2)
                    x_line.append(x)
                    y_line.append(y)
                    z_line.append(z)
                
                fig.add_trace(
                    go.Scatter3d(
                        x=x_line, y=y_line, z=z_line,
                        mode="lines",
                        line={"color": "gray", "width": 1},
                        opacity=0.3,
                        showlegend=False,
                        hoverinfo="skip",
                    ),
                )
        
        # Lines along dimension 1 (vary dim 1, fix dim 0 & 2)
        for val_0 in edges[0][::steps[0]]:
            for val_2 in edges[2][::steps[2]]:
                x_line, y_line, z_line = [], [], []
                for val_1 in edges[1]:
                    x, y, z = self._transform_to_cartesian(val_0, val_1, val_2)
                    x_line.append(x)
                    y_line.append(y)
                    z_line.append(z)
                
                fig.add_trace(
                    go.Scatter3d(
                        x=x_line, y=y_line, z=z_line,
                        mode="lines",
                        line={"color": "gray", "width": 1},
                        opacity=0.3,
                        showlegend=False,
                        hoverinfo="skip",
                    ),
                )
        
        # Lines along dimension 2 (vary dim 2, fix dim 0 & 1)
        for val_0 in edges[0][::steps[0]]:
            for val_1 in edges[1][::steps[1]]:
                x_line, y_line, z_line = [], [], []
                for val_2 in edges[2]:
                    x, y, z = self._transform_to_cartesian(val_0, val_1, val_2)
                    x_line.append(x)
                    y_line.append(y)
                    z_line.append(z)
                
                fig.add_trace(
                    go.Scatter3d(
                        x=x_line, y=y_line, z=z_line,
                        mode="lines",
                        line={"color": "gray", "width": 1},
                        opacity=0.3,
                        showlegend=False,
                        hoverinfo="skip",
                    ),
                )

    def plot_3d(
        self,
        variable: str | None = None,
        path: str = "3D_Objects/grid.html",
        *,
        show_earth: bool = True,
        show_sun: bool = False,
        ) -> "AKRGrid":
        """Plot 3D grid with wireframe (works for ALL coordinate systems)."""
        # 1. Validate grid
        grid = self._validate_and_get_grid()

        # 2. Initialize figure
        fig = go.Figure()
        
        # 3. Add wireframe: reads from grid, writes to fig
        self._add_wireframe(fig, grid)
        
        
        # 4. Add Data Layer (Only if a variable is specified and exists)
        if variable and variable in grid:

            # Retrieve dimension names 
            dim_names = self.get_dimension_names()

            data_array = grid[variable].to_numpy()
            ii, jj, kk = np.where(data_array > 0)

            # Retrieve units from xarray attributes, default to empty string if not found
            unit = grid[variable].attrs.get("units", "")
            unit_str = f" ({unit})" if unit else ""

            # Construct the clean name
            clean_name = variable.replace("_", " ").title()
            display_label = f"{clean_name}{unit_str}"

            if len(ii) > 0:
                fig.add_trace(
                    go.Scatter3d(
                        x=grid[dim_names[0]].to_numpy()[ii],
                        y=grid[dim_names[1]].to_numpy()[jj],
                        z=grid[dim_names[2]].to_numpy()[kk],
                        mode="markers",
                        marker={
                            "size": 5,
                            "color": data_array[ii, jj, kk],
                            "colorscale": "Viridis",
                            "colorbar": {
                                "title": display_label,
                                "thickness": 15,
                            },
                            "opacity": 0.8,
                            "showscale": True,
                        },
                        name=clean_name if variable else "Data",
                        showlegend=False,
                        hovertemplate="Value: %{marker.color:.2f}<extra></extra>",
                    ),
                )

            # Set title
            title = f"3D Grid: {clean_name}"
        else:
            title = "3D Grid Base"

        
        # 5. Add celestial bodies
        add_celestial_bodies(fig, show_earth=show_earth, show_sun=show_sun)

        # 6. Layout
        fig.update_layout(**get_3d_layout_config(title))
        
        # 7. Save plot
        save_plot(fig, path)

        self.fig = fig
        return self
    
