from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, ClassVar, Literal

import numpy as np
import pandas as pd
import plotly.graph_objects as go  # type: ignore[import-untyped]
import xarray as xr

from akr_3d_map.mixins.observation_time import ObservationTimeCalculator
from akr_3d_map.utils import (
    add_celestial_bodies,
    creates_bin1d,
    get_3d_layout_config,
    save_plot,
)
from akr_3d_map.variables import (
    NumericType,
    PositiveNumber,
    earth_image_path_str,
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
    plot_in_cartesian: ClassVar[bool]

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
            Self: with xarray Dataset with coordinates (x, y, z) containing
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
            "observation_count": (
                list(dim_names),
                np.zeros(shape, dtype=np.int32),
                {"units": "count", "dtype": "int32"},
            ),
            "residence_count": (
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
                {"units": "dimensionless", "dtype": "float64"},
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
        """Add wireframe grid lines to 3D plot."""
        dim_names = self.get_dimension_names()

        # Get edges
        edges = [grid.coords[f"{dim_name}_edges"].to_numpy() for dim_name in dim_names]

        # Determine steps
        steps = []
        for i in range(len(edges)):
            num_edges = len(edges[i])
            step = max(1, num_edges // 15)
            steps.append(step)

        # Helper function for coordinate transformation
        def get_plot_coords(
            val_0: float,
            val_1: float,
            val_2: float,
        ) -> tuple[float, float, float]:
            if self.plot_in_cartesian:
                return self._transform_to_cartesian(val_0, val_1, val_2)
            return (val_0, val_1, val_2)  # Use native coordinates

        # Lines along dimension 0
        for val_1 in edges[1][:: steps[1]]:
            for val_2 in edges[2][:: steps[2]]:
                x_line, y_line, z_line = [], [], []
                for val_0 in edges[0]:
                    x, y, z = get_plot_coords(val_0, val_1, val_2)
                    x_line.append(x)
                    y_line.append(y)
                    z_line.append(z)

                fig.add_trace(
                    go.Scatter3d(
                        x=x_line,
                        y=y_line,
                        z=z_line,
                        mode="lines",
                        line={"color": "gray", "width": 1},
                        opacity=0.3,
                        showlegend=False,
                        hoverinfo="skip",
                    ),
                )

        # Lines along dimension 1
        for val_0 in edges[0][:: steps[0]]:
            for val_2 in edges[2][:: steps[2]]:
                x_line, y_line, z_line = [], [], []
                for val_1 in edges[1]:
                    x, y, z = get_plot_coords(val_0, val_1, val_2)
                    x_line.append(x)
                    y_line.append(y)
                    z_line.append(z)

                fig.add_trace(
                    go.Scatter3d(
                        x=x_line,
                        y=y_line,
                        z=z_line,
                        mode="lines",
                        line={"color": "gray", "width": 1},
                        opacity=0.3,
                        showlegend=False,
                        hoverinfo="skip",
                    ),
                )

        # Lines along dimension 2
        for val_0 in edges[0][:: steps[0]]:
            for val_1 in edges[1][:: steps[1]]:
                x_line, y_line, z_line = [], [], []
                for val_2 in edges[2]:
                    x, y, z = get_plot_coords(val_0, val_1, val_2)
                    x_line.append(x)
                    y_line.append(y)
                    z_line.append(z)

                fig.add_trace(
                    go.Scatter3d(
                        x=x_line,
                        y=y_line,
                        z=z_line,
                        mode="lines",
                        line={"color": "gray", "width": 1},
                        opacity=0.3,
                        showlegend=False,
                        hoverinfo="skip",
                    ),
                )

    def _get_axis_labels(self) -> dict[str, str]:
        """Get axis labels for plotting."""
        dim_names = self.get_dimension_names()
        units = self.get_coord_units()

        return {
            "x": f"{dim_names[0]} ({units[0]})",
            "y": f"{dim_names[1]} ({units[1]})",
            "z": f"{dim_names[2]} ({units[2]})",
        }

    def _get_hover_params(
        self,
        grid: xr.Dataset,
        ii: np.ndarray,
        jj: np.ndarray,
        kk: np.ndarray,
        variable: str,
        unit_str: str,
    ) -> dict[str, Any]:
        """Generates customdata and hovertemplate for 3D markers."""
        dim_names = self.get_dimension_names()
        clean_name = variable.replace("_", " ").title()

        # 1. Prepare Native Coordinates for the hover box
        # We extract the actual coordinate values for each active bin
        coord0_vals = grid[dim_names[0]].to_numpy()[ii]
        coord1_vals = grid[dim_names[1]].to_numpy()[jj]
        coord2_vals = grid[dim_names[2]].to_numpy()[kk]

        # Stack into (N, 3) array for Plotly's customdata
        customdata = np.stack((coord0_vals, coord1_vals, coord2_vals), axis=-1)

        # 2. Build the Hover Template
        # %{x}, %{y}, %{z} refer to the positions in the plot
        # %{customdata[i]} refers to the native grid coordinates
        template = (
            f"<b>{clean_name}</b>: %{{marker.color:.2f}}{unit_str}<br>"
            f"----------------<br>"
            f"{dim_names[0]}: %{{customdata[0]:.2f}}<br>"
            f"{dim_names[1]}: %{{customdata[1]:.2f}}<br>"
            f"{dim_names[2]}: %{{customdata[2]:.2f}}<br>"
        )

        # If we are in Cartesian mode, show X, Y, Z plot values as well
        if self.plot_in_cartesian:
            template += "Plot X: %{x:.2f} | Y: %{y:.2f} | Z: %{z:.2f}<br>"

        template += "<extra></extra>"

        return {"customdata": customdata, "hovertemplate": template}

    def plot_3d(
        self,
        variable: Literal[
            "normalised_observation_time",
            "residence_time",
            "observation_time",
            "burst_count",
        ]
        | None = None,
        path_to_save: str = "3D_Objects/grid.html",
        *,
        show_earth: bool = True,
        earth_image_path_str: str = earth_image_path_str,
        show_sun: bool = False,
        colorscale: str = "Viridis",
    ) -> "AKRGrid":
        """
        Plot 3D grid with wireframe (works for ALL coordinate systems).

        Args:
            variable: Name of the variable to plot (optional)
            path_to_save: Path to save the HTML file (default: '3D_Objects/grid.html')
            show_earth: Whether to show Earth in the plot (default: True)
            earth_image_path_str: Add path to the earth image
            show_sun: Whether to show Sun in the plot (default: False)
            colorscale: Choose appropriate color scale from plotly builtin color scales. documentation: https://plotly.com/python/builtin-colorscales/

        Returns:
            Self: for method chaining

        """
        # 1. Validate grid
        grid = self._validate_and_get_grid()
        dim_names = self.get_dimension_names()  # Retrieve dimension names

        # 2. Initialise figure
        fig = go.Figure()

        # 3. Add wireframe: reads from grid, writes to fig
        self._add_wireframe(fig, grid)

        # 4. Add Data Layer (Only if a variable is specified and exists)
        if variable and variable in grid:  # TODO(@Somu796)-01
            data_array = grid[variable].to_numpy()
            ii, jj, kk = np.where(data_array > 0)

            # Retrieve units from xarray attributes, default to empty string if not found
            unit = grid[variable].attrs.get("units", "")
            unit_str = f" ({unit})" if unit else ""

            # Construct the clean name
            clean_name = variable.replace("_", " ").title()
            display_label = f"{clean_name}{unit_str}"

            if len(ii) > 0:
                # Get coordinates - native or transformed
                if self.plot_in_cartesian:
                    # Transform each point
                    plot_x, plot_y, plot_z = [], [], []
                    for i, j, k in zip(ii, jj, kk, strict=True):
                        coord_0: Any = float(grid[dim_names[0]].to_numpy()[i])
                        coord_1: Any = float(grid[dim_names[1]].to_numpy()[j])
                        coord_2: Any = float(grid[dim_names[2]].to_numpy()[k])
                        x, y, z = self._transform_to_cartesian(
                            coord_0,
                            coord_1,
                            coord_2,
                        )
                        plot_x.append(x)
                        plot_y.append(y)
                        plot_z.append(z)
                else:
                    # Use native coordinates
                    plot_x = grid[dim_names[0]].to_numpy()[ii].tolist()
                    plot_y = grid[dim_names[1]].to_numpy()[jj].tolist()
                    plot_z = grid[dim_names[2]].to_numpy()[kk].tolist()

                # Accessing the hover template
                hover_params = self._get_hover_params(
                    grid,
                    ii,
                    jj,
                    kk,
                    variable,
                    unit_str,
                )
                # Plotting the figure
                fig.add_trace(
                    go.Scatter3d(
                        x=plot_x,
                        y=plot_y,
                        z=plot_z,
                        mode="markers",
                        customdata=hover_params["customdata"],
                        hovertemplate=hover_params["hovertemplate"],
                        marker={
                            "size": 5,
                            "color": data_array[ii, jj, kk],
                            "colorscale": colorscale,
                            "colorbar": {
                                "title": display_label,
                                "thickness": 15,
                            },
                            "opacity": 0.8,
                            "showscale": True,
                        },
                        name=clean_name if variable else "Data",
                        showlegend=False,
                    ),
                )

            # Set title
            title = f"3D Grid: {clean_name}"
        else:
            title = "3D Grid Base"

        # 5. Add celestial bodies
        if self.plot_in_cartesian:
            add_celestial_bodies(
                fig,
                show_earth=show_earth,
                earth_image_path=earth_image_path_str,
                show_sun=show_sun,
            )

        # 6. Layout
        if self.plot_in_cartesian:
            fig.update_layout(**get_3d_layout_config(title))
        else:
            axis_labels = self._get_axis_labels()
            fig.update_layout(**get_3d_layout_config(title, axis_labels=axis_labels))

        # 7. Save plot
        save_plot(fig, path_to_save)

        self.fig = fig
        return self


    def plot_3d_from_dataframe( # TODO(@Somu796)-07
        self,
        df: pd.DataFrame,
        path_to_save: str,
        variable: Literal[
            "normalised_observation_time",
            "residence_time",
            "observation_time",
            "burst_count",
            "observation_count",
            "residence_count",
        ],
        coord_colnames: tuple[str, str, str],
        colorscale: str = "Viridis",
        earth_image_path_str: str = earth_image_path_str,
        *,
        show_earth: bool = True,
        show_sun: bool = False,
    ) -> str:
        """
        Plot 3D scatter directly from a DataFrame and save to HTML.

        Args:
            df: DataFrame with coordinate columns and variable column.
            path_to_save: Path to save the HTML file.
            variable: Name of the variable column to plot.
            coord_colnames: Column names for (coord1, coord2, coord3).
                For cartesian: e.g. ("x_gse", "y_gse", "z_gse")
                For ltrmlat:   e.g. ("lt", "r", "mlat")
            colorscale: Plotly colorscale name.
            show_earth: Whether to show Earth (only applies to cartesian).
            earth_image_path_str: Path to earth image.
            show_sun: Whether to show Sun (only applies to cartesian).

        Returns:
            str: Path where the HTML file was saved.

        Example:
            >>> cart.plot_3d_from_dataframe(
            ...     df=cart_data,
            ...     path_to_save="assets/plot.html",
            ...     variable="normalised_observation_time",
            ...     coord_colnames=("x", "y", "z"),
            ... )

        """
        # 1. Validate coord columns exist
        missing = [c for c in coord_colnames if c not in df.columns]
        if missing:
            raise ValueError(f"Coordinate columns not found in DataFrame: {missing}")

        # 2. Validate variable exists
        if variable not in df.columns:
            raise ValueError(
                f"Variable '{variable}' not found in DataFrame columns: {list(df.columns)}"
            )

        # 3. Unpack coordinate columns
        x_col, y_col, z_col = coord_colnames

        # 4. Filter to non-zero values only
        df_plot = df[df[variable] > 0].copy()
        if df_plot.empty:
            raise ValueError(f"No non-zero values found for variable '{variable}'")

        # 5. Initialise figure
        fig = go.Figure()

        # 6. Build display label
        clean_name = variable.replace("_", " ").title()
        display_label = clean_name

        # 7. Convert coordinates if needed and build hover template
        if self.plot_in_cartesian:
            plot_x, plot_y, plot_z = [], [], []
            for _, row in df_plot.iterrows():
                x, y, z = self._transform_to_cartesian(
                    row[x_col], row[y_col], row[z_col]
                )
                plot_x.append(x)
                plot_y.append(y)
                plot_z.append(z)

            hovertemplate = (
                f"<b>{x_col}:</b> %{{x:.2f}} R<sub>E</sub><br>"
                f"<b>{y_col}:</b> %{{y:.2f}} R<sub>E</sub><br>"
                f"<b>{z_col}:</b> %{{z:.2f}} R<sub>E</sub><br>"
                f"<b>{clean_name}:</b> %{{customdata:.4f}}<br>"
                "<extra></extra>"
            )
        else:
            plot_x = df_plot[x_col].tolist()
            plot_y = df_plot[y_col].tolist()
            plot_z = df_plot[z_col].tolist()

            hovertemplate = (
                f"<b>{x_col}:</b> %{{x:.2f}} hrs<br>"
                f"<b>{y_col}:</b> %{{y:.2f}} R<sub>E</sub><br>"
                f"<b>{z_col}:</b> %{{z:.2f}}°<br>"
                f"<b>{clean_name}:</b> %{{customdata:.4f}}<br>"
                "<extra></extra>"
            )

        # 8. Add scatter trace
        fig.add_trace(
            go.Scatter3d(
                x=plot_x,
                y=plot_y,
                z=plot_z,
                mode="markers",
                customdata=df_plot[variable],
                hovertemplate=hovertemplate,
                marker={
                    "size": 5,
                    "color": df_plot[variable],
                    "colorscale": colorscale,
                    "colorbar": {
                        "title": display_label +" (dimensionless)",
                        "thickness": 15,
                    },
                    "opacity": 0.8,
                    "showscale": True,
                },
                name=clean_name,
                showlegend=False,
            ),
        )

        # 9. Add celestial bodies (cartesian only)
        if self.plot_in_cartesian:
            add_celestial_bodies(
                fig,
                show_earth=show_earth,
                earth_image_path=earth_image_path_str,
                show_sun=show_sun,
            )

        # 10. Layout
        title = f"3D Grid: {clean_name}"
        if self.plot_in_cartesian:
            fig.update_layout(**get_3d_layout_config(title))
        else:
            axis_labels = self._get_axis_labels()
            fig.update_layout(**get_3d_layout_config(title, axis_labels=axis_labels))

        # 11. Save and return path
        save_plot(fig, path_to_save)
        return path_to_save

    def save_grid(self, path_to_save: str = "./akr_grid.parquet", fmt: str = "parquet") -> str:
        """
        Saves the xarray Dataset to a specified format and path.

        Args:
            path_to_save: path directory where to save the file
            fmt: what format to save the file, e.g. parquet (default), netcdf, zarr

        """
        grid = self._validate_and_get_grid()  # type: ignore[attr-defined]
        # 1. Ask for format if not provided
        valid_formats = {"netcdf": ".nc", "zarr": ".zarr", "parquet": ".parquet"}

        if fmt not in valid_formats:
            error_message = (
                f"Unsupported format: {fmt}. Use {list(valid_formats.keys())}"
            )
            raise ValueError(error_message)

        # Ensure path has the correct extension if only a directory or base name is provided
        target_path = Path(path_to_save)
        if target_path.suffix != valid_formats[fmt]:
            target_path = target_path.with_suffix(valid_formats[fmt])

        # Logic for different formats
        if fmt == "parquet":
            cols_to_drop = [c for c in grid.coords if "_edges" in str(c)]
            # We use 'ordered' to ensure the coordinate columns stay in X, Y, Z order
            df_flat = grid.drop_vars(cols_to_drop).to_dataframe().reset_index()
            df_flat.to_parquet(target_path, index=False)

        elif fmt == "netcdf":
            grid.to_netcdf(target_path)

        elif fmt == "zarr":
            # consolidated=True is best practice for HPC/Cloud datasets
            grid.to_zarr(target_path, mode="w", consolidated=True)

        print(f"Successfully saved grid to {target_path} as {fmt}")
        return str(target_path)
