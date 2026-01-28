# %%
# Libraries
from dataclasses import dataclass

import numpy as np
import plotly.graph_objects as go  # type: ignore[import-untyped]
# import xarray as xr

from scripts.base_class import AKRGrid
# from scripts.utils import (
#     add_celestial_bodies,
#     add_grid_wireframe,
#     get_3d_layout_cartesian_config,
#     get_3d_layout_LTRMlat_config,
#     save_plot,
# )

# from scripts.base_class import AKRGrid
from scripts.variables import NumericType, PositiveNumber

# %%
# Create grid and visualisation
## Cartesian


@dataclass
class Cartesian(AKRGrid):
    """

    Create 3D grid in Cartesian (GSE) coordinates and help to visualise in 3D.

    Args:
        x_range: (min, max) for x dimension in Earth radii
        y_range: (min, max) for y dimension in Earth radii
        z_range: (min, max) for z dimension in Earth radii
        bin_size: Size of each bin in Earth radii
        grid: xarray Dataset (None until create_grid() is called)

    Example:
        >>> cart = Cartesian(bin_size=2.0)
        >>> cart.create_grid()
        >>> my_grid = cart.grid # Accessing the dataset
        >>> cart.add_data(residence_time, burst_count)
        >>> cart.plot_3d()

    """

    # _____________________Attributes_____________________
    x_range: tuple[NumericType, NumericType] = (-15, 15)
    y_range: tuple[NumericType, NumericType] = (-15, 15)
    z_range: tuple[NumericType, NumericType] = (-10, 10)
    x_bin_size: PositiveNumber = 0.5
    y_bin_size: PositiveNumber = 0.5
    z_bin_size: PositiveNumber = 0.5
    # grid: xr.Dataset | None = None

    # _____________________Methods: Mapping to Parent Class_____________________
    # Private methods to map coord_1/2/3 to x/y/z
    def _get_range_attrs(self) -> tuple[str, str, str]:
        """Map coord_1/2/3_range to x/y/z_range."""
        return ("x_range", "y_range", "z_range")

    def _get_bin_attrs(self) -> tuple[str, str, str]:
        """Map coord_1/2/3_bin to bin_size (uniform bins)."""
        return ("x_bin_size", "y_bin_size", "z_bin_size")

    # Public methods to get dimension names, units, and attributes
    def get_dimension_names(self) -> tuple[str, str, str]:
        return ("x", "y", "z")

    def get_coord_units(self) -> tuple[str, str, str]:
        return ("R_E", "R_E", "R_E")

    def get_grid_attrs(self) -> dict:
        return {
            "coordinate_system": "GSE",
            "units": "Earth_radii",
            "x_bin_size": float(self.x_bin_size),
            "y_bin_size": float(self.y_bin_size),
            "z_bin_size": float(self.z_bin_size),
            "description": "AKR detection grid",
        }
    # For plotting purposes, no transformation needed for Cartesian
    def _transform_to_cartesian(
        self,
        x_val: float,
        y_val: float,
        z_val: float,
    ) -> tuple[float, float, float]:
        """No transformation needed - already Cartesian."""
        return (x_val, y_val, z_val)

    
    # _____________________Cartesian specific Methods_____________________

# %% lt/r/mlat
@dataclass
class LTRMLat(AKRGrid):
    """
    Create 3D grid in Local Time / Radius / Magnetic Latitude coordinates.

    Args:
        lt_range: (min, max) local time in hours [0-24]
        r_range: (min, max) radial distance in Earth radii
        mlat_range: (min, max) magnetic latitude in degrees [-90, 90]
        lt_bin: Bin width for local time (hours)
        r_bin: Bin width for radial distance (R_E)
        mlat_bin: Bin width for magnetic latitude (degrees)

    Example:
        >>> cart = LTRMLat(bin_size=2.0)
        >>> cart.create_grid()
        >>> my_grid = cart.grid # Accessing the dataset
        >>> cart.add_data(residence_time, burst_count)
        >>> cart.plot_3d()

    """

    lt_range: tuple[NumericType, NumericType] = (0, 24)
    r_range: tuple[NumericType, NumericType] = (0, 15)
    mlat_range: tuple[NumericType, NumericType] = (-90, 90)
    lt_bin_size: PositiveNumber = 1.0
    r_bin_size: PositiveNumber = 25.0
    mlat_bin_size: PositiveNumber = 5.0
    # grid: xr.Dataset | None = None

    # _____________________Methods: Mapping to Parent Class_____________________
    # Private methods to map coord_1/2/3 to lt/r/mlat
    def _get_range_attrs(self) -> tuple[str, str, str]:
        """Map coord_1/2/3_range to lt/r/mlat_range."""
        return ("lt_range", "r_range", "mlat_range")

    def _get_bin_attrs(self) -> tuple[str, str, str]:
        """Map coord_1/2/3_bin to bin_size (uniform bins)."""
        return ("lt_bin_size", "r_bin_size", "mlat_bin_size")
    # Public methods to get dimension names, units, and attributes
    def get_dimension_names(self) -> tuple[str, str, str]:
        return ("lt", "r", "mlat")

    def get_coord_units(self) -> tuple[str, str, str]:
        return ("hours", "R_E", "degrees")

    def get_grid_attrs(self) -> dict:
        return {
            "coordinate_system": "GSE",
            "units": "Earth_radii",
            "lt_bin": float(self.lt_bin_size),
            "r_bin": float(self.r_bin_size),
            "mlat_bin": float(self.mlat_bin_size),
            "description": "AKR detection grid",
        }
    
    # For plotting purposes, transform LT/R/MLat to Cartesian
    def _transform_to_cartesian(
        self,
        lt_val: float,
        r_val: float,
        mlat_val: float,
    ) -> tuple[float, float, float]:
        """Transform LT/R/MLat to Cartesian X/Y/Z."""
        theta = (12 - lt_val) * np.pi / 12  # Hours to radians
        mlat_rad = np.radians(mlat_val)
        
        x = r_val * np.cos(mlat_rad) * np.cos(theta)
        y = r_val * np.cos(mlat_rad) * np.sin(theta)
        z = r_val * np.sin(mlat_rad)
        
        return (x, y, z)

    # _____________________LTRMLat specific Methods_____________________

