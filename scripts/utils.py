# Helper Function

from pathlib import Path

# from scripts.grid_3d import Cartesian, LTRMLat
from typing import Annotated

import numpy as np
import plotly.graph_objects as go  # type: ignore[import-untyped]
import plotly.io as pio  # type: ignore[import-untyped]
from pydantic import Field, validate_call

# %% Customised type hint
type NumericType = int | float
PositiveNumber = Annotated[NumericType, Field(gt=0)]


# %% 1D bin creation function
@validate_call
def creates_bin1d(
    start: NumericType,
    end: NumericType,
    bin_size: PositiveNumber = 2,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Create bin edges and centers for one dimension.

    Args:
        start: Starting value
        end: Ending value
        bin_size: Size of each bin (default: 2)

    Returns:
        Tuple of (edges, centers)

    Example:
        >>> edges, centers = creates_bin1d(0, 10, 2)
        >>> edges
        array([0, 2, 4, 6, 8, 10])
        >>> centers
        array([1., 3., 5., 7., 9.])

    """
    if end <= start:
        error_message = f"end ({end}) must be greater than start ({start})"
        raise ValueError(error_message)

    bin_edge = np.arange(start, end + bin_size, bin_size)
    bin_center = (bin_edge[:-1] + bin_edge[1:]) / 2
    return (bin_edge, bin_center)


def save_plot(fig: go.Figure, path: str) -> None:
    # Create the directory structure if it doesn't exist
    Path(path).parent.mkdir(parents=True, exist_ok=True)

    if path.endswith(".html"):
        # Saves as an interactive standalone file
        pio.write_html(fig, path)
    elif path.endswith(".json"):
        # Saves as a dynamic, modifiable data structure (Best for re-editing)
        pio.write_json(fig, path)
    else:
        error_msg = f"Unsupported file extension in path: {path}. Use .html or .json"
        raise ValueError(error_msg)


# %% Grid bin assignment functions
# def assign_bin_indices_cartesian(df: pd.DataFrame, grid: "Cartesian") -> pd.DataFrame:
#     """
#     Assign each position to a Cartesian grid bin.

#     Args:
#         df: DataFrame with x_gse, y_gse, z_gse columns
#         grid: Cartesian grid object

#     Returns:
#         df: DataFrame with new columns 'bin_x', 'bin_y', 'bin_z'

#     """
#     # Get bin edges
#     x_edges = grid.grid.x_edges.to_numpy()
#     y_edges = grid.grid.y_edges.to_numpy()
#     z_edges = grid.grid.z_edges.to_numpy()

#     # Digitize
#     bin_x = np.digitize(df["x_gse"].to_numpy(), x_edges) - 1
#     bin_y = np.digitize(df["y_gse"].to_numpy(), y_edges) - 1
#     bin_z = np.digitize(df["z_gse"].to_numpy(), z_edges) - 1

#     # Mark out-of-bounds
#     bin_x = np.where((bin_x >= 0) & (bin_x < len(x_edges) - 1), bin_x, -1)
#     bin_y = np.where((bin_y >= 0) & (bin_y < len(y_edges) - 1), bin_y, -1)
#     bin_z = np.where((bin_z >= 0) & (bin_z < len(z_edges) - 1), bin_z, -1)

#     df["bin_x"] = bin_x
#     df["bin_y"] = bin_y
#     df["bin_z"] = bin_z

#     return df


# def assign_bin_indices_lt_r_mlat(df: pd.DataFrame, grid: "LTRMLat") -> pd.DataFrame:
#     """
#     Assign each position to a LT/R/MLat grid bin.

#     Args:
#         df: DataFrame with LT_gse, radius, lat_gse columns
#         grid: LTRMLat grid object

#     Returns:
#         df: DataFrame with new columns 'bin_lt', 'bin_r', 'bin_mlat'

#     """
#     # Get bin edges
#     lt_edges = grid.grid.local_time_edges.to_numpy()
#     r_edges = grid.grid.radius_edges.to_numpy()
#     mlat_edges = grid.grid.mlat_edges.to_numpy()

#     # Digitize (use correct column names from your data!)
#     bin_lt = np.digitize(df["LT_gse"].values, lt_edges) - 1
#     bin_r = np.digitize(df["radius"].values, r_edges) - 1
#     bin_mlat = np.digitize(df["lat_gse"].values, mlat_edges) - 1  # or mlat_gse?

#     # Mark out-of-bounds
#     bin_lt = np.where((bin_lt >= 0) & (bin_lt < len(lt_edges) - 1), bin_lt, -1)
#     bin_r = np.where((bin_r >= 0) & (bin_r < len(r_edges) - 1), bin_r, -1)
#     bin_mlat = np.where(
#         (bin_mlat >= 0) & (bin_mlat < len(mlat_edges) - 1), bin_mlat, -1
#     )

#     df["bin_lt"] = bin_lt
#     df["bin_r"] = bin_r
#     df["bin_mlat"] = bin_mlat

#     return df
