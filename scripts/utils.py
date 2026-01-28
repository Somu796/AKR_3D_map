# Helper Function

from pathlib import Path
from typing import Any

import numpy as np
import plotly.graph_objects as go  # type: ignore[import-untyped]
import plotly.io as pio  # type: ignore[import-untyped]

# from scripts.grid_3d import Cartesian, LTRMLat
from PIL import Image
from pydantic import validate_call

from scripts.variables import NumericType, PositiveNumber


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


# %% plot3D Figure layout helper functions
def add_grid_wireframe(
    fig: go.Figure,
    x_edges: np.ndarray,
    y_edges: np.ndarray,
    z_edges: np.ndarray,
) -> go.Figure:
    """Adds gray wireframe lines to the 3D plot."""
    # We collect all coordinates into lists separated by None
    x_lines, y_lines, z_lines = [], [], []

    # Lines along X
    for y in y_edges:
        for z in z_edges:
            x_lines.extend([x_edges[0], x_edges[-1], None])
            y_lines.extend([y, y, None])
            z_lines.extend([z, z, None])

    # Lines along Y
    for x in x_edges:
        for z in z_edges:
            x_lines.extend([x, x, None])
            y_lines.extend([y_edges[0], y_edges[-1], None])
            z_lines.extend([z, z, None])

    # Lines along Z
    for x in x_edges:
        for y in y_edges:
            x_lines.extend([x, x, None])
            y_lines.extend([y, y, None])
            z_lines.extend([z_edges[0], z_edges[-1], None])

    fig.add_trace(
        go.Scatter3d(
            x=x_lines,
            y=y_lines,
            z=z_lines,
            mode="lines",
            line={"color": "lightgray", "width": 1},
            opacity=0.1,
            name="Grid Wireframe",
            showlegend=False,
            hoverinfo="skip",
        ),
    )


def add_celestial_bodies(
    fig: go.Figure,
    *,
    show_earth: bool = True,
    show_sun: bool = False,
    earth_image_path: str = "assets/flat_earth_image_no_cloud.png",
) -> go.Figure:
    """Adds Earth and/or Sun surfaces to a Plotly figure."""
    # 1. Generate base sphere coordinates
    u = np.linspace(0, 2 * np.pi, 50)
    v = np.linspace(0, np.pi, 50)
    x_sphere = np.outer(np.cos(u), np.sin(v))
    y_sphere = np.outer(np.sin(u), np.sin(v))
    z_sphere = np.outer(np.ones(np.size(u)), np.cos(v))

    if show_earth:
        if Path(earth_image_path).exists():
            # Load local texture
            img = Image.open(earth_image_path).convert("L")
            # Convert image to numerical array and normalize
            # We use the mean of RGB to get a brightness map of the continents
            img_data = np.array(img)

            fig.add_trace(
                go.Surface(
                    x=x_sphere,
                    y=y_sphere,
                    z=z_sphere,
                    surfacecolor=np.flipud(img_data),
                    colorscale=[
                        [0, "rgb(10, 20, 50)"],  # Deep Oceans
                        [0.4, "rgb(30, 80, 140)"],  # Coastlines
                        [0.45, "rgb(60, 120, 40)"],  # Vegetation
                        [0.7, "rgb(180, 170, 120)"],  # Desert/Highlands
                        [1, "rgb(255, 255, 255)"],  # Clouds/Ice
                    ],
                    showscale=False,
                    name="Earth",
                    hoverinfo="name",
                    lighting={"ambient": 0.6, "diffuse": 0.8, "specular": 0.1},
                ),
            )
        else:
            print(f"Warning: {earth_image_path} not found. Using fallback sphere.")
            fig.add_trace(
                go.Surface(
                    x=x_sphere,
                    y=y_sphere,
                    z=z_sphere,
                    colorscale="Blues",
                    showscale=False,
                    opacity=0.6,
                    name="Earth",
                    hoverinfo="name",
                ),
            )

    if show_sun:
        # GSE convention: Sun is far away along the +X axis
        # (Though visually we often pull it closer for reference)
        sun_distance = 150
        sun_radius = 5
        fig.add_trace(
            go.Surface(
                x=x_sphere * sun_radius + sun_distance,
                y=y_sphere * sun_radius,
                z=z_sphere * sun_radius,
                colorscale=[[0, "yellow"], [1, "orange"]],
                showscale=False,
                opacity=0.8,
                name="Sun",
                hoverinfo="name",
            ),
        )


def get_3d_layout_config(
    title_text: str,
    axis_labels: dict[str, str] | None = None,
) -> dict[str, Any]:
    """Returns a standard Plotly layout configuration for GSE 3D plots."""
    # Default labels
    if axis_labels is None:
        axis_labels = {
            "x": "X (R<sub>E</sub>)",
            "y": "Y (R<sub>E</sub>)",
            "z": "Z (R<sub>E</sub>)",
        }

    return {
        "title": {
            "text": title_text,
            "font": {"size": 22},
            "x": 0.5,
            "xanchor": "center",
        },
        "font": {
            "family": "Times New Roman, Times, serif",
            "size": 14,
            "color": "#1a1a1a",
        },
        "scene": {
            "xaxis": {
                "title": {"text": axis_labels["x"], "font": {"size": 16}},
                "tickfont": {"size": 12},
                "gridcolor": "#cccccc",
                "showbackground": True,
                "backgroundcolor": "#f5f5f5",
            },
            "yaxis": {
                "title": {"text": axis_labels["y"], "font": {"size": 16}},
                "tickfont": {"size": 12},
                "gridcolor": "#cccccc",
                "showbackground": True,
                "backgroundcolor": "#f5f5f5",
            },
            "zaxis": {
                "title": {"text": axis_labels["z"], "font": {"size": 16}},
                "tickfont": {"size": 12},
                "gridcolor": "#cccccc",
                "showbackground": True,
                "backgroundcolor": "#f5f5f5",
            },
            "camera": {
                "eye": {"x": 0.3, "y": 2.5, "z": 0.8},
                "center": {"x": 0, "y": 0, "z": 0},
                "up": {"x": 0, "y": 0, "z": 1},
            },
            "aspectmode": "data",
            "dragmode": "orbit",
        },
        "width": 1000,
        "height": 800,
        "paper_bgcolor": "white",
        "margin": {"l": 0, "r": 0, "t": 70, "b": 0},
    }


def save_plot(fig: go.Figure, path: str) -> None:
    """Saves a Plotly figure to an HTML or JSON file."""
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
