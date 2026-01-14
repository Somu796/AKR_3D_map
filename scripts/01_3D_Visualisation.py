#%%
# Libraries
import numpy as np
import xarray as xr
import plotly.graph_objects as go
from pydantic import validate_call, Field
from typing_extensions import Annotated

type NumericType = int | float # Customised type hint
PositiveNumber = Annotated[NumericType, Field(gt=0)]

#%%
# Function to create 1D grid
@validate_call
def creates_bin1d(start:NumericType , end:NumericType , bin_size:PositiveNumber=2) -> tuple[np.ndarray, np.ndarray]:
    """
    Create bin edges and centers for one dimension.
    
    Args:
        start: Starting value
        end: Ending value  
        bin_size: Size of each bin (default: 2)
    
    Returns:
        Tuple of (edges, centers)
    
    Example:
        >>> edges, centers = create_bins_1d(0, 10, 2)
        >>> edges
        array([0, 2, 4, 6, 8, 10])
        >>> centers
        array([1., 3., 5., 7., 9.])
    """    
    if end <= start:
        raise ValueError(f"end ({end}) must be greater than start ({start})")

    bin_edge = np.arange(start, end+bin_size, bin_size)
    bin_center = (bin_edge[:-1] + bin_edge[1:])/2
    return (bin_edge, bin_center)

# Creating Cartesian grid
@validate_call
def create_cartesian_grid(
        x_range: tuple[NumericType, NumericType] = (-15, 15),
        y_range: tuple[NumericType, NumericType] = (-15, 15),
        z_range: tuple[NumericType, NumericType] = (-10, 10),
        bin_size: NumericType=0.5,
)-> xr.Dataset:
    """
    Create 3D grid in Cartesian (GSE) coordinates.
    
    Args:
        x_range: (min, max) for x dimension in Earth radii
        y_range: (min, max) for y dimension in Earth radii
        z_range: (min, max) for z dimension in Earth radii
        bin_size: Size of each bin in Earth radii
    
    Returns:
        xarray Dataset with coordinates (x, y, z) containing 
        placeholder data variables for residence_time, burst_count, burst_time, 
        and probability.
    
    Example:
        >>> grid = create_cartesian_grid(bin_size=2.0)
        >>> grid.dims
        Frozen({'x': 15, 'y': 15, 'z': 10})
    """
        
    # Creating edges
    x_edges, x_centers = creates_bin1d(x_range[0], x_range[1], bin_size)
    y_edges, y_centers = creates_bin1d(y_range[0], y_range[1], bin_size)
    z_edges, z_centers = creates_bin1d(z_range[0], z_range[1], bin_size)

    # dimensions (how many bins in each direction)
    x_n: int = len(x_centers)
    y_n: int = len(y_centers)
    z_n: int = len(z_centers)

    # Initialize arrays with proper dtype (?Check Data Type later)
    shape: tuple[int, int, int] = (x_n, y_n, z_n)
    residence_time: np.ndarray = np.zeros(shape, dtype=np.float64)
    burst_count: np.ndarray = np.zeros(shape, dtype=np.int32)
    burst_time: np.ndarray = np.zeros(shape, dtype=np.float64)
    probability: np.ndarray = np.zeros(shape, dtype=np.float64)

    # Create grid xr.Dataset(parameter={key:([],var,{'var_metadata':'})})
    grid: xr.Dataset = xr.Dataset(
        data_vars= {
        "residence_time": (['x', 'y', 'z'], residence_time, {'units': 'seconds', 'dtype': 'float64'}),
        "burst_count": (['x', 'y', 'z'], burst_count, {'units': 'count', 'dtype': 'int32'}),
        "burst_time": (['x', 'y', 'z'], burst_time, {'units': 'seconds', 'dtype': 'float64'}),
        "probability": (['x', 'y', 'z'], probability, {'units': 'percent', 'dtype': 'float64'})
        },
        coords= {
        # Centers
        "x":(['x'], x_centers, {'units': 'R_E', 'dtype': 'float64'}),
        "y": (['y'], y_centers, {'units': 'R_E', 'dtype': 'float64'}),
        "z": (['z'], z_centers, {'units': 'R_E', 'dtype': 'float64'}),

        # Edges
        "x_edges": (['x_edges'], x_edges, {'units': 'R_E', 'dtype': 'float64'}),
        "y_edges": (['y_edges'], y_edges, {'units': 'R_E', 'dtype': 'float64'}),
        "z_edges": (['z_edges'], z_edges, {'units': 'R_E', 'dtype': 'float64'}) 
        },
        attrs ={
        'coordinate_system': 'GSE',
        'units': 'Earth_radii',
        'bin_size': float(bin_size),
        'description': 'AKR detection probability grid'            
        }
    )
    return grid

#Creating lt_r_mlat_grid
@validate_call
def create_lt_r_mlat_grid(
    lt_range: tuple[NumericType, NumericType] = (0, 24),
    r_range: tuple[NumericType, NumericType] = (0, 150),
    mlat_range: tuple[NumericType, NumericType] = (-90, 90),
    lt_bin: PositiveNumber = 1.0,
    r_bin: PositiveNumber = 25.0,
    mlat_bin: PositiveNumber = 5.0
) -> xr.Dataset:  
    """
    Create 3D grid in Local Time / Radius / Magnetic Latitude coordinates.

    Args:
        lt_range: (min, max) local time in hours [0-24]
        r_range: (min, max) radial distance in Earth radii
        mlat_range: (min, max) magnetic latitude in degrees [-90, 90]
        lt_bin: Bin width for local time (hours)
        r_bin: Bin width for radial distance (R_E)
        mlat_bin: Bin width for magnetic latitude (degrees)

    Returns:
        xarray Dataset with coordinates (local_time, radius, mlat)
        containing placeholder data variables for residence_time, burst_count, 
        burst_time, and probability.

    Example:
        >>> grid = create_lt_r_mlat_grid()
        >>> grid.dims
        Frozen({'local_time': 24, 'radius': 6, 'mlat': 36})
    """
    # Creating bins for each dimension
    lt_edges, lt_centers = creates_bin1d(lt_range[0], lt_range[1], lt_bin)
    r_edges, r_centers = creates_bin1d(r_range[0], r_range[1], r_bin)
    mlat_edges, mlat_centers = creates_bin1d(mlat_range[0], mlat_range[1], mlat_bin)

    # dimensions (how many bins in each direction)
    lt_n: int = len(lt_centers)
    r_n: int = len(r_centers)
    mlat_n: int = len(mlat_centers)
        
    # Initialize arrays with proper dtype (?Check Data Type later)
    shape: tuple[int, int, int] = (lt_n, r_n, mlat_n)
    residence_time: np.ndarray = np.zeros(shape, dtype=np.float64)
    burst_count: np.ndarray = np.zeros(shape, dtype=np.int32)
    burst_time: np.ndarray = np.zeros(shape, dtype=np.float64)
    probability: np.ndarray = np.zeros(shape, dtype=np.float64)
    grid: xr.Dataset = xr.Dataset(
        data_vars= {
        "residence_time": (['local_time', 'radius', 'mlat'], residence_time, {'units': 'seconds', 'dtype': 'float64'}),
        "burst_count": (['local_time', 'radius', 'mlat'], burst_count, {'units': 'count', 'dtype': 'int32'}),
        "burst_time": (['local_time', 'radius', 'mlat'], burst_time, {'units': 'seconds', 'dtype': 'float64'}),
        "probability": (['local_time', 'radius', 'mlat'], probability, {'units': 'percent', 'dtype': 'float64'})
        },
        coords= {
        # Centers
        "local_time":(['local_time'], lt_centers, {'units': 'hours', 'dtype': 'float64'}),
        "radius": (['radius'], r_centers, {'units': 'R_E', 'dtype': 'float64'}),
        "mlat": (['mlat'], mlat_centers, {'units': 'degrees', 'dtype': 'float64'}),

        # Edges
        "local_time_edges": (['local_time_edges'], lt_edges, {'units': 'hours', 'dtype': 'float64'}),
        "radius_edges": (['radius_edges'], r_edges, {'units': 'R_E', 'dtype': 'float64'}),
        "mlat_edges": (['mlat_edges'], mlat_edges, {'units': 'degrees', 'dtype': 'float64'}) 
        },
        attrs = {
            'coordinate_system': 'Local Time / Radius / Magnetic Latitude',
            'local_time_units': 'hours',
            'radius_units': 'R_E', 
            'mlat_units': 'degrees',
            'lt_bin_width': float(lt_bin),
            'r_bin_width': float(r_bin),
            'mlat_bin_width': float(mlat_bin),
            'description': 'AKR detection probability grid in LT/R/MLat coordinates',
        }
    )
    return grid
#%% Test Cartesian
# Test Cartesian
cart_grid = create_cartesian_grid(bin_size=2.0)
print("Cartesian Grid:")
print(f"  Dimensions: {cart_grid.dims}")
print(f"  Coords: {list(cart_grid.coords)}")
print(f"  Coord system: {cart_grid.attrs['coordinate_system']}\n")

# Test LT/R/MLat
lt_grid = create_lt_r_mlat_grid()
print("LT/R/MLat Grid:")
print(f"  Dimensions: {lt_grid.dims}")
print(f"  Coords: {list(lt_grid.coords)}")
print(f"  Coord system: {lt_grid.attrs['coordinate_system']}")
print(f"  Reference: {lt_grid.attrs['reference']}")


# # Create the grid
# grid = create_cartesian_grid(
#     x_range=(-15.0, 15.0),
#     y_range=(-15.0, 15.0),
#     z_range=(-10.0, 10.0),
#     bin_size=2.0
# )

# print(f"Grid created: {grid.dims}")
# print(f"Shape: {grid.residence_time.shape}")

# # Access Edges
# # bin_size = grid.attrs['bin_size']
# x_edges = grid.x_edges.values
# y_edges = grid.y_edges.values
# z_edges = grid.z_edges.values

#%% Create figure cartesian
# # Create figure cartesian
def plot_cartesian(grid:xr.Dataset, sun:bool = False):
    """
    Plot cartesian grid in 3D.

    Args:

    """
    # Access Edges
    x_edges = grid.x_edges.values
    y_edges = grid.y_edges.values
    z_edges = grid.z_edges.values

    fig = go.Figure()

    # Draw grid lines along X-axis
    for y in y_edges:
        for z in z_edges:
            fig.add_trace(go.Scatter3d(
                x=x_edges, y=[y]*len(x_edges), z=[z]*len(x_edges),
                mode='lines', line=dict(color='gray', width=1),
                opacity=0.3, showlegend=False, hoverinfo='skip'
            ))

    # Draw grid lines along Y-axis
    for x in x_edges:
        for z in z_edges:
            fig.add_trace(go.Scatter3d(
                x=[x]*len(y_edges), y=y_edges, z=[z]*len(y_edges),
                mode='lines', line=dict(color='gray', width=1),
                opacity=0.3, showlegend=False, hoverinfo='skip'
            ))

    # Draw grid lines along Z-axis
    for x in x_edges:
        for y in y_edges:
            fig.add_trace(go.Scatter3d(
                x=[x]*len(z_edges), y=[y]*len(z_edges), z=z_edges,
                mode='lines', line=dict(color='gray', width=1),
                opacity=0.3, showlegend=False, hoverinfo='skip'
            ))

    # Add Earth sphere at origin
    earth=True
    sun=False
    u = np.linspace(0, 2*np.pi, 50)
    v = np.linspace(0, np.pi, 50)
    x_sphere = np.outer(np.cos(u), np.sin(v))
    y_sphere = np.outer(np.sin(u), np.sin(v))
    z_sphere = np.outer(np.ones(np.size(u)), np.cos(v))

    fig.add_trace(go.Surface(
        x=x_sphere, y=y_sphere, z=z_sphere,
        colorscale='Blues', showscale=False,
        opacity=0.6, name='Earth'
    ))

    # Add Sun (positioned along positive X-axis, typical GSE orientation)
    sun_distance = -150  # Sun at ~150 R_E along -X axis (sunward direction)
    sun_radius = 5  # Visual size

    fig.add_trace(go.Surface(
        x=x_sphere * sun_radius + sun_distance,
        y=y_sphere * sun_radius,
        z=z_sphere * sun_radius,
        colorscale=[[0, 'yellow'], [1, 'orange']],
        showscale=False,
        opacity=0.8,
        name='Sun'
    ))

    # Layout
    fig.update_layout(
        # Global font
        font=dict(
            family="Times New Roman, Times, serif",
            size=14,
            color="#1a1a1a"
        ),
        
        # Figure title
        title=dict(
            text='3D Grid with Earth and Sun' if (sun and earth) else '3D Grid of Earth',
            font=dict(size=22),
            x=0.5,
            xanchor='center'
        ),
        
        # 3D scene
        scene=dict(
            xaxis=dict(
                title=dict(text='X (R<sub>E</sub>)', font=dict(size=16)),
                tickfont=dict(size=12),
                gridcolor='#cccccc',
                showbackground=True,
                backgroundcolor='#f5f5f5'
            ),
            yaxis=dict(
                title=dict(text='Y (R<sub>E</sub>)', font=dict(size=16)),
                tickfont=dict(size=12),
                gridcolor='#cccccc',
                showbackground=True,
                backgroundcolor='#f5f5f5'
            ),
            zaxis=dict(
                title=dict(text='Z (R<sub>E</sub>)', font=dict(size=16)),
                tickfont=dict(size=12),
                gridcolor='#cccccc',
                showbackground=True,
                backgroundcolor='#f5f5f5'
            ),
            
            # Fixed camera (X horizontal, Z vertical)
            camera=dict(
                eye=dict(x=0.3, y=2.5, z=0.8),  # View from Y-axis, slightly elevated
                center=dict(x=0, y=0, z=0),
                up=dict(x=0, y=0, z=1)          # Z points UP
            ),
            
            # Proper scaling
            aspectmode='data',  # True scale based on data ranges
            
            # Optional: Add this for better interaction
            dragmode='orbit'
        ),
        
        # Size and background
        width=1000,
        height=800,
        paper_bgcolor='white',
        margin=dict(l=0, r=0, t=70, b=0)
    )

# fig.show()
 #%%

import xarray as xr
# Create figure lt/r/mlat
def plot_lt_r_mlat_3d(grid: xr.Dataset):
    """
    Plot LT/R/MLat grid in 3D by converting to Cartesian.
    """
    # Get coordinates
    lt = grid.coords['local_time'].values
    r = grid.coords['radius'].values  
    mlat = grid.coords['mlat'].values
    
    # Create meshgrid
    LT, R, MLAT = np.meshgrid(lt, r, mlat, indexing='ij')
    
    # Convert to Cartesian
    theta = (12 - LT) * np.pi / 12  # Hours to radians
    mlat_rad = np.radians(MLAT)
    
    X = R * np.cos(mlat_rad) * np.cos(theta)
    Y = R * np.cos(mlat_rad) * np.sin(theta)
    Z = R * np.sin(mlat_rad)
    
    fig = go.Figure()
    
    # Draw grid lines along Local Time circles
    for r_val in grid.coords['radius_edges'].values:
        for mlat_val in grid.coords['mlat_edges'].values[::5]:  # Every 5th
            theta_line = (12 - lt) * np.pi / 12
            mlat_rad = np.radians(mlat_val)
            
            x_line = r_val * np.cos(mlat_rad) * np.cos(theta_line)
            y_line = r_val * np.cos(mlat_rad) * np.sin(theta_line)
            z_line = r_val * np.sin(mlat_rad) * np.ones_like(theta_line)
            
            fig.add_trace(go.Scatter3d(
                x=x_line, y=y_line, z=z_line,
                mode='lines',
                line=dict(color='gray', width=1),
                opacity=0.3,
                showlegend=False,
                hoverinfo='skip'
            ))
    
    # Draw grid lines along Radius
    for lt_val in grid.coords['local_time_edges'].values[::3]:
        for mlat_val in grid.coords['mlat_edges'].values[::5]:
            theta_val = (12 - lt_val) * np.pi / 12
            mlat_rad = np.radians(mlat_val)
            
            r_line = grid.coords['radius_edges'].values
            x_line = r_line * np.cos(mlat_rad) * np.cos(theta_val)
            y_line = r_line * np.cos(mlat_rad) * np.sin(theta_val)
            z_line = r_line * np.sin(mlat_rad)
            
            fig.add_trace(go.Scatter3d(
                x=x_line, y=y_line, z=z_line,
                mode='lines',
                line=dict(color='gray', width=1),
                opacity=0.3,
                showlegend=False,
                hoverinfo='skip'
            ))
    
    # Add Earth
    u = np.linspace(0, 2*np.pi, 50)
    v = np.linspace(0, np.pi, 50)
    x_sphere = np.outer(np.cos(u), np.sin(v))
    y_sphere = np.outer(np.sin(u), np.sin(v))
    z_sphere = np.outer(np.ones(np.size(u)), np.cos(v))
    
    fig.add_trace(go.Surface(
        x=x_sphere, y=y_sphere, z=z_sphere,
        colorscale='Blues',
        showscale=False,
        opacity=0.6,
        name='Earth'
    ))
    
    fig.update_layout(
        title='LT/R/MLat Grid in 3D',
        scene=dict(
            xaxis_title='X (R_E)',
            yaxis_title='Y (R_E)',
            zaxis_title='Z (R_E)',
            aspectmode='data',
            camera=dict(
                eye=dict(x=1.5, y=1.5, z=1.2),
                up=dict(x=0, y=0, z=1)
            )
        ),
        width=1000,
        height=800
    )
    
    return fig
#%%
def plot_lt_r_mlat_3d(grid: xr.Dataset):
    """
    Plot LT/R/MLat grid in 3D with Earth sphere at center.
    """
    fig = go.Figure()
    
    # Get coordinates
    lt_edges = grid.coords['local_time_edges'].values
    r_edges = grid.coords['radius_edges'].values
    mlat_edges = grid.coords['mlat_edges'].values
    
    # ========== DRAW GRID LINES ==========
    
    # Lines at constant R and MLat (circles around at different heights)
    for r_val in r_edges[::2]:  # Every other radius
        for mlat_val in mlat_edges[::6]:  # Every 6th latitude
            theta_circle = (12 - np.linspace(0, 24, 100)) * np.pi / 12
            mlat_rad = np.radians(mlat_val)
            
            x_circle = r_val * np.cos(mlat_rad) * np.cos(theta_circle)
            y_circle = r_val * np.cos(mlat_rad) * np.sin(theta_circle)
            z_circle = r_val * np.sin(mlat_rad) * np.ones_like(theta_circle)
            
            fig.add_trace(go.Scatter3d(
                x=x_circle, y=y_circle, z=z_circle,
                mode='lines',
                line=dict(color='lightgray', width=1),
                opacity=0.3,
                showlegend=False,
                hoverinfo='skip'
            ))
    
    # Lines at constant LT and MLat (radial lines)
    for lt_val in lt_edges[::3]:  # Every 3rd hour
        for mlat_val in mlat_edges[::6]:
            theta_val = (12 - lt_val) * np.pi / 12
            mlat_rad = np.radians(mlat_val)
            
            r_line = r_edges
            x_line = r_line * np.cos(mlat_rad) * np.cos(theta_val)
            y_line = r_line * np.cos(mlat_rad) * np.sin(theta_val)
            z_line = r_line * np.sin(mlat_rad)
            
            fig.add_trace(go.Scatter3d(
                x=x_line, y=y_line, z=z_line,
                mode='lines',
                line=dict(color='lightgray', width=1),
                opacity=0.3,
                showlegend=False,
                hoverinfo='skip'
            ))
    
    # ========== ADD EARTH SPHERE AT CENTER ==========
    u = np.linspace(0, 2*np.pi, 50)
    v = np.linspace(0, np.pi, 50)
    
    earth_radius = 1  # 1 R_E
    x_earth = earth_radius * np.outer(np.cos(u), np.sin(v))
    y_earth = earth_radius * np.outer(np.sin(u), np.sin(v))
    z_earth = earth_radius * np.outer(np.ones(np.size(u)), np.cos(v))
    
    fig.add_trace(go.Surface(
        x=x_earth, y=y_earth, z=z_earth,
        colorscale='Blues',
        showscale=False,
        opacity=0.7,
        name='Earth (1 R_E)',
        hovertext='Earth',
        hoverinfo='text'
    ))
    
    # ========== ADD LT MARKERS AT OUTER EDGE ==========
    r_max = r_edges[-1]
    for hour in [0, 6, 12, 18]:
        theta = (12 - hour) * np.pi / 12
        x_marker = r_max * 1.1 * np.cos(theta)
        y_marker = r_max * 1.1 * np.sin(theta)
        
        fig.add_trace(go.Scatter3d(
            x=[x_marker], y=[y_marker], z=[0],
            mode='text',
            text=[f'{hour:02d} LT'],
            textfont=dict(size=14, color='black'),
            showlegend=False,
            hoverinfo='skip'
        ))
    
    # ========== ADD SUN INDICATOR ==========
    sun_distance = r_max * 1.2
    fig.add_trace(go.Scatter3d(
        x=[0], y=[sun_distance], z=[0],
        mode='markers+text',
        marker=dict(size=10, color='orange', symbol='circle'),
        text=['☀️ Sun'],
        textposition='top center',
        textfont=dict(size=16, color='orange'),
        name='Sun direction',
        showlegend=False
    ))
    
    # Layout
    fig.update_layout(
        title='LT / Radius / Magnetic Latitude Grid<br>with Earth at Center',
        scene=dict(
            xaxis=dict(
                title='X (R_E)',
                range=[-r_max*1.2, r_max*1.2]
            ),
            yaxis=dict(
                title='Y (R_E)',
                range=[-r_max*1.2, r_max*1.2]
            ),
            zaxis=dict(
                title='Z (R_E)',
                range=[-r_max*1.2, r_max*1.2]
            ),
            aspectmode='data',
            camera=dict(
                eye=dict(x=1.5, y=1.5, z=1.2),
                center=dict(x=0, y=0, z=0),
                up=dict(x=0, y=0, z=1)
            )
        ),
        width=1000,
        height=800,
        font=dict(family="Times New Roman, Times, serif", size=12)
    )
    return fig
#%%
grid = create_lt_r_mlat_grid()
fig_3d = plot_lt_r_mlat_3d(grid)
fig_3d.show()

# %%
# Remove all environmenatal parameters
# %reset -f