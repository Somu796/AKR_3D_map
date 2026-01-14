"""
Create 3D Grid for AKR Detection Probability Mapping

This script creates an empty 3D grid around Earth in GSE coordinates,
fills it with random test data, and visualizes it in 3D.

Author: Sudipta Kumar Hazra
Date: 2025-01-06
"""

import numpy as np
import xarray as xr
import plotly.graph_objects as go
from typing import Tuple, Optional
import json

# ==============================================================================
# STEP 1: DEFINE GRID PARAMETERS (Discussion points with supervisor)
# ==============================================================================

class GridConfig:
    """Configuration for 3D grid parameters."""
    
    # Coordinate system
    COORDINATE_SYSTEM = "GSE"  # Geocentric Solar Ecliptic
    
    # Grid extent (Earth radii, R_E)
    # Discussion point: Should we go further? Different for X,Y,Z?
    X_RANGE: Tuple[float, float] = (-15.0, 15.0)  # Sun-Earth line
    Y_RANGE: Tuple[float, float] = (-15.0, 15.0)  # Perpendicular to Sun
    Z_RANGE: Tuple[float, float] = (-10.0, 10.0)  # North-South (smaller range)
    
    # Bin size (in Earth radii)
    # Discuss the size with supervisor as smaller bins = more detail but more memory/time
    BIN_SIZE: float = 1.0
    
    # Data types for efficiency
    FLOAT_DTYPE = np.float32
    INT_DTYPE = np.int32
    
    def __repr__(self) -> str:
        """String representation of configuration."""
        n_x = int((self.X_RANGE[1] - self.X_RANGE[0]) / self.BIN_SIZE)
        n_y = int((self.Y_RANGE[1] - self.Y_RANGE[0]) / self.BIN_SIZE)
        n_z = int((self.Z_RANGE[1] - self.Z_RANGE[0]) / self.BIN_SIZE)
        total_cells = n_x * n_y * n_z
        
        return f"""
Grid Configuration:
  Coordinate System: {self.COORDINATE_SYSTEM}
  X Range: {self.X_RANGE[0]} to {self.X_RANGE[1]} R_E ({n_x} bins)
  Y Range: {self.Y_RANGE[0]} to {self.Y_RANGE[1]} R_E ({n_y} bins)
  Z Range: {self.Z_RANGE[0]} to {self.Z_RANGE[1]} R_E ({n_z} bins)
  Bin Size: {self.BIN_SIZE} R_E
  Total Grid Cells: {total_cells:,}
  Memory Usage: ~{total_cells * 4 * 4 / 1e6:.1f} MB (4 float32 arrays)
"""

# ==============================================================================
# STEP 2: CREATE EMPTY 3D GRID
# ==============================================================================

def create_3d_grid(config: GridConfig) -> xr.Dataset:
    """
    Create empty 3D grid around Earth.
    
    Parameters
    ----------
    config : GridConfig
        Grid configuration parameters
    
    Returns
    -------
    xr.Dataset
        3D grid with coordinates and empty data arrays
    """
    
    print("Creating 3D grid...")
    
    # Create bin edges
    x_bins = np.arange(
        config.X_RANGE[0], 
        config.X_RANGE[1] + config.BIN_SIZE, 
        config.BIN_SIZE,
        dtype=config.FLOAT_DTYPE
    )
    y_bins = np.arange(
        config.Y_RANGE[0], 
        config.Y_RANGE[1] + config.BIN_SIZE, 
        config.BIN_SIZE,
        dtype=config.FLOAT_DTYPE
    )
    z_bins = np.arange(
        config.Z_RANGE[0], 
        config.Z_RANGE[1] + config.BIN_SIZE, 
        config.BIN_SIZE,
        dtype=config.FLOAT_DTYPE
    )
    
    # Create bin centers (for coordinates)
    x_centers = (x_bins[:-1] + x_bins[1:]) / 2
    y_centers = (y_bins[:-1] + y_bins[1:]) / 2
    z_centers = (z_bins[:-1] + z_bins[1:]) / 2
    
    # Get dimensions
    n_x, n_y, n_z = len(x_centers), len(y_centers), len(z_centers)
    shape = (n_x, n_y, n_z)
    
    print(f"  Grid shape: {shape}")
    print(f"  Total cells: {n_x * n_y * n_z:,}")
    
    # Initialize empty arrays
    residence_time = np.zeros(shape, dtype=config.FLOAT_DTYPE)
    burst_count = np.zeros(shape, dtype=config.INT_DTYPE)
    burst_time = np.zeros(shape, dtype=config.FLOAT_DTYPE)
    probability = np.zeros(shape, dtype=config.FLOAT_DTYPE)
    
    # Create Xarray Dataset
    grid = xr.Dataset(
        data_vars={
            'residence_time': (
                ['x', 'y', 'z'], 
                residence_time,
                {
                    'units': 'seconds',
                    'long_name': 'Total time spacecraft spent in cell',
                    'description': 'Denominator for probability calculation'
                }
            ),
            'burst_count': (
                ['x', 'y', 'z'], 
                burst_count,
                {
                    'units': 'count',
                    'long_name': 'Number of AKR bursts detected',
                    'description': 'Total number of burst events in cell'
                }
            ),
            'burst_time': (
                ['x', 'y', 'z'], 
                burst_time,
                {
                    'units': 'seconds',
                    'long_name': 'Total time detecting AKR in cell',
                    'description': 'Numerator for probability calculation'
                }
            ),
            'probability': (
                ['x', 'y', 'z'], 
                probability,
                {
                    'units': 'percent',
                    'long_name': 'AKR detection probability',
                    'description': 'Probability = (burst_time / residence_time) * 100'
                }
            ),
        },
        coords={
            'x': (
                ['x'], 
                x_centers,
                {
                    'units': 'R_E',
                    'long_name': 'X-GSE (toward Sun)',
                    'description': 'GSE X coordinate (Earth radii)'
                }
            ),
            'y': (
                ['y'], 
                y_centers,
                {
                    'units': 'R_E',
                    'long_name': 'Y-GSE (duskward)',
                    'description': 'GSE Y coordinate (Earth radii)'
                }
            ),
            'z': (
                ['z'], 
                z_centers,
                {
                    'units': 'R_E',
                    'long_name': 'Z-GSE (northward)',
                    'description': 'GSE Z coordinate (Earth radii)'
                }
            ),
        },
        attrs={
            'title': 'AKR Detection Probability Grid',
            'coordinate_system': config.COORDINATE_SYSTEM,
            'bin_size': float(config.BIN_SIZE),
            'x_range': config.X_RANGE,
            'y_range': config.Y_RANGE,
            'z_range': config.Z_RANGE,
            'creation_date': np.datetime64('now').astype(str),
            'description': 'Three-dimensional grid for mapping AKR detection probability in Earth magnetosphere',
            'conventions': 'CF-1.8',  # Climate and Forecast metadata convention
        }
    )
    
    print("✅ Grid created successfully!")
    return grid

# ==============================================================================
# STEP 3: FILL WITH RANDOM TEST DATA
# ==============================================================================

def fill_with_random_data(grid: xr.Dataset, seed: int = 42) -> xr.Dataset:
    """
    Fill grid with realistic random test data.
    
    This creates a physically plausible distribution:
    - Higher probability on nightside (X < 0)
    - Enhanced at auroral latitudes (|Z| ~ 5-7 R_E)
    - Decreases outside magnetosphere (R > 10 R_E)
    
    Parameters
    ----------
    grid : xr.Dataset
        Empty grid to fill
    seed : int, default 42
        Random seed for reproducibility
    
    Returns
    -------
    xr.Dataset
        Grid filled with test data
    """
    
    print("\nFilling grid with random test data...")
    np.random.seed(seed)
    
    # Create coordinate meshes
    X, Y, Z = np.meshgrid(
        grid.x.values, 
        grid.y.values, 
        grid.z.values, 
        indexing='ij'
    )
    
    # Calculate radial distance from Earth
    R = np.sqrt(X**2 + Y**2 + Z**2)
    
    # === Create Realistic Probability Distribution ===
    
    # 1. Nightside enhancement (AKR stronger on nightside)
    nightside_factor = np.exp(-X / 5.0)  # Peaks at X < 0
    
    # 2. Auroral latitude enhancement (peak around |Z| = 5-7 R_E)
    auroral_factor = np.exp(-((np.abs(Z) - 6.0) / 2.5)**2)
    
    # 3. Distance decay (probability drops outside magnetosphere)
    # Magnetopause typically at ~10 R_E on dayside, ~15 R_E on nightside
    distance_factor = np.exp(-(R - 6.5) / 3.0)
    
    # 4. Local time asymmetry (higher probability 18-06 MLT)
    # Approximate local time from Y coordinate
    local_time_angle = np.arctan2(Y, X)
    local_time_factor = 0.8 + 0.2 * np.cos(local_time_angle - np.pi)  # Peak at midnight
    
    # Combine all factors
    probability = (
        40.0 *  # Base probability
        nightside_factor * 
        auroral_factor * 
        distance_factor * 
        local_time_factor +
        np.random.rand(*X.shape) * 8.0  # Add noise
    )
    
    # Clip to valid range [0, 100]
    probability = np.clip(probability, 0, 100)
    
    # Set to zero very close to Earth (< 3 R_E) and far away (> 12 R_E)
    probability[R < 3.0] = 0
    probability[R > 12.0] = 0
    
    # === Generate Corresponding Auxiliary Data ===
    
    # Residence time (random but realistic: 100-10000 seconds per cell)
    residence_time = np.random.uniform(100, 10000, probability.shape).astype(np.float32)
    
    # Only have residence time where we have probability
    residence_time[probability == 0] = 0
    
    # Burst time (derived from probability and residence time)
    burst_time = (probability / 100.0) * residence_time
    
    # Burst count (Poisson-distributed, related to probability)
    burst_count = np.random.poisson(probability / 5.0).astype(np.int32)
    
    # Update grid
    grid['probability'].values = probability.astype(np.float32)
    grid['residence_time'].values = residence_time
    grid['burst_time'].values = burst_time
    grid['burst_count'].values = burst_count
    
    # Print statistics
    mask = probability > 0
    print(f"  Cells with data: {mask.sum():,} / {probability.size:,} ({mask.sum()/probability.size*100:.1f}%)")
    print(f"  Probability range: {probability[mask].min():.1f}% - {probability[mask].max():.1f}%")
    print(f"  Mean probability: {probability[mask].mean():.1f}%")
    print(f"  Total bursts: {burst_count.sum():,}")
    
    print("✅ Grid filled with test data!")
    return grid

# ==============================================================================
# STEP 4: 3D VISUALIZATION WITH PLOTLY
# ==============================================================================

def visualize_3d_grid(
    grid: xr.Dataset, 
    threshold: float = 10.0,
    save_html: bool = True,
    output_file: str = 'akr_3d_grid.html'
) -> go.Figure:
    """
    Create interactive 3D visualization of the grid.
    
    Parameters
    ----------
    grid : xr.Dataset
        Grid to visualize
    threshold : float, default 10.0
        Only show cells with probability > threshold
    save_html : bool, default True
        Save interactive HTML file
    output_file : str, default 'akr_3d_grid.html'
        Output filename
    
    Returns
    -------
    go.Figure
        Plotly figure object
    """
    
    print(f"\nCreating 3D visualization (threshold: {threshold}%)...")
    
    # Get high-probability cells
    prob = grid['probability'].values
    mask = prob > threshold
    
    if mask.sum() == 0:
        print(f"⚠️  No cells above threshold {threshold}%")
        return None
    
    print(f"  Showing {mask.sum():,} cells above threshold")
    
    # Get coordinates of high-probability cells
    X, Y, Z = np.meshgrid(grid.x, grid.y, grid.z, indexing='ij')
    
    x_flat = X[mask]
    y_flat = Y[mask]
    z_flat = Z[mask]
    prob_flat = prob[mask]
    
    # Create figure
    fig = go.Figure()
    
    # Add probability data as 3D scatter
    fig.add_trace(go.Scatter3d(
        x=x_flat,
        y=y_flat,
        z=z_flat,
        mode='markers',
        marker=dict(
            size=4,
            color=prob_flat,
            colorscale='Hot',  # Red-yellow colormap
            colorbar=dict(
                title='AKR Detection<br>Probability (%)',
                x=1.02,
                len=0.8
            ),
            opacity=0.7,
            showscale=True,
            cmin=0,
            cmax=100
        ),
        name='AKR Probability',
        text=[
            f'Position: ({x:.1f}, {y:.1f}, {z:.1f}) R_E<br>'
            f'Probability: {p:.1f}%'
            for x, y, z, p in zip(x_flat, y_flat, z_flat, prob_flat)
        ],
        hovertemplate='%{text}<extra></extra>',
        showlegend=False
    ))
    
    # Add Earth sphere (radius = 1 R_E)
    u = np.linspace(0, 2 * np.pi, 40)
    v = np.linspace(0, np.pi, 40)
    x_earth = np.outer(np.cos(u), np.sin(v))
    y_earth = np.outer(np.sin(u), np.sin(v))
    z_earth = np.outer(np.ones(np.size(u)), np.cos(v))
    
    fig.add_trace(go.Surface(
        x=x_earth,
        y=y_earth,
        z=z_earth,
        colorscale=[[0, 'lightblue'], [1, 'blue']],
        showscale=False,
        opacity=0.9,
        name='Earth',
        hoverinfo='name'
    ))
    
    # Add Sun direction arrow
    sun_distance = 18
    fig.add_trace(go.Cone(
        x=[sun_distance],
        y=[0],
        z=[0],
        u=[3],
        v=[0],
        w=[0],
        colorscale=[[0, 'yellow'], [1, 'orange']],
        showscale=False,
        name='Sun Direction',
        hoverinfo='name'
    ))
    
    # Add coordinate axes
    axis_length = 16
    
    # X-axis (to Sun) - Red
    fig.add_trace(go.Scatter3d(
        x=[0, axis_length], y=[0, 0], z=[0, 0],
        mode='lines+text',
        line=dict(color='red', width=6),
        text=['', '+X (Sun) →'],
        textposition='top center',
        textfont=dict(size=12, color='red'),
        name='X-axis (Sun)',
        showlegend=False,
        hoverinfo='name'
    ))
    
    # Y-axis (Dusk) - Green
    fig.add_trace(go.Scatter3d(
        x=[0, 0], y=[0, axis_length], z=[0, 0],
        mode='lines+text',
        line=dict(color='green', width=6),
        text=['', '+Y (Dusk) →'],
        textposition='top center',
        textfont=dict(size=12, color='green'),
        name='Y-axis (Dusk)',
        showlegend=False,
        hoverinfo='name'
    ))
    
    # Z-axis (North) - Blue
    fig.add_trace(go.Scatter3d(
        x=[0, 0], y=[0, 0], z=[0, axis_length],
        mode='lines+text',
        line=dict(color='blue', width=6),
        text=['', '+Z (North) ↑'],
        textposition='top center',
        textfont=dict(size=12, color='blue'),
        name='Z-axis (North)',
        showlegend=False,
        hoverinfo='name'
    ))
    
    # Add magnetopause boundary (simplified)
    theta = np.linspace(0, 2*np.pi, 50)
    phi = np.linspace(0, np.pi, 25)
    THETA, PHI = np.meshgrid(theta, phi)
    
    # Asymmetric magnetopause (compressed on dayside)
    R_mp = 10 + 3 * np.cos(PHI)
    X_mp = R_mp * np.sin(PHI) * np.cos(THETA)
    Y_mp = R_mp * np.sin(PHI) * np.sin(THETA)
    Z_mp = R_mp * np.cos(PHI) * 0.7  # Flattened
    
    fig.add_trace(go.Surface(
        x=X_mp, y=Y_mp, z=Z_mp,
        colorscale=[[0, 'rgba(255,215,0,0.15)'], [1, 'rgba(255,215,0,0.15)']],
        showscale=False,
        name='Magnetopause',
        opacity=0.15,
        hoverinfo='name'
    ))
    
    # Update layout
    fig.update_layout(
        title=dict(
            text='AKR Detection Probability Map - 3D View<br><sub>Test Data (Random)</sub>',
            x=0.5,
            xanchor='center'
        ),
        scene=dict(
            xaxis=dict(
                title='X-GSE [R_E] → Sun',
                range=[-15, 18],
                showgrid=True,
                gridcolor='lightgray',
                backgroundcolor='rgb(230, 230, 250)'
            ),
            yaxis=dict(
                title='Y-GSE [R_E] → Dusk',
                range=[-15, 15],
                showgrid=True,
                gridcolor='lightgray',
                backgroundcolor='rgb(230, 230, 250)'
            ),
            zaxis=dict(
                title='Z-GSE [R_E] → North',
                range=[-15, 15],
                showgrid=True,
                gridcolor='lightgray',
                backgroundcolor='rgb(230, 230, 250)'
            ),
            aspectmode='manual',
            aspectratio=dict(x=1.1, y=1, z=1),
            bgcolor='rgb(240, 240, 255)',
            camera=dict(
                eye=dict(x=1.5, y=1.5, z=1.2)
            )
        ),
        width=1200,
        height=900,
        showlegend=False,
        hovermode='closest'
    )
    
    # Save HTML
    if save_html:
        fig.write_html(output_file)
        print(f"✅ Saved interactive 3D plot to: {output_file}")
        print(f"   Open in browser to rotate, zoom, and explore!")
    
    return fig

# ==============================================================================
# STEP 5: 2D SLICE VISUALIZATIONS
# ==============================================================================

def create_2d_slices(grid: xr.Dataset, save_file: str = 'akr_2d_slices.html'):
    """Create 2D slice views (top, side, front)."""
    
    print("\nCreating 2D slice views...")
    
    import plotly.subplots as sp
    
    # Get middle slices
    x_mid = len(grid.x) // 2
    y_mid = len(grid.y) // 2
    z_mid = len(grid.z) // 2
    
    prob = grid['probability'].values
    
    # Create subplots
    fig = sp.make_subplots(
        rows=2, cols=2,
        subplot_titles=(
            'Top-Down View (X-Y, Z=0)',
            'Side View (X-Z, Y=0)',
            'Front View (Y-Z, X=0)',
            'Probability Distribution'
        ),
        specs=[
            [{'type': 'heatmap'}, {'type': 'heatmap'}],
            [{'type': 'heatmap'}, {'type': 'histogram'}]
        ]
    )
    
    # 1. Top-down view (X-Y)
    fig.add_trace(
        go.Heatmap(
            x=grid.x.values,
            y=grid.y.values,
            z=prob[:, :, z_mid].T,
            colorscale='Hot',
            zmin=0, zmax=80,
            colorbar=dict(x=0.46, len=0.4, y=0.75)
        ),
        row=1, col=1
    )
    
    # 2. Side view (X-Z)
    fig.add_trace(
        go.Heatmap(
            x=grid.x.values,
            y=grid.z.values,
            z=prob[:, y_mid, :].T,
            colorscale='Hot',
            zmin=0, zmax=80,
            colorbar=dict(x=1.02, len=0.4, y=0.75)
        ),
        row=1, col=2
    )
    
    # 3. Front view (Y-Z)
    fig.add_trace(
        go.Heatmap(
            x=grid.y.values,
            y=grid.z.values,
            z=prob[x_mid, :, :].T,
            colorscale='Hot',
            zmin=0, zmax=80,
            colorbar=dict(x=0.46, len=0.4, y=0.25)
        ),
        row=2, col=1
    )
    
    # 4. Histogram
    fig.add_trace(
        go.Histogram(
            x=prob.flatten(),
            nbinsx=50,
            marker_color='indianred',
            name='Probability'
        ),
        row=2, col=2
    )
    
    # Update axes
    fig.update_xaxes(title_text="X-GSE [R_E]", row=1, col=1)
    fig.update_yaxes(title_text="Y-GSE [R_E]", row=1, col=1)
    fig.update_xaxes(title_text="X-GSE [R_E]", row=1, col=2)
    fig.update_yaxes(title_text="Z-GSE [R_E]", row=1, col=2)
    fig.update_xaxes(title_text="Y-GSE [R_E]", row=2, col=1)
    fig.update_yaxes(title_text="Z-GSE [R_E]", row=2, col=1)
    fig.update_xaxes(title_text="Probability (%)", row=2, col=2)
    fig.update_yaxes(title_text="Count", row=2, col=2)
    
    # Update layout
    fig.update_layout(
        title_text="AKR Detection Probability - 2D Slices (Test Data)",
        height=900,
        width=1200,
        showlegend=False
    )
    
    fig.write_html(save_file)
    print(f"✅ Saved 2D slices to: {save_file}")
    
    return fig

# ==============================================================================
# STEP 6: SAVE GRID DATA
# ==============================================================================

def save_grid(grid: xr.Dataset, filename: str = 'akr_grid_test.nc'):
    """Save grid to NetCDF file."""
    
    print(f"\nSaving grid to {filename}...")
    
    # Add compression
    encoding = {
        var: {'zlib': True, 'complevel': 4, 'dtype': 'float32'}
        for var in grid.data_vars if grid[var].dtype == np.float32
    }
    
    grid.to_netcdf(filename, encoding=encoding)
    
    file_size = Path(filename).stat().st_size / 1e6
    print(f"✅ Saved grid ({file_size:.2f} MB)")

# ==============================================================================
# MAIN EXECUTION
# ==============================================================================

def main():
    """Main execution function."""
    
    print("="*70)
    print(" AKR 3D GRID CREATION - TEST WITH RANDOM DATA")
    print("="*70)
    
    # Step 1: Configure grid
    config = GridConfig()
    print(config)
    
    # Step 2: Create empty grid
    grid = create_3d_grid(config)
    
    # Step 3: Fill with random data
    grid = fill_with_random_data(grid, seed=42)
    
    # Step 4: Visualize in 3D
    fig_3d = visualize_3d_grid(grid, threshold=15, save_html=True, output_file='data/processed/akr_3d_grid.html')
    
    # Step 5: Create 2D slices
    fig_2d = create_2d_slices(grid, save_file='data/processed/akr_2d_slices.html')
    
    # Step 6: Save grid data
    save_grid(grid, filename='data/processed/akr_grid_test.nc')
    
    print("\n" + "="*70)
    print(" ✅ COMPLETE! ")
    print("="*70)
    print("\nGenerated files:")
    print("  1. data/processed/akr_3d_grid.html     - Interactive 3D visualization")
    print("  2. data/processed/akr_2d_slices.html   - 2D slice views")
    print("  3. data/processed/akr_grid_test.nc     - Grid data (NetCDF format)")
    print("\nNext steps:")
    print("  - Open HTML files in browser to explore")
    print("  - Discuss grid parameters with supervisor")
    print("  - Replace random data with real burst catalog")
    print("="*70)

if __name__ == '__main__':
    from pathlib import Path
    
    # Create output directory if needed
    Path('data/processed').mkdir(parents=True, exist_ok=True)
    
    # Run
    main()