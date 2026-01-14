#%%
# Libraries
import numpy as np
import xarray as xr
import plotly.graph_objects as go
from pydantic import validate_call, Field
from typing_extensions import Union, Annotated
from dataclasses import dataclass

# Customised type hint
NumericType = Union[int, float] 
PositiveNumber = Annotated[NumericType, Field(gt=0)]

#%%
# Helper Function
## Function to create 1D grid

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
        >>> edges, centers = creates_bin1d(0, 10, 2)
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

# Create grid and visualisation
## Cartesian

@dataclass
class Cartesian():
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
     
    x_range: tuple[NumericType, NumericType] = (-15, 15)
    y_range: tuple[NumericType, NumericType] = (-15, 15)
    z_range: tuple[NumericType, NumericType] = (-10, 10)
    bin_size: NumericType=0.5
    grid: xr.Dataset | None = None

    def create_grid(self) -> 'Cartesian':
        """
        Creates grid for GSE coordinates.
        Returns:
            xarray Dataset with coordinates (x, y, z) containing 
            placeholder data variables for residence_time, burst_count, burst_time, 
            and probability.
        """

        # Creating edges
        x_edges, x_centers = creates_bin1d(float(self.x_range[0]), float(self.x_range[1]), float(self.bin_size))
        y_edges, y_centers = creates_bin1d(float(self.y_range[0]), float(self.y_range[1]), float(self.bin_size))
        z_edges, z_centers = creates_bin1d(float(self.z_range[0]), float(self.z_range[1]), float(self.bin_size))

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
        self.grid = xr.Dataset(
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
            'bin_size': float(self.bin_size),
            'description': 'AKR detection probability grid'            
            }
        )
        
        return self
    
    def plot_3d(self, show_earth: bool = True, show_sun: bool = False) -> 'Cartesian':
        """
        Plot the 3D Cartesian grid with wireframe.
        
        Args:
            show_earth: Whether to show Earth sphere
            show_sun: Whether to show Sun indicator
        """
        if self.grid is None:
            raise ValueError("Grid not created. Call create_grid() first!")
        # Access Edges
        x_edges = self.grid.x_edges.values
        z_edges = self.grid.z_edges.values
        y_edges = self.grid.y_edges.values

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

        # Creating sphere for earth and sun
        u = np.linspace(0, 2*np.pi, 50)
        v = np.linspace(0, np.pi, 50)
        x_sphere = np.outer(np.cos(u), np.sin(v))
        y_sphere = np.outer(np.sin(u), np.sin(v))
        z_sphere = np.outer(np.ones(np.size(u)), np.cos(v))
        
        # Add Earth sphere at origin
        if show_earth:
            fig.add_trace(go.Surface(
                x=x_sphere, y=y_sphere, z=z_sphere,
                colorscale='Blues', showscale=False,
                opacity=0.6, name='Earth'
            ))

        # Add Sun (positioned along positive X-axis, typical GSE orientation)
        if show_sun:
            sun_distance = -150  # GSE convention: +X points sunward, although the Sun itself lies at −X_GSE
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
                text='3D Grid with Earth and Sun' if (show_sun and show_earth) else '3D Grid of Earth',
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
        
        fig.show()
        self.fig = fig
        return self

# %% lt/r/mlat
@dataclass
class LTRMLat():
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
    r_range: tuple[NumericType, NumericType] = (0, 150)
    mlat_range: tuple[NumericType, NumericType] = (-90, 90)
    lt_bin: PositiveNumber = 1.0
    r_bin: PositiveNumber = 25.0
    mlat_bin: PositiveNumber = 5.0
    grid: xr.Dataset | None = None
           
    def create_grid(self) -> 'LTRMLat':
        """    
        Returns:
            xarray Dataset with coordinates (local_time, radius, mlat)
            containing placeholder data variables for residence_time, burst_count, 
            burst_time, and probability.
        """
        # Creating bins for each dimension
        lt_edges, lt_centers = creates_bin1d(float(self.lt_range[0]), float(self.lt_range[1]), float(self.lt_bin))
        r_edges, r_centers = creates_bin1d(float(self.r_range[0]), float(self.r_range[1]), float(self.r_bin))
        mlat_edges, mlat_centers = creates_bin1d(float(self.mlat_range[0]), float(self.mlat_range[1]), float(self.mlat_bin))

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
        
        # Creating the dataset
        self.grid: xr.Dataset = xr.Dataset(
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
                'lt_bin_width': float(self.lt_bin),
                'r_bin_width': float(self.r_bin),
                'mlat_bin_width': float(self.mlat_bin),
                'description': 'AKR detection probability grid in LT/R/MLat coordinates',
            }
        )
        return self

    # Create figure lt/r/mlat
    def plot_3d(self, show_earth: bool = True)-> 'LTRMLat':
        """
        Plot LT/R/MLat grid in 3D by converting to Cartesian.
        """
        if self.grid is None:
            raise ValueError("Grid not created. Call create_grid() first!")

        # Get coordinates
        lt = self.grid.coords['local_time'].values
        r = self.grid.coords['radius'].values  
        mlat = self.grid.coords['mlat'].values
        
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
        for r_val in self.grid.coords['radius_edges'].values:
            step = max(1, int(10 / self.mlat_bin))
            for mlat_val in self.grid.coords['mlat_edges'].values[::step]:
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
        for lt_val in self.grid.coords['local_time_edges'].values[::3]:
            for mlat_val in self.grid.coords['mlat_edges'].values[::5]:
                theta_val = (12 - lt_val) * np.pi / 12
                mlat_rad = np.radians(mlat_val)
                
                r_line = self.grid.coords['radius_edges'].values
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
        
        if show_earth:
            fig.add_trace(go.Surface(
                x=x_sphere, y=y_sphere, z=z_sphere,
                colorscale='Blues',
                showscale=False,
                opacity=0.6,
                name='Earth'
            ))
        
        fig.update_layout(
            title='LT/R/MLat Grid in 3D',
            font=dict(
                family="Times New Roman, Times, serif",
                size=14,
                color="#1a1a1a"
            ),
            scene=dict(
                xaxis_title='X (R<sub>E</sub>)',
                yaxis_title='Y (R<sub>E</sub>)',
                zaxis_title='Z (R<sub>E</sub>)',
                aspectmode='data',
                camera=dict(
                    eye=dict(x=1.5, y=1.5, z=1.2),
                    up=dict(x=0, y=0, z=1)
                )
            ),
            width=1000,
            height=800
        )
        fig.show()
        self.fig = fig
        return self

