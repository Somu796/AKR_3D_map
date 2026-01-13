#%%
# Libraries
import numpy as np
import xarray as xr
import plotly.graph_objects as go

type NumericType = int | float # Customised type hint
#%%
# Function to create 3D grid
def create_3d_grid(
        x_range: tuple[NumericType, NumericType] = (-15, 15),
        y_range: tuple[NumericType, NumericType] = (-15, 15),
        z_range: tuple[NumericType, NumericType] = (-10, 10),
        bin_size: NumericType=0.5,
)-> xr.Dataset:
    """"""
        
        # Creating edges
    x_edge: np.ndarray = np.arange(x_range[0], x_range[1], bin_size)
    y_edge: np.ndarray = np.arange(y_range[0], y_range[1], bin_size)
    z_edge: np.ndarray = np.arange(z_range[0], z_range[1], bin_size)

    # Creating centers
    x_centers: np.ndarray = (x_edge[:-1]+x_edge[1:])/2
    y_centers: np.ndarray = (y_edge[:-1]+y_edge[1:])/2
    z_centers: np.ndarray = (z_edge[:-1]+z_edge[1:])/2

    # Axis lengths
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
        "x_edge": (['x_edge'], x_edge, {'units': 'R_E', 'dtype': 'float64'}),
        "y_edge": (['y_edge'], y_edge, {'units': 'R_E', 'dtype': 'float64'}),
        "z_edge": (['z_edge'], z_edge, {'units': 'R_E', 'dtype': 'float64'}) 
        },
        attrs ={
        'coordinate_system': 'GSE',
        'units': 'Earth_radii',
        'bin_size': float(bin_size),
        'description': 'AKR detection probability grid'            
        }
    )
    return grid
#%%
# Create the grid
grid = create_3d_grid(
    x_range=(-15.0, 15.0),
    y_range=(-15.0, 15.0),
    z_range=(-10.0, 10.0),
    bin_size=2.0
)

print(f"Grid created: {grid.dims}")
print(f"Shape: {grid.residence_time.shape}")

# Access Edges
# bin_size = grid.attrs['bin_size']
x_edges = grid.x_edge.values
y_edges = grid.y_edge.values
z_edges = grid.z_edge.values

#%%
# Create figure
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

# # Add Sun (positioned along positive X-axis, typical GSE orientation)
# sun=True
# sun_distance = -150  # Sun at ~150 R_E along -X axis (sunward direction)
# sun_radius = 5  # Visual size

# fig.add_trace(go.Surface(
#     x=x_sphere * sun_radius + sun_distance,
#     y=y_sphere * sun_radius,
#     z=z_sphere * sun_radius,
#     colorscale=[[0, 'yellow'], [1, 'orange']],
#     showscale=False,
#     opacity=0.8,
#     name='Sun'
# ))

# Layout
fig.update_layout(
    # ========== GLOBAL FONT ==========
    font=dict(
        family="Times New Roman, Times, serif",
        size=14,
        color="#1a1a1a"
    ),
    
    # ========== TITLE ==========
    title=dict(
        text='3D Grid with Earth and Sun' if (sun and earth) else '3D Grid of Earth',
        font=dict(size=22),
        x=0.5,
        xanchor='center'
    ),
    
    # ========== 3D SCENE ==========
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
        
        # ========== FIXED CAMERA (X horizontal, Z vertical) ==========
        camera=dict(
            eye=dict(x=0.3, y=2.5, z=0.8),  # View from Y-axis, slightly elevated
            center=dict(x=0, y=0, z=0),
            up=dict(x=0, y=0, z=1)          # Z points UP
        ),
        
        # ========== PROPER SCALING ==========
        aspectmode='data',  # True scale based on data ranges
        
        # Optional: Add this for better interaction
        dragmode='orbit'
    ),
    
    # ========== SIZE & BACKGROUND ==========
    width=1000,
    height=800,
    paper_bgcolor='white',
    margin=dict(l=0, r=0, t=70, b=0)
)

fig.show()
# %%
# Remove all environmenatal parameters
%reset -f