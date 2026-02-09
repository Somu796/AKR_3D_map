from typing import Annotated

from pydantic import Field

type NumericType = int | float
PositiveNumber = Annotated[NumericType, Field(gt=0)]

# %% Number of coordinates
n_coord_colnames = 3
padding_grid = 0.01  # Default padding for grid boundaries
background_color = "#1d1d1d"  # "#f5f5f5"
grid_color = "#313131"  # "#f5f5f5"

earth_image_path_str = "assets/temp.jpg"

# %% Datframe column names
burst_id_colname = "original_burst_id"
time_interval_colname = "time_interval"
