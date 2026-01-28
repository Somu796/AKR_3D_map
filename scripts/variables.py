from typing import Annotated

from pydantic import Field

type NumericType = int | float
PositiveNumber = Annotated[NumericType, Field(gt=0)]

# %% Number of coordinates
n_coord_colnames = 3
padding_grid = 0.01  # Default padding for grid boundaries

# %% Datframe column names
burst_id_colname = "original_burst_id"
burst_timestamp_colname = "burst_timestamp"
time_interval_colname = "time_interval"
