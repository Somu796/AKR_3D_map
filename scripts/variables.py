from typing import Annotated

from pydantic import Field

type NumericType = int | float
PositiveNumber = Annotated[NumericType, Field(gt=0)]

# %% Number of coordinates
n_coord_colnames = 3
