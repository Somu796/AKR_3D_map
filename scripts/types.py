# %% Customised type hint
from typing import Annotated

from pydantic import Field

type NumericType = int | float
PositiveNumber = Annotated[NumericType, Field(gt=0)]
