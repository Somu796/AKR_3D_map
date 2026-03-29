import sys
from pathlib import Path

import pandas as pd


def main() -> None:
    project_root = Path.cwd()
    sys.path.append(str(project_root))

    # Define standard data subdirectories
    ASSETS_DIR = project_root / "assets" / "3D_Objects_03"

    # Load data
    df = pd.read_parquet(path=f"{ASSETS_DIR}/cart_akr_grid.parquet")

    # We must wrap the operation in print() to see it in terminal
    print(df.sort_values(by=["normalised_observation_time"], ascending=False).head(5))


if __name__ == "__main__":
    main()
