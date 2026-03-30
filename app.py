# %% Plotting on Complete Data
import sys
from pathlib import Path

import pandas as pd

from akr_3d_map.grid_3d import (
    Cartesian,
    LTRMLat,
)


def main() -> None:
    # 2. Setting project paths
    project_root = Path.cwd()
    sys.path.append(str(project_root))

    # Define standard data subdirectories for easy access later
    ASSETS_DIR = project_root / "assets" / "3D_Objects_03"

    # 3. Importing Data
    wind_data = pd.read_parquet(
        f"{project_root}/data/processed/01_processed_wind_data_fogg_akr_burst_list_1995_2004.parquet",
    )

    residence_data = pd.read_parquet(
        f"{project_root}/data/processed/06_processed_combined_wind_data_residence_time_1995_2004_high_res.parquet",
        # f"{project_root}/data/processed/05_processed_wind_data_residence_time_1995_2004_high_res.parquet",
        # f"{project_root}/data/processed/02_processed_wind_data_residence_time_1995_2004.parquet",
    )

    # Runing Code
    cart = (
        # Provide bin size for the 3D plot
        Cartesian()
        # In case you are not sure about the extreme points
        .decide_boundaries(wind_data, coord_colnames=("x_gse", "y_gse", "z_gse"))
        # Create grid points
        .create_grid()
        # Calculate burst count given wind satellite data
        .add_burst_count(
            wind_data,
            burst_id_colname="original_burst_id",
            coord_colnames=("x_gse", "y_gse", "z_gse"),
        )
        # Calculate observations time given wind satellite data
        .add_observation_time(
            wind_data,
            timestamp_colname="burst_timestamp",
            coord_colnames=("x_gse", "y_gse", "z_gse"),
            burst_id_colname="original_burst_id",
        )
        .add_observation_count(
            wind_data,
            coord_colnames=("x_gse", "y_gse", "z_gse"),
        )
        # Calculate residence time given wind satellite data
        .add_residence_time(
            residence_data,
            timestamp_colname="time_stamp",
            coord_colnames=("x_gse", "y_gse", "z_gse"),
        )
        .add_residence_count(
            df=residence_data,
            coord_colnames=("x_gse", "y_gse", "z_gse"),
            residence_timestamp_colname="time_stamp",
            gap_hours=0,  # we don't separate by pass time
        )
        # Calculate normalised observation time given wind satellite data
        .add_normalised_observation_time(
            akr_df=wind_data,
            satellite_residence_df=residence_data,
            coord_colnames=("x_gse", "y_gse", "z_gse"),
            akr_timestamp_colname="burst_timestamp",
            residence_timestamp_colname="time_stamp",
        )
    )

    cart.plot_3d(
        variable="burst_count",
        path=f"{ASSETS_DIR}/01_cartesian_grid_with_burst_counts.json",
        show_earth=True,
        colorscale = "Viridis", 
        earth_image_path_str=f"{project_root}/assets/temp.jpg",
        show_sun=False,
    )
    cart.plot_3d(
        variable="observation_time",
        path=f"{ASSETS_DIR}/02_cartesian_grid_with_observation_time.json",
        colorscale = "Viridis", 
        show_earth=True,
        earth_image_path_str=f"{project_root}/assets/temp.jpg",
        show_sun=False,
    )
    cart.plot_3d(
        variable="residence_time",
        path=f"{ASSETS_DIR}/03_cartesian_grid_with_residence_time.json",
        colorscale = "Viridis", 
        show_earth=True,
        earth_image_path_str=f"{project_root}/assets/temp.jpg",
        show_sun=False,
    )
    cart.plot_3d(
        variable="normalised_observation_time",
        path=f"{ASSETS_DIR}/04_cartesian_grid_with_normalised_observation_time.json",
        colorscale = "Viridis", 
        show_earth=True,
        earth_image_path_str=f"{project_root}/assets/temp.jpg",
        show_sun=False,
    )

    ltrmlat = (
        # Provide bin size for the 3D plot
        LTRMLat()
        # In case you are not sure about the extreme points
        .decide_boundaries(wind_data, coord_colnames=("LT_gse", "radius", "lat_gse"))
        # Create grid points
        .create_grid()
        # Calculate burst count (Unique Events)
        .add_burst_count(
            wind_data,
            burst_id_colname="original_burst_id",
            coord_colnames=("LT_gse", "radius", "lat_gse"),
        )
        # Calculate observations time (Duration in seconds)
        .add_observation_time(
            wind_data,
            timestamp_colname="burst_timestamp",
            coord_colnames=("LT_gse", "radius", "lat_gse"),
            burst_id_colname="original_burst_id",
        )
        # NEW: Total Point Count (Add 1 for every row)
        .add_observation_count(
            wind_data,
            coord_colnames=("LT_gse", "radius", "lat_gse"),
        )
        # Calculate residence time (Spacecraft total seconds)
        .add_residence_time(
            residence_data,
            timestamp_colname="time_stamp",
            coord_colnames=("LT_gse", "radius", "lat_gse"),
        )
        # NEW: Unique Spacecraft Passes (The "Once or Twice" logic)
        .add_residence_count(
            df=residence_data,
            coord_colnames=("LT_gse", "radius", "lat_gse"),
            residence_timestamp_colname="time_stamp",
            gap_hours=0,  # we don't separate by pass time
        )
        # Calculate normalised observation time (The Probability)
        .add_normalised_observation_time(
            akr_df=wind_data,
            satellite_residence_df=residence_data,
            coord_colnames=("LT_gse", "radius", "lat_gse"),
            akr_timestamp_colname="burst_timestamp",
            residence_timestamp_colname="time_stamp",
        )
    )

    ltrmlat.plot_3d(
        variable="burst_count",
        path=f"{ASSETS_DIR}/01_ltrmat_grid_with_burst_counts.json",
        colorscale = "Viridis", 
        show_earth=True,
        earth_image_path_str=f"{project_root}/assets/temp.jpg",
        show_sun=False,
    )
    ltrmlat.plot_3d(
        variable="observation_time",
        path=f"{ASSETS_DIR}/02_ltrmat_grid_with_observation_time.json",
        colorscale = "Viridis", 
        show_earth=True,
        earth_image_path_str=f"{project_root}/assets/temp.jpg",
        show_sun=False,
    )
    ltrmlat.plot_3d(
        variable="residence_time",
        path=f"{ASSETS_DIR}/03_ltrmat_grid_with_residence_time.json",
        colorscale = "Viridis", 
        show_earth=True,
        earth_image_path_str=f"{project_root}/assets/temp.jpg",
        show_sun=False,
    )
    ltrmlat.plot_3d(
        variable="normalised_observation_time",
        path=f"{ASSETS_DIR}/04_ltrmat_grid_with_normalised_observation_time.json",
        colorscale = "Viridis", 
        show_earth=True,
        earth_image_path_str=f"{project_root}/assets/temp.jpg",
        show_sun=False,
    )

    # Saving the Data
    cart.save_grid(path=f"{ASSETS_DIR}/cart_akr_grid.parquet", fmt="parquet")
    ltrmlat.save_grid(path=f"{ASSETS_DIR}/ltrmlat_akr_grid.parquet", fmt="parquet")
    print("--- Job Completed Successfully ---")


# %% Main section
if __name__ == "__main__":
    main()
