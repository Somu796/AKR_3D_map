<h1 align="center"> 📡 🌍 AKR 3D Map 📻 🌐 </h1>

# Work in Progress as of Dec 2025

Here we provide code to read in various AKR event lists, and map their occurrence in the near-Earth space environment.

# Acknowledgements

* SH's work at DIAS was supported by a SCOSTEP PRESTO Database Construction grant entitled "AKR as a Barometer for Space Weather: a new, interactive map".
* [ARF](https://github.com/arfogg)'s work at DIAS was supported by Taighde Éireann - Research Ireland Laureate Consolidator award SOLMEX to [CMJ](https://github.com/caitrionajackman).

<p align="center">
<img src="assets/SCOSTEP_logo.png" width="200">
<img src="assets/PRESTO_logo.png" width="100">
<img src="assets/Research_Ireland_RGB_logo_green.webp" width="200">
</p>

# Thinking process

CSV File (Python conversion) ->  TFCat JSON (dates as ISO strings) -> (MongoDB insert with conversion) -> MongoDB BSON (dates as ISODate objects) (Application query) ->  Python datetime objects / JavaScript Date objects -> (Analysis/Display)

# Good Practices will be followed all over

It should be.

1. OOP
2. Type check
3. Write pytests
4. Try and Catch error handling.

# Calculations

## Residence time

1. (Done) Clean data put as numeric or datetime in a list object, from "," separated text string
2. (Done) Assign corresponding data types like datetime and float
3. (Done) Explode the data to remove Nan entries, exploding data help with optimised computation (less RAM usage)
4. Calculate Time Intervals: tn-1 - tn, where n = sttime_index+1
5. Find corresponding coordinates

```py
# Auto-determine boundaries from data
cart = (Cartesian()
        .decide_boundaries(df)      # Analyze data, set ranges
        .create_grid()               # Create grid structure
        .calculate_residence_time(df) # Calculate and store
        .plot_3d(variable='residence_time'))  # Visualize

# Or just plot grid structure
cart = (Cartesian(x_range=(-15, 15), bin_size=2.0)
        .create_grid()
        .plot_3d())  # No variable → just wireframe

# Or plot after manual calculation
cart.plot_3d(variable='residence_time')  # Plot the data

```

# Folder Structure

```md
Akr3Map_project/
├── .venv/                      # Virtual environment
├── data/
│   ├── raw/                    # Original fogg_akr_burst_list CSV
│   └── processed/              # Cleaned Parquet files
├── 3D_Objects/                 # Output folder for .html and .json plots
├── scripts/
│   ├── __init__.py             # Makes scripts a package
│   ├── grid_3d.py              # Cartesian and LTRMLat classes
│   ├── residence.py            # Residence time & interval calculations
│   └── utils.py                # Binning, plotting, and type definitions
├── main.py                     # Your primary execution script
└── pyproject.toml              # (Optional) For project dependencies
```

# To be done

1. Colorscale, need to be better.
2. Earth 3D image failed to add.
3. Also better to add the orbit of the satellite. To see better.
4. Camera angle should be better.
5. Add data layer as function so can be performed for many variables then name, and color need to be checked.
