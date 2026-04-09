<h1 align="center"> 📡 🌍 AKR 3D Map 📻 🌐 </h1>

# Project Details

<p align="center">
<img src="assets/SCOSTEP_logo.png" width="200">
<img src="assets/PRESTO_logo.png" width="100">
<img src="assets/Research_Ireland_RGB_logo_green.webp" width="200">
</p>

To understand the scientific background of the project and code implementation, please follow the [documentation website](https://somu796.github.io/AKR_3D_map/) and the [objective slides](https://somu796.github.io/AKR_3D_map/slides.html).

During the project several directories has been created. The below details will give you idea about each.

* `akr_3d_map/`: The main OOP code for processing AKR Burst and Residence data lies in `akr_3d_map/`. It has,
        - `base_class.py`: An ABC base class for reating grid.
        - `grid_3d.py`: It is the implementation of the `base_class.py`.
        - `mixins/observation_time.py`: It has implementation of all features that were calculated.
        - `utils.py`: It contains supportive functions.
        - `variables.py`: It contains all constants and data classes.

* `assets/`: It contains all the outcomes of the project.
        - `3D_Objects_03/`: It has all the visuals and corresponding calculated dataset. During the project multiple version of `3D_Objects_XX` were created. Please consider `3D_Objects_03/` as the final outcomes.

        - `gaussian_processes_model_checkpoints/`: It has model checkpoint and elbow plot for the gaussian process modeling. However it didn't fit well.

```markdown
somu796-akr_3d_map/
├── README.md # Current File
├── _quarto.yml
├── app.py # Running calculations in HPC
├── check_akr_grid_change.py 
├── decision_tree.py 
├── index.qmd # Main page for qmd website
├── index.quarto_ipynb
├── LICENSE
├── modeling.py # Running the Gaussian Processes Modeling
├── pyproject.toml
├── references.bib # Reference file for Quarto Website
├── requirements.txt
├── slides.qmd # Quarto Slides for the Project 
├── style.css # Custom CSS for website
├── todo.md # Left to be worked upon
├── .python-version 
├── akr_3d_map/
│   ├── README.md
│   ├── __init__.py
│   ├── base_class.py # parent ABC class for cart and ltrmlat child class
│   ├── calculations.py # Previous implementation of features which now implemented as mixin
│   ├── grid_3d.py # Child class with implementation of parent class
│   ├── utils.py # Supporting functions can be converted to classes and used as composition
│   ├── variables.py # Contains the constants and data classes
│   ├── wind_data_reading.py # Reading raw data
│   └── mixins/
│       ├── README.md
│       ├── __init__.py
│       └── observation_time.py # Mixin class calculates all the features
├── assets/
│   ├── 3D_Objects/
│   │   └── cart_akr_grid.parquet
│   ├── 3D_Objects_02/
│   │   └── readme.txt
│   ├── 3D_Objects_03/ # This should be considered as the main file for accessing processed data and 3D visuals
│   │   └── readme.txt
│   └── gaussian_processes_model_checkpoints/
│       ├── checkpoint
│       ├── ckpt-9.index
│       └── scaler.pkl
├── docs/
└── .github/
└── workflows/
        └── quarto-publish.yml
```

# Future Implementations

Later on the `akr_3d_map/` library/class can be extended to use TFCat as input instead of dataframe in `.parquet` format. The sole reason of using `.parquet` for the current project is the space optimisation capability of the format and easy to handle features of datatables instead of `.json` files.

# Acknowledgements

* Sudipta's work at DIAS was supported by a SCOSTEP PRESTO Database Construction grant entitled "AKR as a Barometer for Space Weather: a new, interactive map".
* [ARF](https://github.com/arfogg)'s work at DIAS was supported by Taighde Éireann - Research Ireland Laureate Consolidator award SOLMEX to [CMJ](https://github.com/caitrionajackman).
