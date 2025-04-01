# Morphology Measurement Methods

## Setup
```bash
# Install Miniconda
# First, install Miniconda on your system. You can download it from the Miniconda website.

# Create a Conda Environment
conda create --name myenv python=3.12

# Activate Conda Environment
conda activate myenv

# Install Required Packages
pip install -r requirements.txt
```

In general python (.py files) require access to simualtion data, while Jupyter Notebooks (.ipynb files) are used for data analysis and visualization.


## Code Directory
### Config.py
Setups folder structure and simulation information for the analysis.
Edit the simulation dictionary files in Code/SimInfoDicts to change simualation information, such as file paths, feedback models, and simulation names.

### CollectAll.py
Runs the specified script on all simulations in the simulation dictionary files in Code/SimInfoDicts.

### PlotClasses.py
Contains classes for generating plots in the Jupyter Notebooks, i.e. StellarDMTracing.ipynb

## Images and profies
Scripts in the "IsophoteImaging" directory are responsible for generating the mock-observation V-band images.

### ImageCollection.py
Can be automatically run on all sims via CollectAll.py. Requires initialization of Image directory through Config.py.<br>

**Description**: Generates v-band luminosity density images for visual inspection. Also generates luminosity profile data. Images/Profiles are generated at all orientations. 

**Required Arguments**<br>
\-s/--simulation: Name of simulation to analyze (as defined in Config.py)<br>
\-f/--feedback: Feedback model of simulation<br>
\-n/--numproc: The number of processers to use for multiprocessing<br>

**Optional Flags**<br>
\-i/--image: Generate v-band images in addition to profile data (will take significantly longer to run)<br>
\-o/--overwrite: Overwrite existing data<br>
\-v/--verbose: Output halo ID's as they're being analyzed<br>

**Outputs**<br>
Generates "Profiles" files for each sim in the "Data" directory. If the "--image" flag is given, it will also generate "Images" files and Images for each halo in the "Figures/Images" directory.

### Halo properties
GetMasses.py in the "HaloProperties" directory is responsible for calculating halo properties such as masses, dynamical time, and Jz_Jcirc.


### Intrinsic Shape Calculation
Scripts in the "IntrinsicShapes" directory are responsible for calculating intrinsic 3D shapes of our galaxies and dark matter halos. These shapes can be projected into 2D at various orientations for direct comparison to isophote measurements. They should be run in the following sequence:<br>


### 3DShapeCollection.py
Can be automatically run on all sims via CollectAll.py.

**Description**: Calculate the 3D axis-rations B/A and C/A as a function of radius based on shape tensor calculations. 

**Required Arguments**<br>
\-s/--simulation: Name of simulation to analyze (as defined in Config.py)<br>
\-f/--feedback: Feedback model of simulation<br>
\-n/--numproc: The number of processers to use for multiprocessing<br>

**Optional Flags**<br>
\-v/--verbose: Output halo ID's as they're being analyzed<br>

**Outputs**<br>
Generates 3DShapes/DMShapes files for the given simulation giving 3D ellipsoid fits as a function of radius.

### 3DShapeSmoothing.ipynb
Requires 3DShapeCollection to be run on both Stars and Dark Matter

**Description**: Smoothes the B/A and C/A radial profiles for each galaxy.

**Outputs**<br>
Updates the 3DShapes and DMShapes dictionary files with new keys for smoothed profiles. Also generates images showing the raw and smoothed radial profiles in each simulation's subdirectory in "Figures/3DShapes".


### Orientations.ipynb
Calculates differences in angles between the major axes of the 3D shapes and the angular momentum vectors of the stars and dark matter near 2 effective radii.

### StellarDMTracing.ipynb
Primary notebook for generating figures in the paper. Requires 3DShapeCollection 
### StellarDMTracing.py
Helper functions for loading and plotting data in StellarDMTracing.ipynb

### Tangos
Code in the tangos directory is used to extract merger information and generate merger plots


# Data License

[![CC BY 4.0][cc-by-shield]][cc-by]

The data in this repository is licensed under a
[Creative Commons Attribution 4.0 International License][cc-by].

[![CC BY 4.0][cc-by-image]][cc-by]

This license covers all datasets, figure data, and processed results included in this repository. Core simulations and intermediate data remain proprietary and are not included in this license.

## Attribution Requirements

If you use this data in your research, please cite:

[10.48550/arXiv.2501.16317]: 10.48550/arXiv.2501.16317

And reference this data repository:

[Repository citation with DOI once deposited in Zenodo]

[cc-by]: http://creativecommons.org/licenses/by/4.0/
[cc-by-image]: https://i.creativecommons.org/l/by/4.0/88x31.png
[cc-by-shield]: https://img.shields.io/badge/License-CC%20BY%204.0-lightgrey.svg

