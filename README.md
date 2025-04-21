# War-Induced Building Damage Detection with Machine Learning

This project aims to detect war-related building damage in Gaza using Synthetic Aperture Radar (SAR) imagery. This implementation follows a methodology similar to that presented inD ietrich, O., Peters, T., Sainte Fare Garnot, V., Sticher, V., Ton-That Whelan, T., Schindler, K., & Wegner, J. D. (2025). An open-source tool for mapping war destruction at scale in Ukraine using Sentinel-1 time series. Communications Earth & Environment, 6(1), 1–10. https://doi.org/10.1038/s43247-025-02183-7, adapted for the Gaza context.

## Project Overview

This project aims to develop a scalable method for estimating building damage in Gaza resulting from armed conflict using Sentinel-1 SAR imagery. The approach involves:

1. Analyzing time series of Sentinel-1 SAR imagery before and after conflict periods
2. Training machine learning models to detect changes indicative of structural damage
3. Generating probabilistic damage estimates at the building level
4. Validating results against available damage assessment data

## Data Sources

- **Sentinel-1 SAR Imagery**
- **Building Footprints**: Vector data defining building locations in Gaza
- **Damage Labels**: UNOSAT - Training data for supervised learning

## Project Structure

```
gaza-damage-detection/
├── data/
│   ├── raw/                # Raw unprocessed data
│   │   ├── sentinel/       # Raw Sentinel-1 data
│   │   ├── building_footprints/  # Building footprint data
│   │   └── labels/         # Damage labels/training data
│   ├── processed/          # Processed/cleaned data
│   │   ├── features/       # Extracted features from time series
│   │   └── timeseries/     # Processed time series data
│   └── validation/         # Validation datasets
├── notebooks/              # Jupyter notebooks for exploration and visualization
│   ├── 01_data_exploration.ipynb
│   ├── 02_feature_extraction.ipynb
│   ├── 03_model_comparison.ipynb
│   └── 04_results_visualization.ipynb
├── src/                    # Source code modules
│   ├── data/               # Data handling modules   
│   │   ├── download.py     # Functions to download/access Sentinel-1 data
│   │   └── preprocess.py   # Data preprocessing functions
│   ├── features/           # Feature extraction code  
│   │   └── timeseries.py   # Time series feature extraction
│   ├── models/             # Model implementations 
│   │   ├── random_forest.py
│   │   └── other_models.py # Alternative models to test
│   └── visualization/      # Visualization utilities
│       └── maps.py         # Map plotting functions
├── scripts/                # Utility scripts
│   ├── download_data.py    # Script to download data
│   ├── train_model.py      # Script to train models
│   └── generate_maps.py    # Script to generate final damage maps
├── tests/                  # Unit tests
│   ├── test_data.py
│   └── test_models.py
├── results/                # Output results and models
│   ├── figures/            # Generated figures
│   ├── maps/               # Generated damage maps
│   └── models/             # Saved model files
├── environment.yml         # Conda environment file
├── setup.py                # For packaging the project
├── README.md               # This file
└── LICENSE                 # Project license
```

## Getting Started

### Prerequisites

- Python 3.8+

### Installation
1. Clone this repository:

`git clone https://github.com/MatteoLarrode/ML-BuildingDamageViaSatellite?tab=readme-ov-file`

`cd ML-BuildingDamageViaSatellite`


2. Create and activate a conda environment:

`conda env create -f environment.yml`

`conda activate ML-summative`



