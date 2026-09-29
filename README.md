# Ground-Motion Usability Classification

Repository supporting the study **“Automated Usability Classification of Ground-Motion Records Using Machine Learning and Waveform-Based Pattern Recognition”** by Sara Tahajomi Banafshehvaragh, Pengfei Wang, and Tadahiro Kishida.

## Overview

This repository is intended to provide reproducibility materials for an automated machine-learning framework that classifies raw ground-motion records as usable or non-usable using waveform-derived features.

The study uses two feature families:

- Multi-configuration STA/LTA-derived features.
- Envelope-percentile features describing the temporal context of waveform amplitudes.

The repository will contain the record labels and train/test assignments, feature-generation code, trained model files, and example workflows. The selected raw ground-motion records are large and will be added separately.

## Repository structure

```text
ground-motion-usability-classification/
├── README.md
├── LICENSE
├── requirements.txt
├── data/
│   ├── README.md
│   ├── record_labeling.csv              # to be added
│   ├── selected_records_training/       # data to be added
│   └── selected_records_testing/        # data to be added
├── src/
│   └── feature_generation/              # scripts to be added
├── models/                              # trained model(s) to be added
└── examples/                            # example usage to be added
```

## Data

The study uses raw ground-motion waveforms obtained from IRIS Data Services, with earthquake metadata obtained from the USGS Comprehensive Earthquake Catalog (ComCat).

The final data release is planned to include:

- `record_labeling.csv`: analyst-defined usability labels, training/testing assignment, and dataset inclusion/exclusion information.
- `selected_records_training/`: selected waveform records used for model development/training.
- `selected_records_testing/`: selected waveform records from held-out events used for testing.

The waveform folders are intentionally not included in the initial GitHub repository because of their size. Their public data-repository location and DOI will be added here when the dataset upload is finalized.

## Feature generation

Feature-generation scripts will be placed in `src/feature_generation/`.

The manuscript evaluates a multi-configuration STA/LTA feature set spanning 12 STA/LTA window pairs, 7 thresholds, and 6 minimum exceedance durations (504 configurations), together with envelope-percentile waveform features.

## Models

Final trained model artifacts required for reproducibility will be placed in `models/` after the manuscript/model version is finalized.

## Examples

Reproducible example workflows for feature generation and model application will be placed in `examples/`.

## Installation

After cloning the repository, install the Python dependencies with:

```bash
pip install -r requirements.txt
```

Exact package versions will be pinned when the final reproducibility environment is archived.

## Data availability

Large waveform data are not stored in the initial GitHub repository. A persistent link/DOI for the complete training and testing dataset will be added here after archival.

## Citation

Citation information will be updated when the associated manuscript and dataset have their final bibliographic identifiers.

## License

The source code in this repository is released under the MIT License. Third-party waveform data remain subject to the terms and policies of their original data providers.

## Contact

Sara Tahajomi Banafshehvaragh  
Department of Civil and Environmental Engineering  
Old Dominion University
