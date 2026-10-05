# Ground-Motion Usability Classification

Repository supporting the study **“Automated Usability Classification of Ground-Motion Records Using Machine Learning and Waveform-Based Pattern Recognition”** by Sara Tahajomi Banafshehvaragh, Pengfei Wang, and Tadahiro Kishida.

## Python usability library

The installable library is available in [`python_library/`](python_library/README.md). It includes the original saved LightGBM model and embedded ranking; no retraining or external ranking CSV is needed.

- **Full:** 1,368 features (504 binary STA/LTA + 864 envelope medians).
- **Reduced:** directly generates only the top 500 ranked features (39 STA/LTA + 461 envelope medians).
- **Inputs:** two-column TXT and MiniSEED; each MiniSEED trace is processed separately.
- **Outputs:** usability probability, binary label, and original feature names.

```bash
python -m pip install "git+https://github.com/saratahajomi/ground-motion-usability-classification.git#subdirectory=python_library"
```

```python
from gm_usability import UsabilityModel

model = UsabilityModel()
predictions, features = model.predict_file("record.txt", mode="reduced")
print(predictions)
```

Both modes use the unchanged 1,368-input classifier, fitted imputer, and saved 0.91 threshold. Reduced mode fills the 868 uncomputed model inputs using the saved imputer; its predictions can differ from full mode. The original full-model accuracy is not a validation of reduced-mode accuracy.

See [installation and usage](python_library/README.md), the [runnable example](examples/test_usability_library.py), and [validation results](python_library/verification.json). Full inference reproduces all 1,755 original testing labels. The large training and testing waveform datasets remain a separate planned release.

## Overview

This repository is intended to provide reproducibility materials for an automated machine-learning framework that classifies raw ground-motion records as usable or non-usable using waveform-derived features.

The study uses two feature families:

- Multi-configuration STA/LTA-derived features.
- Envelope-percentile features describing the temporal context of waveform amplitudes.

The repository includes the installable inference library, bundled trained model, and example workflows. Record labels, train/test assignments, and additional research scripts are planned separately. The selected raw ground-motion records are large and will be added separately.

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
├── python_library/                      # installable library and bundled model
├── models/                              # additional research artifacts planned
└── examples/                            # runnable library example and sample outputs
```

## Data

The study uses raw ground-motion waveforms obtained from IRIS Data Services, with earthquake metadata obtained from the USGS Comprehensive Earthquake Catalog (ComCat).

The final data release is planned to include:

- `record_labeling.csv`: analyst-defined usability labels, training/testing assignment, and dataset inclusion/exclusion information.
- `selected_records_training/`: selected waveform records used for model development/training.
- `selected_records_testing/`: selected waveform records from held-out events used for testing.

The waveform folders are intentionally not included in the initial GitHub repository because of their size. Their public data-repository location and DOI will be added here when the dataset upload is finalized.

## Feature generation

The Python library implements feature generation in `python_library/gm_usability/features.py`. Additional research scripts will be placed in `src/feature_generation/`.

The manuscript evaluates a multi-configuration STA/LTA feature set spanning 12 STA/LTA window pairs, 7 thresholds, and 6 minimum exceedance durations (504 configurations), together with envelope-percentile waveform features.

## Models

The library bundles the unchanged 1,368-input LightGBM model in `python_library/gm_usability/data/model.sav`. Its SHA-256, input order, and threshold are recorded in `python_library/model_manifest.json`. Additional model artifacts may be placed in `models/`.

## Examples

Run `examples/test_usability_library.py` with a TXT or MiniSEED path. Example prediction outputs are in `examples/library_outputs/`.

## Installation

After cloning the repository, install the usability library with:

```bash
python -m pip install ./python_library
```

Python 3.10 or newer is required. The library pins scikit-learn 1.7.2 and LightGBM 4.6.0; see its README for an isolated environment setup.

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
