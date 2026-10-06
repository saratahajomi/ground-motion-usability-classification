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
predictions, features = model.predict_file("examples/20031222191507_NC_PMM_HNN.txt", mode="reduced")
print(predictions)
```

Both modes use the unchanged 1,368-input classifier, fitted imputer, and saved 0.91 threshold. Reduced mode fills the 868 uncomputed model inputs using the saved imputer; its predictions can differ from full mode. The original full-model accuracy is not a validation of reduced-mode accuracy.

See [installation and usage](python_library/README.md), the [runnable example](examples/test_usability_library.py), and [validation results](python_library/verification.json). Full inference reproduces all 1,755 original testing labels. The large training and testing waveform datasets remain a separate planned release.

## Run the included waveform examples

This repository includes two supplied sample inputs in `examples/`:

- `20031222191507_NC_PMM_HNN.txt`
- `60_antarctica2012_10deg.miniseed`

After installation, run from the repository root:

```powershell
python examples/test_usability_library.py examples/20031222191507_NC_PMM_HNN.txt
python examples/test_usability_library.py examples/60_antarctica2012_10deg.miniseed
```

Each command tests full and reduced modes and saves feature and prediction CSVs in `library_results/`. For MiniSEED, each valid stored trace receives its own prediction. Version 0.1.1 skips invalid traces with explicit warnings; the Antarctica file contains 53 traces, of which two have only 40 samples and are skipped, leaving 51 valid traces.

See [the example guide](examples/README.md) for isolated-environment and PyCharm instructions. Uploading files does not delete existing GitHub paths. Remove obsolete root `models/`, `src/`, `data/`, and `requirements.txt` placeholders separately when updating an existing repository.

## Overview

This repository is intended to provide reproducibility materials for an automated machine-learning framework that classifies raw ground-motion records as usable or non-usable using waveform-derived features.

The study uses two feature families:

- Multi-configuration STA/LTA-derived features.
- Envelope-percentile features describing the temporal context of waveform amplitudes.

The repository includes the installable inference library, bundled trained model, and example workflows. Two waveform inputs are included for testing; the complete research dataset is a separate release.

## Repository structure

```text
ground-motion-usability-classification/
├── README.md
├── LICENSE
├── python_library/
│   ├── pyproject.toml
│   ├── README.md
│   ├── model_manifest.json
│   ├── verification.json
│   ├── gm_usability/
│   │   ├── features.py          # feature generation
│   │   ├── inference.py         # saved-model inference
│   │   ├── _ranking.py          # embedded feature ranking
│   │   ├── cli.py
│   │   ├── __init__.py
│   │   └── data/model.sav       # original saved model
│   └── tests/test_library.py
└── examples/
    ├── README.md
    ├── test_usability_library.py
    ├── 20031222191507_NC_PMM_HNN.txt
    ├── 60_antarctica2012_10deg.miniseed
    ├── input_manifest.json
    └── library_outputs/
```

## Feature generation

The Python library implements feature generation in `python_library/gm_usability/features.py`.

The manuscript evaluates a multi-configuration STA/LTA feature set spanning 12 STA/LTA window pairs, 7 thresholds, and 6 minimum exceedance durations (504 configurations), together with envelope-percentile waveform features.

## Models

The library bundles the unchanged 1,368-input LightGBM model in `python_library/gm_usability/data/model.sav`. Its SHA-256, input order, and threshold are recorded in `python_library/model_manifest.json`.

## Examples

Run `examples/test_usability_library.py` with a TXT or MiniSEED path. Example prediction outputs are in `examples/library_outputs/`.

## Installation

After cloning the repository, install the usability library with:

```bash
python -m pip install ./python_library
```

Python 3.10 or newer is required. The library pins scikit-learn 1.7.2 and LightGBM 4.6.0; see its README for an isolated environment setup.

## Data availability

The two example waveforms are included in `examples/`. The complete training/testing waveform collection is not included. A persistent link/DOI for the complete training and testing dataset will be added here after archival.

## Citation

Citation information will be updated when the associated manuscript and dataset have their final bibliographic identifiers.

## License

The source code in this repository is released under the MIT License. Third-party waveform data remain subject to the terms and policies of their original data providers.

## Contact

Sara Tahajomi Banafshehvaragh  
Department of Civil and Environmental Engineering  
Old Dominion University

## Detailed function reference

See [the complete function guide](python_library/FUNCTION_GUIDE.md) for every public function, argument, return value, reduced-mode policy, missing-column handling, internal helper, CLI option, and validation rule.
