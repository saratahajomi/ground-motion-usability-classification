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

See [installation and usage](python_library/README.md), the [runnable example](examples/test_usability_library.py), and [validation results](python_library/verification.json). Full inference reproduces all 1,755 original testing labels. The project data location is [DesignSafe PRJ-6525](https://www.designsafe-ci.org/data/browser/projects/PRJ-6525/workdir); see Data availability below for access and usage details.

## Run the included waveform examples

This repository includes two supplied sample inputs in `examples/`:

- `20031222191507_NC_PMM_HNN.txt`
- `60_antarctica2012_10deg.miniseed`

After installation, run from the repository root:

```powershell
python examples/test_usability_library.py examples/20031222191507_NC_PMM_HNN.txt
python examples/test_usability_library.py examples/60_antarctica2012_10deg.miniseed
```

Each command tests full and reduced modes and saves feature and prediction CSVs in `library_results/`. For MiniSEED, each valid stored trace receives its own prediction. 

See [the example guide](examples/README.md) for isolated-environment and PyCharm instructions. 

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

The library must reproduce the feature-generation methodology implemented in the original scripts, without modifying feature definitions, names, parameters, or calculations.

Two feature-generation options must be supported:

1. Full feature set (1,368 features)

504 STA/LTA features: Generated using 12 STA/LTA window pairs, 7 thresholds, and 6 minimum exceedance durations.

864 envelope-based features: Generated using moving-window percentile envelopes and peak-context calculations. 

2. Reduced feature set (top 500 features)

Select the top 500 features using the supplied full-feature ranking.

Embed the ranking directly in the library.

Calculate only the selected features and their necessary intermediate dependencies, rather than generating all 1,368 features and subsequently filtering them.

Reuse shared STA/LTA and envelope calculations to minimize computational time.

Ensure every reduced feature produces the same numerical value as its corresponding feature in the full feature set.

Apply the same feature-generation methodology to .txt and .mseed waveforms, ensuring consistent preprocessing, sampling-interval handling, and feature calculations.


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

### DesignSafe Data Depot — project PRJ-6525

Research data supporting **“Automated Usability Classification of Ground-Motion Records Using Machine Learning and Waveform-Based Pattern Recognition”** have been uploaded to [DesignSafe project PRJ-6525 — working directory](https://www.designsafe-ci.org/data/browser/projects/PRJ-6525/workdir). The project supports reproducible, waveform-based classification of ground-motion records as usable or non-usable.

The project page identifies Sara Tahajomi Banafshehvaragh as PI and Pengfei Wang as Co-PI, with Earthquake as the natural hazard type.

### Uploaded training and testing data

| Dataset folder | Purpose | Direct link |
|---|---|---|
| `training_01032026` | Training data supplied for the study's model-development workflow | [Browse training data](https://www.designsafe-ci.org/data/browser/projects/PRJ-6525/workdir/%2Ftraining_01032026) |
| `testing_01032026` | Testing data supplied for evaluating the trained model | [Browse testing data](https://www.designsafe-ci.org/data/browser/projects/PRJ-6525/workdir/%2Ftesting_01032026) |

The folders are stored on DesignSafe rather than bundled with this GitHub repository. Keep the training and testing assignments separate when reproducing the study. The inference library uses the saved model directly: downloading training data does not trigger training, and no retraining is required to analyze testing waveforms or new records.

Open either folder link to browse its uploaded contents and download the files you need. Refer to the project's accompanying metadata for exact record membership, analyst labels, and any processing or split definitions. The folder names are preserved exactly as uploaded; no inference about their date format or record counts is made here.


DesignSafe Data Depot supports browsing, downloading, sharing, and publication of research data. This link points to the project's working directory. Access can depend on sign-in and project permissions; a working-directory URL does not by itself establish public access or identify a published DOI release. See the [DesignSafe Data Depot guide](https://designsafe-ci.org/user-guide/datadepot/) for access and download information.

To use downloaded waveforms with this library:

1. Open the project link and sign in if prompted.
2. Browse the available project files and download the waveform you want to analyze. If the project is unavailable to your account, contact the project owner for access.
3. Use a two-column TXT waveform (time in seconds, then amplitude) or MiniSEED input with the installed library:

```python
from pathlib import Path
from gm_usability import UsabilityModel

# Replace this with the location where you saved the downloaded waveform.
waveform = Path(r"YOUR_DOWNLOAD_DIRECTORY/record.miniseed")
model = UsabilityModel()
predictions, features = model.predict_file(waveform, mode="full")
predictions.to_csv("predictions.csv", index=False)
features.to_csv("features.csv", index_label="record_id")
```

Use `mode="reduced"` to generate only the embedded top 500 and impute the remaining classifier inputs. The library reads local files; it does not authenticate to DesignSafe or download data automatically. The saved classifier and ranking are bundled, so accessing DesignSafe is not required to run the included examples.

### Included examples and data reuse

Two example waveforms are included in `examples/`: `20031222191507_NC_PMM_HNN.txt` and `60_antarctica2012_10deg.miniseed`. The uploaded training and testing folders are linked above and are not bundled with this repository.

For project-specific file descriptions, record labels, train/test assignments, licensing, and citation information, consult the metadata and documentation available in the DesignSafe project. Follow the applicable dataset and original waveform-provider terms. This repository's MIT source-code license does not determine the licensing of third-party waveform data.


## Citation

Allen, R.V. (1978). Automatic earthquake recognition and timing from single traces. Bulletin of the Seismological Society of America, 68(5), 1521–1532.

Ancheta, T.D., Darragh, R.B., Stewart, J.P., Seyhan, E., Silva, W.J., Chiou, B.S.-J., Wooddell, K.E., Graves, R.W., Kottke, A.R., Boore, D.M., Kishida, T., & Donahue, J.L. (2014). NGA-West2 database. Earthquake Spectra, 30(3), 989–1005.

Bellagamba, X., Lee, R., & Bradley, B.A. (2019). A neural network for automated quality screening of ground motion records from small magnitude earthquakes. Earthquake Spectra, 35(4), 1637–1661.

Boore, D.M., & Bommer, J.J. (2005). Processing of strong-motion accelerograms: needs, options and consequences. Soil Dynamics and Earthquake Engineering, 25(2), 93–115.

Breiman, L. (2001). Random forests. Machine Learning, 45, 5–32.

Chen, T., and Guestrin, C. (2016). XGBoost: A scalable tree boosting system. Proceedings of the 22nd ACM SIGKDD International Conference on Knowledge Discovery and Data Mining (pp. 785–794).

Cox, D. R. (1958). The regression analysis of binary sequences. Journal of the Royal Statistical Society: Series B (Methodological), 20(2), 215–232.

Dawood, H.M., Rodriguez-Marek, A., Bayless, J., Goulet, C.A., & Thompson, E.M. (2016). A flatfile for the KiK-net database processed using an automated protocol. Earthquake Spectra, 32(2), 1281–1302.

Freund, Y., and Schapire, R. E. (1997). A decision-theoretic generalization of on-line learning and an application to boosting. Journal of Computer and System Sciences, 55(1), 119–139.

Friedman, J. H. (2001). Greedy function approximation: A gradient boosting machine. The Annals of Statistics, 29(5), 1189–1232.

Geurts, P., Ernst, D., & Wehenkel, L. (2006). Extremely randomized trees. Machine Learning, 63, 3–42.

Ke, G., Meng, Q., Finley, T., Wang, T., Chen, W., Ma, W., Ye, Q., and Liu, T.-Y. (2017). LightGBM: A highly efficient gradient boosting decision tree. In Advances in Neural Information Processing Systems 30.

Kishida, T., Giacinto, D.D., & Iaccarino, G. (2017). Comparison of manual and automated ground motion processing for small-to-moderate magnitude earthquakes in Japan. Earthquake Spectra, 33(3), 875–894.

Kleckner, J.K., Withers, K.B., Thompson, E.M., Rekoske, J.M., Wolin, E., Moschetti, M.P. (2022). Automated detection of clipping in broadband earthquake records. Seismological Research Letters, 93(2A), 880–896.

Pedregosa, F., Varoquaux, G., Gramfort, A., Michel, V., Thirion, B., Grisel, O., Blondel, M., Prettenhofer, P., Weiss, R., Dubourg, V., VanderPlas, J., Passos, A., Cournapeau, D., Brucher, M., Perrot, M., and Duchesnay, E. (2011). Scikit-learn: Machine learning in Python. Journal of Machine Learning Research, 12, 2825–2830.

Prokhorenkova, L., Gusev, G., Vorobev, A., Dorogush, A. V., and Gulin, A. (2018). CatBoost: Unbiased boosting with categorical features. In Advances in Neural Information Processing Systems 31.

Ramos-Sepúlveda, M.E., Brandenberg, S.J., Buckreis, T.E., Parker, G.A., & Stewart, J.P. (2025). High-pass corner-frequency selection and review tool for use in ground-motion processing. Seismological Research Letters, 96(5), 3244–3252.

Thompson, E.M., Hearne, M., Aagaard, B.T., Rekoske, J.M., et al. (2024). Automated, near real-time ground-motion processing at the USGS. Seismological Research Letters, 96(1), 538–553.

Trnkoczy, A. (2012): Understanding and parameter setting of STA/LTA trigger algorithm. - In: Bormann, P. (Ed.), New Manual of Seismological Observatory Practice 2 (NMSOP-2), Potsdam : Deutsches GeoForschungsZentrum GFZ, 1–20.

Wang, S., & Zhang, X. (2020). Automatic detection of amplitude-distorted samples from clipped seismic waveforms. Seismological Research Letters, 91(6), 3563–3573.

Bodenmann, L., Baker, J.W., & Stojadinović, B. (2023). Accounting for path and site effects in spatial ground-motion correlation models using Bayesian inference. Natural Hazards and Earth System Sciences, 23(7), 2387–2402.


## License

The source code in this repository is released under the MIT License. Third-party waveform data remain subject to the terms and policies of their original data providers.

## Contact

Sara Tahajomi Banafshehvaragh  
Department of Civil and Environmental Engineering  
Old Dominion University

sa.tahajomi@gmail.com

staha002@odu.edu

## Dependency versions

Python **3.10 or newer** is declared by the package. Local verification used Python **3.12**. A separate Python 3.13 environment can be used without changing an existing project interpreter. Dependency compatibility on every Python/platform combination has not been tested.

| Requirement | Package installation constraint | Exact local validation version |
|---|---|---|
| NumPy | >=1.26, <3 | 2.2.6 |
| pandas | >=2, <3 | 2.3.2 |
| SciPy | >=1.11 | 1.16.1 |
| scikit-learn | ==1.7.2 | 1.7.2 |
| LightGBM | ==4.6.0 | 4.6.0 |
| Joblib | >=1.3 | 1.5.2 |
| ObsPy | >=1.4, <2 | 1.5.1 |
| setuptools (build dependency) | >=68 | Installed by the build system; no exact build version recorded |

The model was saved with scikit-learn 1.7.2. LightGBM 4.6.0 was used for inference verification. Other installation constraints remain ranges; the validation column records the exact tested direct dependencies rather than claiming all installs use those versions. Transitive dependencies are resolved by pip and are not fully locked.

For the exact direct-dependency validation set, from the repository root:

```powershell
python -m pip install -r python_library/requirements-validation.txt
python -m pip install ./python_library
```

Use the separate environment's Python instead of `python` if you want to preserve an existing environment. The versioned validation requirements are inside `python_library/`; no redundant root requirements file is needed.

## TXT and MiniSEED code examples

These examples are adapted from the supplied test scripts. Personal absolute paths and interpreter paths have been removed. Inputs resolve relative to each script's own folder, so changing the terminal working directory does not change their default waveform paths.

- [TXT example: test_library.py](examples/test_library.py)
- [MiniSEED example: test_library_mseed.py](examples/test_library_mseed.py)

From the repository root, using an isolated Windows environment:

```powershell
py -3.13 -m venv .venv-gm
.\.venv-gm\Scripts\python.exe -m pip install ./python_library
.\.venv-gm\Scripts\python.exe examples/test_library.py
.\.venv-gm\Scripts\python.exe examples/test_library_mseed.py
```

For PyCharm, keep your existing project interpreter selected and run those commands in its Terminal. Both examples run full and reduced modes with the unchanged saved model. They print predictions and elapsed times and save full/reduced feature and prediction CSVs in `library_results/`.

To use another waveform, edit the input assignment in the corresponding script:

```python
from pathlib import Path

# Replace the placeholder with the full path to your own input.
txt_file = Path(r"YOUR_INPUT_PATH/record.txt")
mseed_file = Path(r"YOUR_INPUT_PATH/record.miniseed")

# Optional: choose where output CSVs should be saved.
OUTPUT_DIR = Path(r"YOUR_OUTPUT_DIRECTORY")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
```

Use only the relevant `txt_file` or `mseed_file` assignment in its script. On Windows, a raw string allows backslashes in a full path. Each script already contains its default input and output assignments plus comments showing where to edit them.

## Detailed function reference

See [the complete function guide](python_library/FUNCTION_GUIDE.md) for every public function, argument, return value, reduced-mode policy, missing-column handling, internal helper, CLI option, and validation rule.


## Complete function-by-function reference

This guide describes the shipped version 0.1.1 implementation. The library generates waveform features and applies the unchanged saved LightGBM classifier. There are no training or fitting steps. Feature names and the complete ranking are embedded in Python code; no ranking file is required at runtime.

## Imports

```python
from gm_usability import (
    UsabilityModel,
    read_waveforms,
    extract_features,
    FULL_FEATURE_NAMES,
    RANKED_FEATURE_NAMES,
    TOP500_FEATURE_NAMES,
)
```

## `UsabilityModel(model_path=None)`

Loads the saved model bundle with Joblib. When `model_path` is omitted, it loads `gm_usability/data/model.sav` bundled with the package. Supply a path to use a compatible saved bundle instead.

The bundle contains the classifier, fitted median imputer, exact model input order, and threshold. Construction checks that its 1,368 base names match the library schema and that missingness indicators were not added. No object is fitted or retrained.

Useful attributes:

| Attribute | Meaning |
|---|---|
| `model.names` | Exact base feature names in the saved classifier's input order |
| `model.threshold` | Saved probability threshold: 0.91 for the bundled model |
| `model.bundle` | Loaded bundle, including `model`, `imputer`, and metadata |

Generation order, importance-ranking order, and saved model input order serve different purposes. The library aligns by feature name before prediction; it does not assume these orders are the same.

## `model.predict_file(path, mode="full", *, reduced_policy="impute")`

Reads one TXT or MiniSEED file, generates features per valid waveform, and predicts usability. Returns `(predictions, features)`, both pandas DataFrames. The `reduced_policy` argument must be supplied by name.

### Modes and policies

| Arguments | Features actually generated per waveform | Feature columns returned | Model inputs omitted before imputation |
|---|---:|---:|---:|
| `mode="full"` | 1,368 | 1,368 | 0 |
| `mode="reduced", reduced_policy="impute"` | 500 | 500 | 868 |
| `mode="reduced", reduced_policy="full"` | 1,368 | 500 | 0 |

Full mode generates 504 binary STA/LTA features and 864 envelope medians. Reduced generation selects the embedded ranked top 500: 39 STA/LTA features and 461 envelope medians. It computes only their required parameter combinations while sharing intermediate calculations within each waveform.

### `reduced_policy="impute"` — default

In reduced mode, only 500 feature values are generated. The saved classifier still requires 1,368 columns. The other 868 columns are inserted as missing values, then filled by the existing fitted imputer. Any missing values among the 500 generated features are also imputed.

The imputer uses the statistics stored when it was originally fitted: training-data medians, with a saved fallback for entirely empty training columns. It does not calculate new medians from the input file, fit a new imputer, or retrain the classifier. The classifier receives 1,368 inputs comprising generated and imputed values.

This reduces feature-generation computation. It does not make this model a classifier trained on only 500 features. Predictions can differ from full mode; the original full-model accuracy does not establish reduced-mode accuracy. The amount of runtime improvement depends on waveform size and how much intermediate computation the selected features share.

### `reduced_policy="full"`

Generates all 1,368 values and uses them for prediction, but returns only the top 500 feature columns. This preserves full-input predictions and does not reduce feature-generation work.

For `mode="full"`, either valid policy value produces full inference. An invalid policy value still raises an error.

### Examples

```python
model = UsabilityModel()

# Complete input and feature output.
predictions, features = model.predict_file("record.txt", mode="full")

# Faster generation: only the top 500; remaining inputs are imputed.
predictions, features = model.predict_file(
    "record.miniseed", mode="reduced", reduced_policy="impute"
)

# Complete input for prediction, but expose only the top 500.
predictions, features = model.predict_file(
    "record.txt", mode="reduced", reduced_policy="full"
)
```

### Returned prediction columns

| Column | Meaning |
|---|---|
| `predicted_probability_class_1` | Saved model's probability output for usable class 1 |
| `predicted_label` | Integer 1 if probability >= 0.91; otherwise integer 0 |
| `record_id` | TXT filename stem or MiniSEED trace identifier |
| `mode` | Requested `full` or `reduced` mode |
| `generated_features` | Actual number generated: 1,368 or 500 |
| `omitted_model_inputs` | Number uncomputed and inserted as missing: 0 or 868 |
| `threshold` | Probability threshold used |

`omitted_model_inputs` counts uncomputed columns, not every nonfinite value that needs imputation. Full generation can still produce missing values, for example when a waveform is too short for an averaging window.

The feature DataFrame has one row per valid waveform, indexed by record ID. Full columns follow generation order; reduced columns follow ranking order. Values retain exact original feature names. Save it using:

```python
predictions.to_csv("predictions.csv", index=False)
features.to_csv("features.csv", index_label="record_id")
```

## `model.predict_features(features, *, allow_omitted=False)`

Applies the saved preprocessing and classifier to an existing feature table. Accepts a pandas DataFrame, a single pandas Series, or another object convertible to a DataFrame. It does not read waveforms or generate features. Returns a DataFrame with `predicted_probability_class_1` and `predicted_label`, retaining the input row index.

Processing steps:

1. Check for required model columns.
2. Align to the 1,368 base names in the saved bundle; extra columns are ignored.
3. Convert values to numbers; non-numeric values become missing.
4. Replace positive and negative infinity with missing values.
5. Apply the saved fitted imputer with `transform()`.
6. Use `predict_proba(... )[:, 1]` and the saved threshold.

### Missing columns versus missing values

A **missing column** means a required feature name is absent from the table. By default, this raises an error to catch incomplete or incorrectly named inputs.

A **missing value** means the column exists, but an individual row contains NaN, a non-numeric value, or an infinity. This is handled by the saved imputer even when `allow_omitted=False`.

### `allow_omitted=True`

Explicitly permits absent model columns. These are inserted as missing and imputed. It does not restrict omission to the top-500 schema: callers must provide the intended feature names, since misspelled names are treated as extra columns and their corresponding required names become omitted inputs.

```python
# All 1,368 required columns are present.
result = model.predict_features(full_feature_dataframe)

# Only 500 are present: omission must be explicitly allowed.
result = model.predict_features(top500_dataframe, allow_omitted=True)
```

You do not need to pass `allow_omitted` to `predict_file()`. That method enables omission internally for reduced generation with the impute policy.

## `extract_features(time, amplitude, mode="full")`

Generates features from equal-length one-dimensional time and amplitude arrays. Time is in seconds. Returns a pandas Series of 1,368 full or 500 reduced features, with exact source names. No classifier is loaded and no prediction is made.

It validates the arrays, fits a linear trend with an intercept, and calculates the absolute residual signal. STA/LTA features use trailing averages, threshold comparisons, and minimum consecutive exceedance lengths. Envelope medians use window percentiles, the first envelope maximum, and peak-context side ratios. Quantiles use type-7-compatible interpolation. Left and right ratios are calculated internally to obtain medians but are not returned as separate model features.

Short averaging windows and missing envelope contexts can produce NaN. Original nonfinite median behavior is retained, so some feature values can be infinite before inference normalizes them to missing.

```python
record_id, time, amplitude = read_waveforms("record.txt")[0]
row = extract_features(time, amplitude, mode="reduced")
result = model.predict_features(row, allow_omitted=True)
```

## `read_waveforms(path)`

Returns a list of `(record_id, time_array, amplitude_array)` tuples. It does not generate features or load the model.

### TXT

Accepts `.txt`, case-insensitively. Expects whitespace-separated columns with time first and amplitude second; additional columns are ignored. Numeric conversion removes sample pairs where either value is nonfinite. The record ID is the filename without its extension. Invalid TXT waveforms raise an error.

### MiniSEED

Accepts `.mseed`, `.miniseed`, `.ms`, and `.seed`, case-insensitively, and reads the file as MiniSEED through ObsPy. Each stored trace is processed separately with time in seconds relative to its own start. No trace merging, resampling, unit conversion, or instrument-response removal is performed.

IDs have the form `file_stem:trace.id:start_time:trace_index`. For example:

```text
60_antarctica2012_10deg:ER.HOO..BH1:2012-06-01T05:08:17.000000Z:2
```

This identifies network ER, station HOO, blank location, channel BH1, start time, and zero-based stored-trace index 2. Its corresponding name under the supplied TXT naming convention is `20120601050817_ER_HOO_BH1.txt`. The library does not automatically create that TXT export.

Invalid traces are skipped with warnings identifying the trace and reason. If no valid traces remain, an error is raised. The included Antarctica file has 53 stored traces; two have only 40 samples and are skipped, leaving 51 valid traces.

### Validation rules

- Equal-length one-dimensional arrays.
- At least 64 finite sample pairs after filtering.
- Nonzero amplitude variance.
- Time sorted if any consecutive interval is nonpositive; sorting does not deduplicate timestamps.
- Finite median sampling interval from 1e-6 through 1 second.

The median interval is used without resampling. The implementation does not independently reject every irregular or duplicate timestamp if the median interval passes validation.

## Feature-name constants

| Constant | Contents and order |
|---|---|
| `FULL_FEATURE_NAMES` | All 1,368 names in generation order: STA/LTA followed by envelope medians |
| `RANKED_FEATURE_NAMES` | All 1,368 names in supplied ranking order |
| `TOP500_FEATURE_NAMES` | First 500 ranked names |

These are immutable tuples. All names preserve the original spelling, double underscores, parameter formatting, and suffixes. No runtime ranking CSV is read.

## Internal helpers

These are implementation details in `features.py`; ordinary callers should use the public functions above.

| Function | Behavior |
|---|---|
| `_validate(x, y)` | Converts arrays to float, filters finite pairs, checks sample count and variance, sorts time if needed, checks median interval, returns `(x, y, dt)` |
| `_quantile(v, p)` | Retains finite values and computes percentile `p` on a 0–100 scale with linear/type-7 interpolation; returns NaN for an empty finite set |
| `_mean(signal, n)` | Computes a trailing mean with cumulative sums; incomplete initial windows are NaN |

## `cli.main()` and the command line

`cli.main()` parses arguments, loads `UsabilityModel`, calls `predict_file`, writes CSVs, and prints predictions. Installation exposes it as `gm-usability`.

```powershell
gm-usability examples/20031222191507_NC_PMM_HNN.txt --mode full --output full_predictions.csv --features full_features.csv
gm-usability examples/60_antarctica2012_10deg.miniseed --mode reduced --reduced-policy impute --output reduced_predictions.csv --features top500_features.csv
```

| Argument | Meaning |
|---|---|
| `input` | Required TXT or MiniSEED path |
| `--mode` | `full` (default) or `reduced` |
| `--reduced-policy` | `impute` (default) or `full` |
| `--model` | Optional compatible saved bundle path |
| `--output` | Prediction CSV destination; default `predictions.csv` |
| `--features` | Optional feature CSV destination |

Relative paths resolve from the working directory. The CLI does not create missing destination parent folders.

## Repository example script

`examples/test_usability_library.py` defines `main()`, which runs both modes, measures total time, prints predictions, and saves four CSVs per input. Its optional input path defaults to the included TXT beside the script. `--output-dir` defaults to `library_results`; this script creates the destination directory.

Use a separate environment from PyCharm's Terminal to keep an existing project interpreter unchanged. Installation commands and complete example runs are in [the example guide](examples/README.md).

## Verification and practical limits

Full inference reproduced all 1,755 original testing labels, with maximum probability difference about 1.1e-16. Feature values were checked against original R functions with numerical tolerance, and a separately supplied record comparison found exact matches for all 1,368 names and values as provided. Reduced values matched the selected full subset. Five package tests passed, including invalid MiniSEED trace handling. Details are in [`verification.json`](python_library/verification.json).

These checks validate the tested inputs and inference behavior; they do not establish reduced-mode accuracy or guarantee bit-for-bit numerical identity on every platform. Neither the saved classifier nor its fitted preprocessing has been retrained.

### Additional code examples for constants and internal helpers

```python
from gm_usability import FULL_FEATURE_NAMES, RANKED_FEATURE_NAMES, TOP500_FEATURE_NAMES

print(len(FULL_FEATURE_NAMES))     # 1368
print(len(TOP500_FEATURE_NAMES))   # 500
print(RANKED_FEATURE_NAMES[0])     # highest-ranked original feature name
top500_table = full_feature_dataframe.reindex(columns=TOP500_FEATURE_NAMES)
```

The internal helpers below are implementation details rather than the supported public workflow. These small examples show their behavior; ordinary applications should call `read_waveforms`, `extract_features`, and `UsabilityModel` instead.

```python
import numpy as np
from gm_usability.features import _validate, _quantile, _mean

time = np.arange(100) * 0.02
amplitude = np.sin(time * 3)
time, amplitude, dt = _validate(time, amplitude)
print(dt)                         # approximately 0.02 seconds
print(_quantile(np.array([1., 2., 3., 4.]), 50))  # 2.5
print(_mean(np.array([1., 2., 3., 4.]), 2))       # [nan, 1.5, 2.5, 3.5]
```

`cli.main()` is invoked by the installed `gm-usability` command. It can also be invoked with `python -m gm_usability.cli` using the same command-line arguments; it parses the process arguments rather than accepting a waveform argument directly.
