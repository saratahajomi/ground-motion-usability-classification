# Library function reference

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

Use a separate environment from PyCharm's Terminal to keep an existing project interpreter unchanged. Installation commands and complete example runs are in [the example guide](../examples/README.md).

## Verification and practical limits

Full inference reproduced all 1,755 original testing labels, with maximum probability difference about 1.1e-16. Feature values were checked against original R functions with numerical tolerance, and a separately supplied record comparison found exact matches for all 1,368 names and values as provided. Reduced values matched the selected full subset. Five package tests passed, including invalid MiniSEED trace handling. Details are in `verification.json`.

These checks validate the tested inputs and inference behavior; they do not establish reduced-mode accuracy or guarantee bit-for-bit numerical identity on every platform. Neither the saved classifier nor its fitted preprocessing has been retrained.
