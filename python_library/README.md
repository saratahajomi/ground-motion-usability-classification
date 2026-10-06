# Ground-motion usability library

Uses your unchanged saved `BINARY_STALTA_PLUS_CONTEXT_best_validation_model_09102026.sav`. There is no training step and no runtime ranking CSV. All original feature names are preserved exactly.

## Install

From the repository root, run `python -m pip install ./python_library`. Alternatively, from this `python_library` directory:

```powershell
python -m pip install .
```

The model was saved with scikit-learn 1.7.2; the package pins that version. LightGBM 4.6.0 was used for validation. A working Python 3.10 or newer is required.

## Python

```python
from gm_usability import UsabilityModel

model = UsabilityModel()  # bundled original model
predictions, features = model.predict_file("record.txt", mode="full")
predictions, features = model.predict_file("record.miniseed", mode="reduced")
print(predictions)
features.to_csv("features.csv", index_label="record_id")
```

`full` generates **1,368 features: 504 binary STA/LTA + 864 envelope medians**. `reduced` generates **only the top 500** in your embedded ranking order: 39 STA/LTA + 461 envelope medians. Shared intermediate envelopes and STA/LTA ratios are cached within each waveform. Both modes use the same original classifier, fitted median imputer, saved input order, and **0.91 threshold**. Class 1 is usable and class 0 is non-usable.

The classifier requires 1,368 columns. In reduced mode, the other 868 columns are marked missing and filled by its existing fitted imputer. Nothing is refitted. Reduced predictions can differ from full predictions; reduced accuracy has not been established by the original full-model evaluation. `generated_features` and `omitted_model_inputs` make this visible in predictions. An optional `reduced_policy="full"` generates the full input for prediction and exposes only 500 outputs, but does not provide reduced computation.

To predict existing feature tables:

```python
predictions = model.predict_features(feature_dataframe)
# Explicitly allow an incomplete feature table:
predictions = model.predict_features(top500_dataframe, allow_omitted=True)
```

Input columns are aligned to the order stored in the model, not the ranking order. Non-numeric values and infinities become missing values before the saved imputer transforms them. Prediction uses `predict_proba(... )[:, 1] >= 0.91`.

## Command line

```powershell
gm-usability record.txt --mode full --output full_predictions.csv --features full_features.csv
gm-usability record.miniseed --mode reduced --output reduced_predictions.csv --features top500_features.csv
```

## Waveform behavior

TXT expects whitespace-separated time in seconds and amplitude in its first two columns. Finite sample pairs are retained, time is sorted if necessary, and the median time interval is used. At least 64 finite samples, nonzero variance, and a median interval from 1 microsecond through 1 second are required, matching the supplied R reader. No resampling is performed.

MiniSEED is read with ObsPy. Each stored trace is processed separately using seconds relative to its own start time. Multiple traces produce multiple rows. Trace IDs, start times, and trace indices distinguish rows. Traces are not merged and instrument response is not removed. Invalid MiniSEED traces are skipped with warnings identifying each trace and reason; if no valid traces remain, an error is raised. Invalid TXT waveforms raise an error. MiniSEED output reflects the samples stored in the file; equivalence to another TXT export requires that export to use the same samples and units.

Feature extraction translates the supplied R functions: linear detrending, absolute residuals, trailing STA/LTA means, R-style rounding, continuous threshold-exceedance runs, type-7 quantiles, first envelope maximum, inclusive peak-window boundaries, and original median/Inf/NA behavior. Left and right ratios are calculated internally only to obtain requested median outputs.

## Validation

See `verification.json` and `../examples/library_outputs/` for sample prediction CSVs. Checks include all 1,755 supplied full-model prediction rows, the original R feature functions on the supplied TXT waveform, equality of reduced features with the selected full subset, and MiniSEED processing. `model_manifest.json` records the original model SHA-256 and exact saved input order. The packaged model is a byte-for-byte copy.

The supplied current training script describes a 504-feature STA/LTA-only bundle, while the supplied saved model and testing predictions correspond to the 1,368-feature LightGBM bundle. The delivered inference was verified against the latter; no training script was executed.

Run package tests after installation:

```powershell
python -m unittest discover -s tests
```
