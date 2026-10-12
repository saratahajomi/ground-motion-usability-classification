"""
GROUND-MOTION USABILITY LIBRARY — FUNCTION GUIDE

This library classifies ground-motion waveforms as usable or non-usable
using the existing saved LightGBM model. No retraining is performed.

MAIN FUNCTIONS
--------------

UsabilityModel(model_path=None)
    Loads the saved classifier, fitted median imputer, feature input
    order, and probability threshold.
    Uses the model bundled with the library unless a path is supplied.

model.predict_file(path, mode="full", reduced_policy="impute")
    Reads a TXT or MiniSEED file, generates features, and predicts
    usability. Returns two pandas DataFrames:
        predictions: record IDs, probabilities, labels, and mode details.
        features: generated feature values with their exact original names.

    mode="full":
        Generates all 1,368 features:
        504 binary STA/LTA features + 864 envelope median features.

    mode="reduced":
        Generates only the embedded top 500 ranked features:
        39 STA/LTA features + 461 envelope median features.
        The remaining 868 classifier inputs are filled using the
        saved fitted imputer. Predictions may differ from full mode.

    reduced_policy="impute" (default):
        When mode="reduced", generates ONLY the top 500 features.
    reduced_policy="full":
        Optional alternative for reduced mode. Generates all 1,368
        inputs for prediction, but returns only the top 500 features.
        This option does not reduce feature-generation computation.

model.predict_features(features, allow_omitted=False)
    Predicts from an existing feature DataFrame or a single feature Series.
    Aligns columns to the saved model input order, converts non-numeric
    and infinite values to missing values, applies the saved imputer,
    and predicts usability.
    Missing columns raise an error unless allow_omitted=True.

extract_features(time, amplitude, mode="full")
    Generates features directly from time and amplitude arrays.
    Applies linear detrending and uses the absolute residual signal.
    Returns a pandas Series containing 1,368 or 500 named features.
    Does not load the classifier or produce a prediction.

read_waveforms(path)
    Reads a whitespace-separated TXT file or a MiniSEED file.
    TXT: first column is time in seconds; second column is amplitude.
    MiniSEED: processes each stored trace separately.
    Returns a list of (record_id, time_array, amplitude_array).

    Requires at least 64 finite sample pairs, nonzero amplitude variance,
    and a median sampling interval between 1e-6 and 1 second.
    Invalid MiniSEED traces are skipped with warnings.
    Raises an error if no valid traces remain.
    No merging, resampling, or instrument-response removal is performed.

FEATURE-NAME CONSTANTS
---------------------

FULL_FEATURE_NAMES
    Exact names of the 1,368 full features in generation order.

RANKED_FEATURE_NAMES
    All 1,368 feature names in the embedded importance-ranking order.

TOP500_FEATURE_NAMES
    The first 500 names from the embedded ranking.
    No ranking CSV is required at runtime.

INTERNAL FEATURE HELPERS
------------------------

_validate(time, amplitude)
    Validates waveform arrays, retains finite sample pairs, sorts time
    when necessary, and calculates the median sampling interval.

_quantile(values, percentile)
    Calculates a percentile using R-compatible type-7 interpolation.
    Returns NaN when no finite values are available.

_mean(signal, window_samples)
    Calculates a trailing moving average.
    Initial incomplete windows are represented as NaN.

cli.main()
    Provides the gm-usability command-line interface for reading an
    input file and saving prediction and optional feature CSVs.

PREDICTION INTERPRETATION
------------------------

predicted_probability_class_1:
    The saved model's probability output for the usable class.

predicted_label:
    1 = usable; 0 = non-usable.
    A probability greater than or equal to 0.91 produces label 1.

All original feature names are preserved exactly.
Reduced-mode accuracy has not been established by the original
full-model evaluation.
"""
from pathlib import Path
from time import perf_counter

from gm_usability import UsabilityModel

# Input defaults to the supplied waveform beside this script.
# To use your own file, replace this assignment with Path(r"YOUR_INPUT_PATH").
txt_file = Path(__file__).resolve().parent / "20031222191507_NC_PMM_HNN.txt"

# Outputs are saved under the repository's library_results folder.
# Change OUTPUT_DIR below if you want another destination.
OUTPUT_DIR = Path(__file__).resolve().parents[1] / "library_results"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

model = UsabilityModel()  # Loads your existing saved model; no retraining.

for mode in ("full", "reduced"):
    start = perf_counter()

    predictions, features = model.predict_file(txt_file, mode=mode)

    elapsed = perf_counter() - start

    predictions.to_csv(
        OUTPUT_DIR / f"{txt_file.stem}_{mode}_predictions.csv",
        index=False,
    )
    features.to_csv(
        OUTPUT_DIR / f"{txt_file.stem}_{mode}_features.csv",
        index_label="record_id",
    )

    label = int(predictions.iloc[0]["predicted_label"])
    probability = predictions.iloc[0]["predicted_probability_class_1"]

    print(f"\nMode: {mode}")
    print(f"Generated features: {features.shape[1]}")
    print(f"Probability of usable: {probability:.8f}")
    print(f"Prediction: {'Usable' if label == 1 else 'Non-usable'}")
    print(f"Elapsed time: {elapsed:.3f} seconds")

print(f"\nResults saved in: {OUTPUT_DIR}")
