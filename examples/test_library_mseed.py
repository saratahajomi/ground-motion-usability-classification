from pathlib import Path
from time import perf_counter

from gm_usability import UsabilityModel

# Input defaults to the supplied waveform beside this script.
# To use your own file, replace this assignment with Path(r"YOUR_INPUT_PATH").
mseed_file = Path(__file__).resolve().parent / "60_antarctica2012_10deg.miniseed"

# Outputs are saved under the repository's library_results folder.
# Change OUTPUT_DIR below if you want another destination.
OUTPUT_DIR = Path(__file__).resolve().parents[1] / "library_results"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

model = UsabilityModel()  # Existing saved model; no retraining.

for mode in ("full", "reduced"):
    start = perf_counter()
    predictions, features = model.predict_file(mseed_file, mode=mode)
    elapsed = perf_counter() - start

    predictions.to_csv(
        OUTPUT_DIR / f"{mseed_file.stem}_{mode}_predictions.csv",
        index=False,
    )
    features.to_csv(
        OUTPUT_DIR / f"{mseed_file.stem}_{mode}_features.csv",
        index_label="record_id",
    )

    print(f"\nMode: {mode}")
    print(f"Processed traces: {len(predictions)}")
    print(f"Features per trace: {features.shape[1]}")
    print(f"Elapsed time: {elapsed:.3f} seconds")
    print(
        predictions[
            ["record_id", "predicted_probability_class_1", "predicted_label"]
        ].to_string(index=False)
    )

print(f"\nResults saved in: {OUTPUT_DIR}")
