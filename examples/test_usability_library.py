"""Run: python examples/test_usability_library.py path/to/record.txt

PyCharm: edit INPUT_FILE below, then run this file using the environment
where gm-usability is installed. Your existing project interpreter may be
left unchanged by running the separate environment's python in Terminal.
"""
from pathlib import Path
from time import perf_counter
import argparse
from gm_usability import UsabilityModel

INPUT_FILE = Path(__file__).resolve().parent / "20031222191507_NC_PMM_HNN.txt"  # Set this to your TXT or MiniSEED file.


def main():
    parser = argparse.ArgumentParser(description="Compare full and top-500 inference")
    parser.add_argument("input", nargs="?", type=Path, default=INPUT_FILE)
    parser.add_argument("--output-dir", type=Path, default=Path("library_results"))
    args = parser.parse_args()
    if not args.input.is_file():
        parser.error(f"Input does not exist: {args.input}. Supply a path or edit INPUT_FILE.")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    model = UsabilityModel()
    for mode in ("full", "reduced"):
        start = perf_counter()
        predictions, features = model.predict_file(args.input, mode=mode)
        elapsed = perf_counter() - start
        predictions.to_csv(args.output_dir / f"{args.input.stem}_{mode}_predictions.csv", index=False)
        features.to_csv(args.output_dir / f"{args.input.stem}_{mode}_features.csv", index_label="record_id")
        print(f"\n{mode}: {len(predictions)} records, {features.shape[1]} features, {elapsed:.3f} seconds")
        print(predictions.to_string(index=False))
    print(f"\nResults saved in {args.output_dir.resolve()}")


if __name__ == "__main__":
    main()
