import argparse
from .inference import UsabilityModel


def main():
    parser = argparse.ArgumentParser(description="Predict ground-motion usability without retraining")
    parser.add_argument("input")
    parser.add_argument("--mode", choices=["full", "reduced"], default="full")
    parser.add_argument("--reduced-policy", choices=["full", "impute"], default="impute")
    parser.add_argument("--model")
    parser.add_argument("--output", default="predictions.csv")
    parser.add_argument("--features")
    args = parser.parse_args()
    predictions, features = UsabilityModel(args.model).predict_file(args.input, args.mode, reduced_policy=args.reduced_policy)
    predictions.to_csv(args.output, index=False)
    if args.features:
        features.to_csv(args.features, index_label="record_id")
    print(predictions.to_string(index=False))


if __name__ == "__main__":
    main()
