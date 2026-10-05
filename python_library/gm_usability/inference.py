from importlib.resources import files
import joblib
import numpy as np
import pandas as pd
from .features import extract_features, read_waveforms
from ._ranking import FULL_FEATURE_NAMES, RANKED_FEATURE_NAMES


class UsabilityModel:
    """Original saved classifier and fitted preprocessing; no fitting methods.

    reduced_policy='full' preserves full predictions and exposes top 500 features.
    reduced_policy='impute' generates 500 and treats omitted inputs as missing;
    this is an explicit approximation, not the original full inference behavior.
    """
    def __init__(self, model_path=None):
        self.bundle = joblib.load(model_path or files("gm_usability").joinpath("data/model.sav"))
        self.names = tuple(self.bundle["base_feature_names"])
        if len(self.names) != 1368 or set(self.names) != set(FULL_FEATURE_NAMES):
            raise ValueError("Saved model does not match the original 1,368-feature schema")
        if self.bundle["missingness_indicators_added"]:
            raise ValueError("Unexpected missingness indicators in model")
        self.threshold = float(self.bundle["threshold"])

    def predict_features(self, features, *, allow_omitted=False):
        """Align by saved input order, coerce, replace infinities, transform, predict."""
        frame = features.to_frame().T if isinstance(features, pd.Series) else pd.DataFrame(features)
        missing = set(self.names) - set(frame.columns)
        if missing and not allow_omitted:
            raise ValueError(f"Missing {len(missing)} model features; use full features or explicitly allow omission")
        matrix = frame.reindex(columns=self.names).apply(pd.to_numeric, errors="coerce").to_numpy(dtype=float)
        matrix[~np.isfinite(matrix)] = np.nan
        matrix = self.bundle["imputer"].transform(matrix)
        model = self.bundle["model"]
        probability = np.asarray(model.predict_proba(matrix)[:, 1], dtype=float)
        return pd.DataFrame({"predicted_probability_class_1": probability,
                             "predicted_label": (probability >= self.threshold).astype(int)}, index=frame.index)

    def predict_file(self, path, mode="full", *, reduced_policy="impute"):
        if mode not in {"full", "reduced"} or reduced_policy not in {"full", "impute"}:
            raise ValueError("Invalid mode or reduced_policy")
        results, feature_rows = [], []
        generation_mode = "reduced" if mode == "reduced" and reduced_policy == "impute" else "full"
        for record_id, time, amplitude in read_waveforms(path):
            row = extract_features(time, amplitude, generation_mode)
            result = self.predict_features(row, allow_omitted=generation_mode == "reduced").iloc[0].to_dict()
            result["predicted_label"] = int(result["predicted_label"])
            result.update(record_id=record_id, mode=mode, generated_features=len(row),
                          omitted_model_inputs=1368-len(row), threshold=self.threshold)
            results.append(result)
            visible = row if mode == "full" else row.reindex(RANKED_FEATURE_NAMES[:500])
            visible.name = record_id
            feature_rows.append(visible)
        return pd.DataFrame(results), pd.DataFrame(feature_rows)
