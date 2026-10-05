import unittest
import numpy as np
import pandas as pd
from gm_usability import UsabilityModel, extract_features, FULL_FEATURE_NAMES, TOP500_FEATURE_NAMES, RANKED_FEATURE_NAMES


class LibraryTests(unittest.TestCase):
    def test_schema_and_reduced_values(self):
        x = np.arange(6000) * .02
        y = np.sin(x * 3) * (1 + 10 * np.exp(-((x-45)/4)**2))
        full = extract_features(x, y)
        reduced = extract_features(x, y, 'reduced')
        self.assertEqual(len(full), 1368)
        self.assertEqual(len(reduced), 500)
        self.assertEqual(len(set(RANKED_FEATURE_NAMES)), 1368)
        self.assertEqual(set(FULL_FEATURE_NAMES), set(RANKED_FEATURE_NAMES))
        np.testing.assert_array_equal(reduced.values, full.reindex(TOP500_FEATURE_NAMES).values)

    def test_short_record_missing_stalta(self):
        x = np.arange(64) * .02
        y = np.sin(x * 4)
        full = extract_features(x, y)
        self.assertTrue(full[[n for n in full.index if n.startswith('stalta__')]].isna().all())

    def test_saved_inference_and_order(self):
        m = UsabilityModel()
        self.assertEqual(m.threshold, .91)
        raw = pd.DataFrame([dict(zip(m.names, m.bundle['imputer'].statistics_))])
        a = m.predict_features(raw)
        b = m.predict_features(raw[raw.columns[::-1]])
        np.testing.assert_array_equal(a.values, b.values)
        with self.assertRaises(ValueError):
            m.predict_features(raw.iloc[:, :500])
        missing = raw.copy()
        missing.iloc[0, 0] = np.inf
        nan = raw.copy()
        nan.iloc[0, 0] = np.nan
        np.testing.assert_array_equal(m.predict_features(missing).values, m.predict_features(nan).values)

    def test_invalid_input(self):
        with self.assertRaises(ValueError):
            extract_features(np.arange(100), np.zeros(100))
        with self.assertRaises(ValueError):
            extract_features(np.arange(20), np.arange(20))


if __name__ == '__main__':
    unittest.main()
