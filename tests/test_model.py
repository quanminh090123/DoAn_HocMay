
import json
import unittest
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from pandas.testing import assert_frame_equal
from sklearn.linear_model import LogisticRegression

from src.train import MODELS_DIR, load_xy, make_pipeline, split_train_test


class ModelHandoffTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.X, cls.y = load_xy()

    def test_split_is_reproducible_and_stratified(self):
        X_train_a, X_test_a, y_train_a, y_test_a = split_train_test(self.X, self.y)
        X_train_b, X_test_b, y_train_b, y_test_b = split_train_test(self.X, self.y)

        self.assertEqual(set(X_train_a.index), set(X_train_b.index))
        self.assertEqual(set(X_test_a.index), set(X_test_b.index))

        self.assertEqual(len(X_train_a), 320)
        self.assertEqual(len(X_test_a), 80)
        self.assertEqual(len(set(X_train_a.index) & set(X_test_a.index)), 0)

        for subset in (y_train_a, y_test_a):
            self.assertAlmostEqual(subset.mean(), self.y.mean(), delta=0.03)

        self.assertEqual(y_test_a.isna().sum(), 0)

    def test_pipeline_accepts_raw_rows_with_missing_values(self):
        _, X_test, _, _ = split_train_test(self.X, self.y)

        sample = X_test.head(2).copy().astype(object)
        sample.iloc[1] = np.nan
        before = sample.copy(deep=True)

        pipeline = make_pipeline(
            LogisticRegression(max_iter=1000, random_state=42), coding="ordinal"
        )
        X_train, _, y_train, _ = split_train_test(self.X, self.y)
        pipeline.fit(X_train.head(100), y_train.head(100))

        proba = pipeline.predict_proba(sample)

        self.assertEqual(proba.shape, (2, 2))
        self.assertTrue(np.isfinite(proba).all())
        self.assertTrue(((proba >= 0) & (proba <= 1)).all())
        np.testing.assert_allclose(proba.sum(axis=1), 1.0, rtol=1e-9)

        assert_frame_equal(before, sample)

        preds = pipeline.predict(sample)
        self.assertTrue(set(np.unique(preds)) <= {0, 1})

    def test_pipeline_without_id_column_still_works(self):
        X_train, X_test, y_train, _ = split_train_test(self.X, self.y)
        pipeline = make_pipeline(
            LogisticRegression(max_iter=1000, random_state=42), coding="onehot"
        )
        pipeline.fit(X_train.head(100), y_train.head(100))
        proba = pipeline.predict_proba(X_test.head(5).drop(columns="id"))
        self.assertEqual(proba.shape, (5, 2))

    @unittest.skipUnless(
        (MODELS_DIR / "pipeline.joblib").exists()
        and (MODELS_DIR / "metadata.json").exists(),
        "chưa chạy python -m src.train nên chưa có models/pipeline.joblib",
    )
    def test_saved_pipeline_roundtrip(self):
        pipeline = joblib.load(MODELS_DIR / "pipeline.joblib")
        metadata = json.loads(
            (MODELS_DIR / "metadata.json").read_text(encoding="utf-8")
        )

        _, X_test, _, y_test = split_train_test(self.X, self.y)
        sample = X_test.head(10)

        proba_before = pipeline.predict_proba(sample)
        proba_after = joblib.load(MODELS_DIR / "pipeline.joblib").predict_proba(sample)
        np.testing.assert_allclose(proba_before, proba_after)

        self.assertIn(metadata["model"], ("logreg", "knn", "dt", "rf"))
        self.assertIn(metadata["coding"], ("ordinal", "onehot"))
        self.assertGreaterEqual(metadata["cv_recall_ckd_mean"], 0.0)
        self.assertLessEqual(metadata["cv_recall_ckd_mean"], 1.0)
        self.assertEqual(metadata["n_test"], len(X_test))

        preds = pipeline.predict(X_test)
        self.assertEqual(len(preds), len(y_test))
        self.assertFalse(np.isnan(preds).any())


if __name__ == "__main__":
    unittest.main(verbosity=2)
