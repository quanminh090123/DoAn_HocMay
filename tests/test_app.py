
import unittest
import joblib
import numpy as np
import pandas as pd

from pathlib import Path
from src.config import FEATURE_COLUMNS, RAW_DATA_PATH
from src.preprocess import clean_features


ROOT = Path(__file__).resolve().parents[1]
MODEL_PATH = ROOT / "models" / "pipeline.joblib"


class TestKidneyApplication(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        # Su dung du lieu va mo hinh cua nhom
        cls.model = joblib.load(MODEL_PATH)

        raw = pd.read_csv(RAW_DATA_PATH)

        # Loai nhan thuc te, chi giu dau vao
        cls.X = raw.loc[:, list(FEATURE_COLUMNS)]

    def test_01_du_doan_hop_le(self):
        sample = self.X.head(5)
        predictions = self.model.predict(sample)

        self.assertEqual(len(predictions), 5)
        self.assertTrue(set(predictions).issubset({0, 1}))

    def test_02_thieu_cot_age(self):
        sample = self.X.head(1).drop(columns=["age"])

        with self.assertRaises(ValueError):
            clean_features(sample)

    def test_03_tuoi_la_chu(self):
        sample = self.X.head(1).copy()
        sample["age"] = "abc"

        with self.assertRaises(ValueError):
            clean_features(sample)

    def test_04_huyet_ap_am(self):
        sample = self.X.head(1).copy()
        sample["bp"] = -80

        # Giong quy tac kiem tra so am tren website
        cleaned = clean_features(sample)

        self.assertTrue(
            (cleaned["bp"].dropna() < 0).any()
        )

    def test_05_gia_tri_phan_loai_sai(self):
        sample = self.X.head(1).copy()
        sample["rbc"] = "sai_gia_tri"

        with self.assertRaises(ValueError):
            clean_features(sample)

    def test_06_thieu_mot_gia_tri(self):
        sample = self.X.head(1).copy()
        sample["age"] = np.nan

        predictions = self.model.predict(sample)

        self.assertEqual(len(predictions), 1)
        self.assertIn(predictions[0], [0, 1])

    def test_07_du_doan_lap_lai(self):
        sample = self.X.head(5)

        result1 = self.model.predict(sample)
        result2 = self.model.predict(sample)

        np.testing.assert_array_equal(
            result1, result2
        )

    def test_08_xac_suat_hop_le(self):
        sample = self.X.head(5)

        probabilities = self.model.predict_proba(sample)

        self.assertEqual(probabilities.shape, (5, 2))

        self.assertTrue(
            np.all((probabilities >= 0) &
                   (probabilities <= 1))
        )

        np.testing.assert_allclose(
            probabilities.sum(axis=1),
            np.ones(5)
        )


if __name__ == "__main__":
    unittest.main()
