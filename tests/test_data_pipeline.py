"""Unit tests for WorkPulse data pipeline."""
import numpy as np, sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from src.data_pipeline import generate_dataset, split_and_scale, FEATURE_COLS, TARGET_COL

class TestGenerateDataset:
    def test_shape(self):
        df = generate_dataset(n=1000)
        assert df.shape[0] == 1000
        assert df.shape[1] == len(FEATURE_COLS) + 1

    def test_binary_target(self):
        df = generate_dataset(n=1000)
        assert set(df[TARGET_COL].unique()).issubset({0, 1})

    def test_target_balance(self):
        df = generate_dataset(n=10000)
        assert 0.30 < df[TARGET_COL].mean() < 0.40

    def test_no_nulls(self):
        df = generate_dataset(n=1000)
        assert df.isnull().sum().sum() == 0

    def test_feature_ranges(self):
        df = generate_dataset(n=5000)
        assert df["overtime_index"].between(0, 1).all()
        assert df["high_stress_flag"].isin([0, 1]).all()
        assert df["age_group"].isin([0, 1, 2, 3]).all()

    def test_reproducibility(self):
        import pandas as pd
        df1 = generate_dataset(n=100, seed=42)
        df2 = generate_dataset(n=100, seed=42)
        pd.testing.assert_frame_equal(df1, df2)

    def test_different_seeds(self):
        df1 = generate_dataset(n=100, seed=42)
        df2 = generate_dataset(n=100, seed=99)
        assert not df1.equals(df2)

class TestSplitAndScale:
    def test_split_sizes(self):
        df = generate_dataset(n=1000)
        X_tr, X_te, y_tr, y_te, _, _, _ = split_and_scale(df)
        assert X_tr.shape[0] == 800
        assert X_te.shape[0] == 200

    def test_stratified(self):
        df = generate_dataset(n=2000)
        _, _, y_tr, y_te, _, _, _ = split_and_scale(df)
        assert abs(y_tr.mean() - y_te.mean()) < 0.02

    def test_scaler(self):
        df = generate_dataset(n=1000)
        _, _, _, _, X_sc, _, _ = split_and_scale(df)
        assert abs(X_sc.mean()) < 0.05

    def test_feature_count(self):
        df = generate_dataset(n=1000)
        X_tr, _, _, _, _, _, _ = split_and_scale(df)
        assert X_tr.shape[1] == len(FEATURE_COLS)

if __name__ == "__main__":
    import pytest
    pytest.main([__file__, "-v"])
