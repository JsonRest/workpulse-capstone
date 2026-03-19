"""Unit tests for WorkPulse data pipeline."""

import numpy as np
import pandas as pd
import pytest
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.data_pipeline import (
    generate_dataset,
    split_and_scale,
    FEATURE_COLS,
    TARGET_COL,
)


class TestGenerateDataset:
    """Tests for the data generation function."""

    def test_default_shape(self):
        df = generate_dataset(n=1000)
        assert df.shape[0] == 1000
        assert df.shape[1] == len(FEATURE_COLS) + 1  # features + target

    def test_target_is_binary(self):
        df = generate_dataset(n=1000)
        assert set(df[TARGET_COL].unique()).issubset({0, 1})

    def test_target_balance(self):
        df = generate_dataset(n=10000)
        positive_rate = df[TARGET_COL].mean()
        assert 0.30 < positive_rate < 0.40, f"Expected ~35%, got {positive_rate:.2%}"

    def test_no_missing_values(self):
        df = generate_dataset(n=1000)
        assert df.isnull().sum().sum() == 0

    def test_feature_ranges(self):
        df = generate_dataset(n=5000)
        assert df["overtime_index"].between(0, 1).all()
        assert df["wellbeing_composite"].between(0, 1).all()
        assert df["high_stress_flag"].isin([0, 1]).all()
        assert df["tenure_risk_flag"].isin([0, 1]).all()
        assert df["age_group"].isin([0, 1, 2, 3]).all()

    def test_reproducibility(self):
        df1 = generate_dataset(n=100, seed=42)
        df2 = generate_dataset(n=100, seed=42)
        pd.testing.assert_frame_equal(df1, df2)

    def test_different_seeds_differ(self):
        df1 = generate_dataset(n=100, seed=42)
        df2 = generate_dataset(n=100, seed=99)
        assert not df1.equals(df2)


class TestSplitAndScale:
    """Tests for the train/test split and scaling function."""

    def test_split_sizes(self):
        df = generate_dataset(n=1000)
        X_train, X_test, y_train, y_test, _, _, _ = split_and_scale(df)
        assert X_train.shape[0] == 800
        assert X_test.shape[0] == 200

    def test_stratified_split(self):
        df = generate_dataset(n=2000)
        _, _, y_train, y_test, _, _, _ = split_and_scale(df)
        train_rate = y_train.mean()
        test_rate = y_test.mean()
        assert abs(train_rate - test_rate) < 0.02, "Split should be stratified"

    def test_scaler_fitted_on_train(self):
        df = generate_dataset(n=1000)
        _, _, _, _, X_train_sc, _, _ = split_and_scale(df)
        assert abs(X_train_sc.mean()) < 0.05, "Scaled train should have mean ≈ 0"
        assert abs(X_train_sc.std() - 1.0) < 0.05, "Scaled train should have std ≈ 1"

    def test_feature_count(self):
        df = generate_dataset(n=1000)
        X_train, X_test, _, _, _, _, _ = split_and_scale(df)
        assert X_train.shape[1] == len(FEATURE_COLS)
        assert X_test.shape[1] == len(FEATURE_COLS)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
